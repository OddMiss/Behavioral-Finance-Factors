import rqdatac
rqdatac.init()
import pandas as pd
import rqdatac
from pathlib import Path
from math import lgamma
import json
import numpy as np

def _slsv_adjustment_factor(n_funds, market_buy_ratio):
    """
    AF(i,t) = E[|K / N - p_t|]
    K ~ Binomial(N, p_t)

    注意：这里包含 K=0 项，符合完整的二项分布期望。
    """
    if n_funds <= 0:
        return np.nan
    if market_buy_ratio <= 0.0 or market_buy_ratio >= 1.0:
        return 0.0
    k = np.arange(n_funds + 1, dtype=float)
    log_combination = np.array([lgamma(n_funds + 1) - lgamma(int(value) + 1) - lgamma(n_funds - int(value) + 1) for value in k])
    log_probability = log_combination + k * np.log(market_buy_ratio) + (n_funds - k) * np.log(1.0 - market_buy_ratio)
    probability = np.exp(log_probability)
    return np.sum(probability * np.abs(k / n_funds - market_buy_ratio))

def _slsv_get_report_files(holdings_dir, report_months=(6, 12)):
    """
    返回：
    {report_date: parquet_file_path}
    """
    report_files = {}
    for file_path in Path(holdings_dir).glob('*.parquet'):
        try:
            report_date = pd.Timestamp(file_path.stem)
        except ValueError:
            continue
        if report_date.month in report_months:
            report_files[report_date] = file_path
    if not report_files:
        raise FileNotFoundError('未在 {} 找到报告期持仓文件。'.format(holdings_dir))
    return report_files

def _slsv_read_holding_snapshot(file_path):
    """
    读取一个报告期的基金 A 股持仓，并聚合重复记录。
    """
    required_columns = ['fund_id', 'report_date', 'release_date', 'order_book_id', 'shares', 'type', 'category']
    data = pd.read_parquet(file_path, columns=required_columns)
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError('{} 缺少字段：{}'.format(file_path.name, sorted(missing_columns)))
    data['report_date'] = pd.to_datetime(data['report_date'])
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['shares'] = pd.to_numeric(data['shares'], errors='coerce')
    data = data[data['type'].eq('Stock') & data['category'].astype(str).str.lower().eq('ashare')]
    data = data.dropna(subset=['fund_id', 'report_date', 'release_date', 'order_book_id', 'shares'])
    data = data.groupby(['fund_id', 'report_date', 'order_book_id'], as_index=False).agg(shares=('shares', 'sum'), release_date=('release_date', 'max'))
    return data

def _slsv_load_trading_dates(indicators_dir):
    """
    从 comp_con_pe_ftm 文件名提取交易日。

    若目录中存在更多指标文件，也可将 glob 改为 rglob。
    """
    fpe_dir = Path(indicators_dir) / 'comp_con_pe_ftm'
    trading_dates = []
    for file_path in fpe_dir.glob('*.parquet'):
        try:
            trading_dates.append(pd.Timestamp(file_path.stem))
        except ValueError:
            continue
    trading_dates = pd.DatetimeIndex(sorted(set(trading_dates)))
    if trading_dates.empty:
        raise RuntimeError('无法从 get_comp_indicators/comp_con_pe_ftm 提取交易日。')
    return trading_dates

def _slsv_build_report_factor(holdings_dir, report_months=(6, 12), share_tolerance=1e-08):
    """
    按相邻报告期计算 LSV 与 SLSV。

    输出字段：
    B, S, N, p_i, p_t, AF, LSV, SLSV,
    report_date, available_date
    """
    report_files = _slsv_get_report_files(holdings_dir=holdings_dir, report_months=report_months)
    report_dates = sorted(report_files.keys())
    if len(report_dates) < 2:
        raise RuntimeError('至少需要两个报告期的持仓数据。')
    factor_parts = []
    previous_date = report_dates[0]
    previous = _slsv_read_holding_snapshot(report_files[previous_date])
    for current_date in report_dates[1:]:
        current = _slsv_read_holding_snapshot(report_files[current_date])
        common_funds = np.intersect1d(previous['fund_id'].unique(), current['fund_id'].unique())
        if len(common_funds) > 0:
            previous_common = previous[previous['fund_id'].isin(common_funds)]
            current_common = current[current['fund_id'].isin(common_funds)]
            merged = pd.merge(current_common[['fund_id', 'order_book_id', 'shares']].rename(columns={'shares': 'shares_current'}), previous_common[['fund_id', 'order_book_id', 'shares']].rename(columns={'shares': 'shares_previous'}), on=['fund_id', 'order_book_id'], how='outer')
            merged['shares_current'] = merged['shares_current'].fillna(0.0)
            merged['shares_previous'] = merged['shares_previous'].fillna(0.0)
            holding_change = merged['shares_current'] - merged['shares_previous']
            buy_count = merged.loc[holding_change > share_tolerance].groupby('order_book_id')['fund_id'].nunique().rename('B')
            sell_count = merged.loc[holding_change < -share_tolerance].groupby('order_book_id')['fund_id'].nunique().rename('S')
            stock_changes = pd.concat([buy_count, sell_count], axis=1).fillna(0.0)
            stock_changes['N'] = stock_changes['B'] + stock_changes['S']
            stock_changes = stock_changes[stock_changes['N'] > 0].copy()
            if not stock_changes.empty:
                stock_changes['p_i'] = stock_changes['B'] / stock_changes['N']
                market_buy_ratio = stock_changes['p_i'].mean()
                stock_changes['p_t'] = market_buy_ratio
                af_map = {int(n_funds): _slsv_adjustment_factor(n_funds=int(n_funds), market_buy_ratio=market_buy_ratio) for n_funds in stock_changes['N'].unique()}
                stock_changes['AF'] = stock_changes['N'].map(af_map)
                stock_changes['LSV'] = (stock_changes['p_i'] - market_buy_ratio).abs() - stock_changes['AF']
                stock_changes['SLSV'] = stock_changes['LSV'] * np.sign(stock_changes['p_i'] - market_buy_ratio)
                stock_changes = stock_changes.reset_index().rename(columns={'index': 'order_book_id'})
                stock_changes['report_date'] = current_date
                stock_changes['available_date'] = current_common['release_date'].max()
                factor_parts.append(stock_changes)
        previous_date = current_date
        previous = current
    if not factor_parts:
        raise RuntimeError('没有生成有效的 SLSV 报告期快照。')
    return pd.concat(factor_parts, ignore_index=True)

def calculate_slsv_daily_from_local(start_date, end_date, candidate_ids=None, data_dir='data', report_months=(6, 12), share_tolerance=1e-08):
    """
    使用本地数据计算日频 SLSV。

    Returns
    -------
    pd.DataFrame
        index  : [date, order_book_id]
        column : SLSV
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    data_dir = Path(data_dir)
    holdings_dir = data_dir / 'fund_holdings'
    indicators_dir = data_dir / 'get_comp_indicators'
    if not holdings_dir.exists():
        raise FileNotFoundError('未找到目录：{}'.format(holdings_dir))
    if not indicators_dir.exists():
        raise FileNotFoundError('未找到目录：{}'.format(indicators_dir))
    report_factor = _slsv_build_report_factor(holdings_dir=holdings_dir, report_months=report_months, share_tolerance=share_tolerance)
    trading_dates = _slsv_load_trading_dates(indicators_dir=indicators_dir)

    def next_trading_date(date):
        position = trading_dates.searchsorted(pd.Timestamp(date), side='right')
        if position >= len(trading_dates):
            return pd.NaT
        return trading_dates[position]
    report_factor['signal_date'] = report_factor['available_date'].map(next_trading_date)
    report_factor = report_factor.dropna(subset=['signal_date'])
    report_factor = report_factor[report_factor['signal_date'] <= end_date].copy()
    if candidate_ids is None:
        candidate_ids = report_factor['order_book_id'].drop_duplicates().sort_values().tolist()
    elif isinstance(candidate_ids, str):
        candidate_ids = [candidate_ids]
    else:
        candidate_ids = list(dict.fromkeys(candidate_ids))
    report_factor = report_factor[report_factor['order_book_id'].isin(candidate_ids)].copy()
    report_factor = report_factor.sort_values(['signal_date', 'report_date']).drop_duplicates(subset=['signal_date', 'order_book_id'], keep='last')
    snapshot_wide = report_factor.pivot(index='signal_date', columns='order_book_id', values='SLSV')
    target_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
    if target_dates.empty:
        raise ValueError('所选区间内没有交易日。')
    all_dates = target_dates.union(snapshot_wide.index).sort_values()
    daily_wide = snapshot_wide.reindex(index=all_dates, columns=candidate_ids).ffill().reindex(target_dates)
    factor_df = daily_wide.stack(dropna=True).rename('SLSV').to_frame().sort_index()
    factor_df.index = factor_df.index.set_names(['date', 'order_book_id'])
    return factor_df
import pandas as pd
from pathlib import Path
import json
import pandas as pd

def save_result_item(name, value):
    """将米筐因子回测结果中的可用数据保存为 CSV。"""
    if value is None:
        print(f'跳过 {name}：结果为空')
        return
    if isinstance(value, pd.Series):
        value.rename(value.name or name).to_csv(save_dir / f'{name}.csv', index_label='date')
    elif isinstance(value, pd.DataFrame):
        value.to_csv(save_dir / f'{name}.csv', index=True)
    elif isinstance(value, dict):
        pd.Series(value, name='value').to_csv(save_dir / f'{name}.csv', index_label='metric')
    else:
        print(f'跳过 {name}：类型为 {type(value)}，不是表格数据')
        return
    print(f'已保存：{name}.csv')
from _factor_io import save_from_cli

def build_factor(start_date, end_date, data_dir):
    return calculate_slsv_daily_from_local(start_date=start_date, end_date=end_date, candidate_ids=None, data_dir=data_dir, report_months=(6, 12), share_tolerance=1e-08)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='SLSV 模型（对称 LSV 模型）', factor_field='SLSV')
