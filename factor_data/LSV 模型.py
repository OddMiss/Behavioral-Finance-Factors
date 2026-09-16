import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from math import lgamma
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import pandas as pd
import zipfile
from pathlib import Path
from math import lgamma
import numpy as np
import pandas as pd

def _normalize_lsv_order_book_ids(raw_ids):
    """将本地持仓文件中的裸北交所代码规范为米筐合法代码。"""
    normalized = raw_ids.copy()
    has_value = normalized.notna()
    normalized.loc[has_value] = normalized.loc[has_value].astype(str).str.strip().str.upper()
    bare_bjse = normalized.astype(str).str.fullmatch('920\\d{3}')
    normalized.loc[bare_bjse] = normalized.loc[bare_bjse] + '.BJSE'
    return normalized

def _lsv_adjustment_factor(n_funds, market_buy_ratio):
    """
    AF(i,t) = E[|K / N - p_t|], K ~ Binomial(N, p_t)
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

def _prepare_local_data(data_zip_path='data.zip', extract_dir='data'):
    """
    若尚未解压，则将同目录 data.zip 解压至 data/。
    """
    data_zip_path = Path(data_zip_path)
    extract_dir = Path(extract_dir)
    if not extract_dir.exists():
        if not data_zip_path.exists():
            raise FileNotFoundError('未找到 data.zip，请确认其与 Notebook 位于同一目录。')
        with zipfile.ZipFile(data_zip_path, 'r') as zip_file:
            zip_file.extractall('.')
    holdings_dir = extract_dir / 'fund_holdings'
    indicators_dir = extract_dir / 'get_comp_indicators'
    if not holdings_dir.exists():
        raise FileNotFoundError('未找到 data/fund_holdings 目录。')
    if not indicators_dir.exists():
        raise FileNotFoundError('未找到 data/get_comp_indicators 目录。')
    return (holdings_dir, indicators_dir)

def _load_local_holdings(holdings_dir, report_months=(6, 12)):
    """
    读取本地基金持仓，并仅保留 A 股股票持仓。

    数据字段要求：
    fund_id, report_date, release_date,
    order_book_id, shares, type, category
    """
    files = sorted(Path(holdings_dir).glob('*.parquet'))
    if not files:
        raise RuntimeError('fund_holdings 目录中没有 parquet 文件。')
    required_columns = {'fund_id', 'report_date', 'release_date', 'order_book_id', 'shares', 'type', 'category'}
    parts = []
    for file_path in files:
        temp = pd.read_parquet(file_path)
        missing_columns = required_columns - set(temp.columns)
        if missing_columns:
            raise ValueError('{} 缺少字段：{}'.format(file_path.name, sorted(missing_columns)))
        temp['report_date'] = pd.to_datetime(temp['report_date'])
        temp['release_date'] = pd.to_datetime(temp['release_date'])
        temp['shares'] = pd.to_numeric(temp['shares'], errors='coerce')
        temp['order_book_id'] = _normalize_lsv_order_book_ids(temp['order_book_id'])
        temp = temp[temp['report_date'].dt.month.isin(report_months)]
        legal_cs_id = temp['order_book_id'].astype(str).str.fullmatch('(?:(?:0|3)\\d{5}\\.XSHE|6\\d{5}\\.XSHG|920\\d{3}\\.BJSE)', na=False)
        temp = temp[legal_cs_id].copy()
        temp = temp[temp['type'].eq('Stock') & temp['category'].astype(str).str.lower().eq('ashare')]
        temp = temp.dropna(subset=['fund_id', 'report_date', 'release_date', 'order_book_id', 'shares'])
        parts.append(temp[['fund_id', 'report_date', 'release_date', 'order_book_id', 'shares']])
    holdings = pd.concat(parts, ignore_index=True)
    holdings = holdings.groupby(['fund_id', 'report_date', 'order_book_id'], as_index=False).agg(shares=('shares', 'sum'), release_date=('release_date', 'max'))
    return holdings

def _load_local_trading_dates(indicators_dir):
    """
    从 get_comp_indicators 的文件名中提取交易日。

    文件结构示例：
    data/get_comp_indicators/xxx/2025-01-02.parquet
    """
    trading_dates = []
    for file_path in Path(indicators_dir).rglob('*.parquet'):
        try:
            trading_dates.append(pd.Timestamp(file_path.stem))
        except ValueError:
            continue
    trading_dates = pd.DatetimeIndex(sorted(set(trading_dates)))
    if trading_dates.empty:
        raise RuntimeError('无法从 get_comp_indicators 中提取交易日。')
    return trading_dates

def _build_lsv_report_factor(holdings, share_tolerance=1e-08, min_funds_per_stock=10):
    """
    基于相邻报告期持仓构造报告期 LSV 横截面。
    """
    if min_funds_per_stock < 1:
        raise ValueError('min_funds_per_stock 至少应为 1。')
    report_dates = sorted(holdings['report_date'].drop_duplicates().tolist())
    factor_parts = []
    for previous_date, current_date in zip(report_dates[:-1], report_dates[1:]):
        previous = holdings[holdings['report_date'] == previous_date].copy()
        current = holdings[holdings['report_date'] == current_date].copy()
        common_funds = np.intersect1d(previous['fund_id'].unique(), current['fund_id'].unique())
        if len(common_funds) == 0:
            continue
        previous = previous[previous['fund_id'].isin(common_funds)]
        current = current[current['fund_id'].isin(common_funds)]
        merged = pd.merge(current[['fund_id', 'order_book_id', 'shares']].rename(columns={'shares': 'shares_current'}), previous[['fund_id', 'order_book_id', 'shares']].rename(columns={'shares': 'shares_previous'}), on=['fund_id', 'order_book_id'], how='outer')
        merged['shares_current'] = merged['shares_current'].fillna(0.0)
        merged['shares_previous'] = merged['shares_previous'].fillna(0.0)
        holding_change = merged['shares_current'] - merged['shares_previous']
        buy_count = merged.loc[holding_change > share_tolerance].groupby('order_book_id')['fund_id'].nunique().rename('B')
        sell_count = merged.loc[holding_change < -share_tolerance].groupby('order_book_id')['fund_id'].nunique().rename('S')
        stock_changes = pd.concat([buy_count, sell_count], axis=1).fillna(0.0)
        stock_changes['N'] = stock_changes['B'] + stock_changes['S']
        stock_changes = stock_changes[stock_changes['N'] > 0].copy()
        if stock_changes.empty:
            continue
        market_buy_count = stock_changes['B'].sum()
        market_trade_count = stock_changes['N'].sum()
        market_buy_ratio = market_buy_count / market_trade_count
        stock_changes = stock_changes[stock_changes['N'] >= min_funds_per_stock].copy()
        if stock_changes.empty:
            continue
        stock_changes['p_i'] = stock_changes['B'] / stock_changes['N']
        stock_changes['p_t'] = market_buy_ratio
        stock_changes['market_B_total'] = market_buy_count
        stock_changes['market_N_total'] = market_trade_count
        af_map = {n_funds: _lsv_adjustment_factor(int(n_funds), market_buy_ratio) for n_funds in stock_changes['N'].unique()}
        stock_changes['AF'] = stock_changes['N'].map(af_map)
        stock_changes['LSV'] = (stock_changes['p_i'] - market_buy_ratio).abs() - stock_changes['AF']
        stock_changes = stock_changes.reset_index().rename(columns={'index': 'order_book_id'})
        stock_changes['report_date'] = current_date
        stock_changes['available_date'] = current['release_date'].max()
        factor_parts.append(stock_changes)
    if not factor_parts:
        raise RuntimeError('没有足够的相邻报告期用于计算 LSV。')
    return pd.concat(factor_parts, ignore_index=True)

def calculate_lsv_daily_from_local(start_date, end_date, candidate_ids=None, data_zip_path='data.zip', extract_dir='data', report_months=(6, 12), share_tolerance=1e-08, min_funds_per_stock=10):
    """
    使用本地压缩包构建日频 LSV 因子。

    Parameters
    ----------
    start_date, end_date : str
        因子输出的起止日期。
    candidate_ids : list[str] or str, optional
        指定股票池；默认为全部发生基金持仓变化的 A 股。
    report_months : tuple
        默认使用中报和年报：(6, 12)。
        若需要季频持仓比较，使用：(3, 6, 9, 12)。

    Returns
    -------
    pd.DataFrame
        MultiIndex: [date, order_book_id]
        Column: LSV
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    holdings_dir, indicators_dir = _prepare_local_data(data_zip_path=data_zip_path, extract_dir=extract_dir)
    holdings = _load_local_holdings(holdings_dir=holdings_dir, report_months=report_months)
    report_factor = _build_lsv_report_factor(holdings=holdings, share_tolerance=share_tolerance, min_funds_per_stock=min_funds_per_stock)
    trading_dates = _load_local_trading_dates(indicators_dir=indicators_dir)

    def next_trading_date(date):
        position = trading_dates.searchsorted(pd.Timestamp(date), side='right')
        if position >= len(trading_dates):
            return pd.NaT
        return trading_dates[position]
    report_factor['signal_date'] = report_factor['available_date'].map(next_trading_date)
    report_factor = report_factor.dropna(subset=['signal_date'])
    if candidate_ids is None:
        candidate_ids = report_factor['order_book_id'].drop_duplicates().sort_values().tolist()
    elif isinstance(candidate_ids, str):
        candidate_ids = [candidate_ids]
    else:
        candidate_ids = list(dict.fromkeys(candidate_ids))
    report_factor = report_factor[report_factor['order_book_id'].isin(candidate_ids)].copy()
    report_factor = report_factor.sort_values(['signal_date', 'report_date']).drop_duplicates(subset=['signal_date', 'order_book_id'], keep='last')
    snapshot_wide = report_factor.pivot(index='signal_date', columns='order_book_id', values='LSV')
    all_dates = trading_dates[(trading_dates >= snapshot_wide.index.min()) & (trading_dates <= end_date)]
    daily_wide = pd.DataFrame(np.nan, index=all_dates, columns=candidate_ids, dtype=float)
    snapshot_dates = snapshot_wide.index.sort_values()
    for pos, signal_date in enumerate(snapshot_dates):
        next_signal_date = snapshot_dates[pos + 1] if pos + 1 < len(snapshot_dates) else pd.Timestamp.max
        effective_dates = all_dates[(all_dates >= signal_date) & (all_dates < next_signal_date)]
        daily_wide.loc[effective_dates, snapshot_wide.columns] = snapshot_wide.loc[signal_date].values
    daily_wide = daily_wide.loc[(daily_wide.index >= start_date) & (daily_wide.index <= end_date)]
    result = daily_wide.stack(dropna=True).rename('LSV').to_frame()
    result.index = result.index.set_names(['date', 'order_book_id'])
    return (result, report_factor)
from _factor_io import save_from_cli

def build_factor(start_date, end_date, data_dir):
    return calculate_lsv_daily_from_local(start_date=start_date, end_date=end_date, candidate_ids=None, data_zip_path='', extract_dir=data_dir, report_months=(6, 12), share_tolerance=1e-08, min_funds_per_stock=10)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='LSV 模型', factor_field='LSV')
