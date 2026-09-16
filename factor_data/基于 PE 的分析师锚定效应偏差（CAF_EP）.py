import rqdatac
rqdatac.init()
from pathlib import Path
import numpy as np
import pandas as pd
import rqdatac
FPE_FIELD = 'comp_con_pe_ftm'
INDUSTRY_FIELD = 'first_industry_code'

def _get_dated_files(directory):
    """
    读取 YYYY-MM-DD.parquet 文件并返回：
    {Timestamp: Path}
    """
    directory = Path(directory)
    result = {}
    for file_path in directory.glob('*.parquet'):
        try:
            result[pd.Timestamp(file_path.stem)] = file_path
        except ValueError:
            continue
    if not result:
        raise FileNotFoundError('目录中未找到日期命名的 parquet 文件：{}'.format(directory))
    return result

def _read_positive_fpe(file_path, calculation_ids=None):
    """
    读取单日预测 PE。

    非正预测 PE 无经济含义，按缺失值处理；
    后续会以前一有效值向前填充。
    """
    data = pd.read_parquet(file_path, columns=['order_book_id', FPE_FIELD])
    required_columns = {'order_book_id', FPE_FIELD}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError('{} 缺少字段：{}'.format(file_path.name, sorted(missing_columns)))
    data[FPE_FIELD] = pd.to_numeric(data[FPE_FIELD], errors='coerce')
    data = data.loc[data[FPE_FIELD] > 0, ['order_book_id', FPE_FIELD]]
    if calculation_ids is not None:
        data = data[data['order_book_id'].isin(calculation_ids)]
    return data.drop_duplicates(subset=['order_book_id'], keep='last').set_index('order_book_id')[FPE_FIELD]

def _read_industry(file_path):
    """
    读取单个行业分类快照。

    本地数据中使用一级中信行业：
    first_industry_code
    """
    data = pd.read_parquet(file_path, columns=['order_book_id', INDUSTRY_FIELD])
    required_columns = {'order_book_id', INDUSTRY_FIELD}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError('{} 缺少字段：{}'.format(file_path.name, sorted(missing_columns)))
    return data.dropna(subset=['order_book_id']).drop_duplicates(subset=['order_book_id'], keep='last').set_index('order_book_id')[INDUSTRY_FIELD]

def _normalize_ids(ids):
    if ids is None:
        return None
    if isinstance(ids, str):
        return {ids}
    return set(dict.fromkeys(ids))

def calculate_caf_ep_daily_from_local(start_date, end_date, candidate_ids=None, industry_universe_ids=None, data_dir='data', min_industry_stocks=5, initial_lookback_days=370):
    """
    基于本地数据构建日频 CAF_EP 因子。

    Parameters
    ----------
    start_date, end_date : str
        输出因子的起止日期。

    candidate_ids : list[str] or str, optional
        因子输出股票池。None 表示所有可计算股票。

    industry_universe_ids : list[str] or str, optional
        计算行业预测 PE 中位数的股票池。
        None 表示全部可取得有效预测 PE 的股票。

    data_dir : str
        本地数据目录，默认 data。

    min_industry_stocks : int
        行业内当日有效预测 PE 股票数下限。
        少于该数量的行业不输出因子值。

    initial_lookback_days : int
        向前读取预测 PE 的自然日窗口，用于前向填充。

    Returns
    -------
    pd.DataFrame
        MultiIndex: [date, order_book_id]
        Column: CAF_EP
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    if start_date > end_date:
        raise ValueError('start_date 不能晚于 end_date。')
    if min_industry_stocks < 1:
        raise ValueError('min_industry_stocks 至少应为 1。')
    data_dir = Path(data_dir)
    fpe_dir = data_dir / 'get_comp_indicators' / FPE_FIELD
    industry_dir = data_dir / 'get_instrument_industry_monthly'
    fpe_files = _get_dated_files(fpe_dir)
    industry_files = _get_dated_files(industry_dir)
    fpe_dates = pd.DatetimeIndex(sorted(fpe_files.keys()))
    industry_dates = pd.DatetimeIndex(sorted(industry_files.keys()))
    candidate_set = _normalize_ids(candidate_ids)
    industry_universe_set = _normalize_ids(industry_universe_ids)
    if candidate_set is None and industry_universe_set is None:
        calculation_ids = None
    else:
        calculation_ids = (candidate_set or set()) | (industry_universe_set or set())
    lookback_start = start_date - pd.Timedelta(days=initial_lookback_days)
    process_dates = fpe_dates[(fpe_dates >= lookback_start) & (fpe_dates <= end_date)]
    target_dates = process_dates[process_dates >= start_date]
    if len(target_dates) == 0:
        raise ValueError('所选区间内没有可用的预测 PE 交易日。')
    last_valid_fpe = {}
    current_industry = pd.Series(dtype=object)
    current_industry_position = None
    result_parts = []
    for date in process_dates:
        daily_fpe = _read_positive_fpe(file_path=fpe_files[date], calculation_ids=calculation_ids)
        if not daily_fpe.empty:
            last_valid_fpe.update(daily_fpe.to_dict())
        if date < start_date or not last_valid_fpe:
            continue
        industry_position = industry_dates.searchsorted(date, side='right') - 1
        if industry_position < 0:
            continue
        if industry_position != current_industry_position:
            current_industry_position = industry_position
            current_industry = _read_industry(industry_files[industry_dates[industry_position]])
        fpe = pd.Series(last_valid_fpe, name='FPE', dtype=float)
        frame = pd.DataFrame({'FPE': fpe, 'industry': current_industry.reindex(fpe.index)})
        frame = frame.dropna(subset=['FPE', 'industry'])
        if industry_universe_set is None:
            median_universe = frame
        else:
            median_universe = frame.loc[frame.index.isin(industry_universe_set)]
        if median_universe.empty:
            continue
        industry_median_fpe = median_universe.groupby('industry')['FPE'].median()
        industry_fpe_count = median_universe.groupby('industry')['FPE'].count()
        eligible_industries = industry_fpe_count[industry_fpe_count >= min_industry_stocks].index
        if len(eligible_industries) == 0:
            continue
        frame = frame.loc[frame['industry'].isin(eligible_industries)].copy()
        frame['industry_median_fpe'] = frame['industry'].map(industry_median_fpe)
        frame['CAF_EP'] = frame['industry_median_fpe'] / frame['FPE'] - 1.0
        if candidate_set is not None:
            frame = frame.loc[frame.index.isin(candidate_set)]
        if frame.empty:
            continue
        result_part = frame[['CAF_EP']].reset_index().rename(columns={'index': 'order_book_id'})
        result_part.insert(0, 'date', date)
        result_parts.append(result_part)
    if not result_parts:
        raise RuntimeError('未生成 CAF_EP 因子，请检查数据路径、行业分类覆盖与 min_industry_stocks 设置。')
    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(['date', 'order_book_id']).set_index(['date', 'order_book_id'])
    result.index = result.index.set_names(['date', 'order_book_id'])
    return result
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
    return calculate_caf_ep_daily_from_local(start_date=start_date, end_date=end_date, candidate_ids=None, industry_universe_ids=None, data_dir=data_dir, min_industry_stocks=5, initial_lookback_days=370)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='基于 PE 的分析师锚定效应偏差（CAF_EP）', factor_field='CAF_EP')
