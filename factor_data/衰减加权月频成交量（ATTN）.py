import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates
import rqdatac

def _get_ids(stock_ids=None):
    """默认返回米筐 CS 股票全集。"""
    if stock_ids is not None:
        return list(dict.fromkeys(stock_ids))
    return all_instruments('CS').order_book_id.dropna().drop_duplicates().tolist()

def _prepare_dates(start_date, end_date, lookback_days):
    """
    返回：
    target_dates：最终需要输出因子的交易日；
    price_dates：为计算滚动收益而额外向前展开后的价格日期。
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    calendar = pd.DatetimeIndex(get_trading_dates(start_date=start_date - pd.Timedelta(days=max(lookback_days * 3, 90)), end_date=end_date))
    target_dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if len(target_dates) == 0:
        raise ValueError('所选区间内没有交易日。')
    first_loc = calendar.get_loc(target_dates[0])
    last_loc = calendar.get_loc(target_dates[-1])
    if first_loc < lookback_days:
        raise ValueError('历史交易日不足，无法满足滚动窗口。')
    price_dates = calendar[first_loc - lookback_days:last_loc + 1]
    return (target_dates, price_dates)

def _get_batch_close_volume(batch, price_dates):
    """
    米筐多标的查询不能使用 skip_suspended=True；
    因此下载成交量，并用 volume > 0 排除当日停牌股票。
    """
    raw = get_price(order_book_ids=batch, start_date=price_dates[0], end_date=price_dates[-1], frequency='1d', fields=['close', 'volume'], adjust_type='pre', skip_suspended=False, expect_df=True)
    if raw is None or raw.empty:
        return (None, None)
    if isinstance(raw.index, pd.MultiIndex):
        close = raw['close'].unstack('order_book_id')
        volume = raw['volume'].unstack('order_book_id')
    else:
        close = raw[['close']].rename(columns={'close': batch[0]})
        volume = raw[['volume']].rename(columns={'volume': batch[0]})
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    return (close.reindex(index=price_dates, columns=batch), volume.reindex(index=price_dates, columns=batch))

def _stack_factor(factor_wide, factor_name):
    """日期 × 股票宽表转为 date、order_book_id 两层索引。"""
    panel = factor_wide.stack(dropna=True).rename(factor_name).to_frame()
    panel.index = panel.index.set_names(['date', 'order_book_id'])
    return panel

def calculate_attn_daily(start_date, end_date, completed_months=12, candidate_ids=None, batch_size=500):
    """
    计算衰减加权月频成交量（ATTN），并映射为日频因子。

    对当前输出月 M，使用此前 12 个已完成月：
    M-1、M-2、...、M-12。

    权重为：
    12/78、11/78、...、1/78。

    因子在每月第一个交易日更新，月内保持不变。
    """
    if completed_months != 12:
        raise ValueError('当前版本严格对应 12 个月线性衰减，completed_months 必须为 12。')
    required_lookback = 400
    target_dates, volume_dates = _prepare_dates(start_date=start_date, end_date=end_date, lookback_days=required_lookback)
    candidate_ids = _get_ids(candidate_ids)
    factor_parts = []
    weight_denominator = completed_months * (completed_months + 1) / 2.0
    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]
        _, volume = _get_batch_close_volume(batch, volume_dates)
        if volume is None:
            continue
        month_index = volume.index.to_period('M')
        monthly_volume = volume.groupby(month_index).sum()
        monthly_trade_days = volume.gt(0).groupby(month_index).sum()
        monthly_attn = pd.DataFrame(0.0, index=monthly_volume.index, columns=batch)
        valid_history = pd.DataFrame(True, index=monthly_volume.index, columns=batch)
        for lag in range(1, completed_months + 1):
            weight = (completed_months + 1 - lag) / weight_denominator
            monthly_attn += weight * monthly_volume.shift(lag)
            valid_history &= monthly_trade_days.shift(lag).gt(0)
        monthly_attn = monthly_attn.where(valid_history)
        daily_attn = monthly_attn.reindex(month_index)
        daily_attn.index = volume.index
        daily_attn = daily_attn.where(volume.gt(0))
        factor_wide = daily_attn.reindex(target_dates)
        factor_parts.append(_stack_factor(factor_wide, 'ATTN'))
    if not factor_parts:
        return pd.DataFrame(columns=['ATTN'])
    return pd.concat(factor_parts).sort_index()
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
    return calculate_attn_daily(start_date=start_date, end_date=end_date, candidate_ids=None, batch_size=500)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='衰减加权月频成交量（ATTN）', factor_field='ATTN')
