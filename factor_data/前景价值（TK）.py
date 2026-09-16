import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates
import rqdatac

def _tk_probability_weight(probability, parameter):
    """Tversky-Kahneman 概率权重函数。"""
    probability = np.asarray(probability, dtype=float)
    numerator = np.power(probability, parameter)
    denominator = np.power(np.power(probability, parameter) + np.power(1.0 - probability, parameter), 1.0 / parameter)
    return numerator / denominator

def _tk_value_from_return_window(return_window, alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69):
    """
    对一个“历史收益窗口 × 股票”的 ndarray 计算 TK。
    每一列是一只股票的过去 N 日收益率。
    """
    n_obs, n_stocks = return_window.shape
    result = np.full(n_stocks, np.nan)
    valid = np.isfinite(return_window).all(axis=0)
    if not valid.any():
        return result
    values = return_window[:, valid]
    loss_count = (values < 0).sum(axis=0)
    gain_count = (values > 0).sum(axis=0)
    loss_sorted = np.sort(np.where(values < 0, values, np.inf), axis=0)
    gain_sorted = np.sort(np.where(values > 0, values, np.inf), axis=0)
    tk_valid = np.zeros(values.shape[1])
    for count in np.unique(loss_count):
        if count == 0:
            continue
        columns = np.flatnonzero(loss_count == count)
        loss_values = loss_sorted[:count, columns]
        probabilities = np.arange(1, count + 1, dtype=float) / n_obs
        previous_probabilities = np.arange(0, count, dtype=float) / n_obs
        weights = _tk_probability_weight(probabilities, delta) - _tk_probability_weight(previous_probabilities, delta)
        loss_utility = -loss_aversion * np.power(-loss_values, alpha)
        tk_valid[columns] += (loss_utility * weights[:, None]).sum(axis=0)
    for count in np.unique(gain_count):
        if count == 0:
            continue
        columns = np.flatnonzero(gain_count == count)
        gain_values = gain_sorted[:count, columns]
        probabilities = np.arange(count, 0, -1, dtype=float) / n_obs
        previous_probabilities = np.arange(count - 1, -1, -1, dtype=float) / n_obs
        weights = _tk_probability_weight(probabilities, gamma) - _tk_probability_weight(previous_probabilities, gamma)
        gain_utility = np.power(gain_values, alpha)
        tk_valid[columns] += (gain_utility * weights[:, None]).sum(axis=0)
    result[valid] = tk_valid
    return result

def _tk_get_close_volume(batch, price_dates):
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

def calculate_tk_daily(start_date, end_date, window=60, candidate_ids=None, alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69, batch_size=500):
    """
    前景价值因子（TK）

    输出：
        index  : [date, order_book_id]
        column : TK

    注意：
        window=60 表示使用过去 60 个交易日收益率；
        因子在当日收盘后完整可得，应于下一交易日使用。
    """
    if window < 2:
        raise ValueError('window 至少应为 2。')
    if candidate_ids is None:
        candidate_ids = all_instruments('CS')['order_book_id'].dropna().drop_duplicates().tolist()
    else:
        candidate_ids = list(dict.fromkeys(candidate_ids))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    calendar = pd.DatetimeIndex(get_trading_dates(start_date=start_date - pd.Timedelta(days=max((window + 1) * 3, 120)), end_date=end_date))
    target_dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if len(target_dates) == 0:
        raise ValueError('所选区间内没有交易日。')
    first_loc = calendar.get_loc(target_dates[0])
    last_loc = calendar.get_loc(target_dates[-1])
    if first_loc < window:
        raise ValueError('历史交易日不足，无法计算 TK。')
    price_dates = calendar[first_loc - window:last_loc + 1]
    result_parts = []
    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]
        close, volume = _tk_get_close_volume(batch=batch, price_dates=price_dates)
        if close is None:
            continue
        stock_return = close.pct_change(fill_method=None).iloc[1:]
        stock_return = stock_return.where(volume.iloc[1:].gt(0))
        return_values = stock_return.values.astype(float)
        target_locs = stock_return.index.get_indexer(target_dates)
        tk_wide = pd.DataFrame(index=target_dates, columns=batch, dtype=float)
        for output_loc, return_loc in enumerate(target_locs):
            history_returns = return_values[return_loc - window + 1:return_loc + 1, :]
            tk_wide.iloc[output_loc] = _tk_value_from_return_window(return_window=history_returns, alpha=alpha, loss_aversion=loss_aversion, gamma=gamma, delta=delta)
        part = tk_wide.stack(dropna=True).rename('TK').to_frame()
        result_parts.append(part)
    if not result_parts:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=['date', 'order_book_id'])
        return pd.DataFrame(index=empty_index, columns=['TK'])
    result = pd.concat(result_parts).sort_index()
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
    return calculate_tk_daily(start_date=start_date, end_date=end_date, candidate_ids=None, window=60, alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69, batch_size=500)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='前景价值（TK）', factor_field='TK')
