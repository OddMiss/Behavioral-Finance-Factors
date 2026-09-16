import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_factor, get_trading_dates
import rqdatac

def _pb_tk_probability_weight(probability, parameter):
    """前景理论概率权重函数。"""
    probability = np.asarray(probability, dtype=float)
    numerator = np.power(probability, parameter)
    denominator = np.power(np.power(probability, parameter) + np.power(1.0 - probability, parameter), 1.0 / parameter)
    return numerator / denominator

def _pb_tk_value_from_window(value_changes, alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69):
    """
    对一个“过去 N 期 PB 变化率 × 股票”的数组计算 TK。
    每列对应一只股票。
    """
    n_obs, n_stocks = value_changes.shape
    result = np.full(n_stocks, np.nan)
    valid = np.isfinite(value_changes).all(axis=0)
    if not valid.any():
        return result
    values = value_changes[:, valid]
    loss_count = (values < 0).sum(axis=0)
    gain_count = (values > 0).sum(axis=0)
    loss_sorted = np.sort(np.where(values < 0, values, np.inf), axis=0)
    gain_sorted = np.sort(np.where(values > 0, values, np.inf), axis=0)
    tk_value = np.zeros(values.shape[1])
    for count in np.unique(loss_count):
        if count == 0:
            continue
        columns = np.flatnonzero(loss_count == count)
        loss_values = loss_sorted[:count, columns]
        probability = np.arange(1, count + 1, dtype=float) / n_obs
        previous_probability = np.arange(0, count, dtype=float) / n_obs
        weights = _pb_tk_probability_weight(probability, delta) - _pb_tk_probability_weight(previous_probability, delta)
        utility = -loss_aversion * np.power(-loss_values, alpha)
        tk_value[columns] += (utility * weights[:, None]).sum(axis=0)
    for count in np.unique(gain_count):
        if count == 0:
            continue
        columns = np.flatnonzero(gain_count == count)
        gain_values = gain_sorted[:count, columns]
        probability = np.arange(count, 0, -1, dtype=float) / n_obs
        previous_probability = np.arange(count - 1, -1, -1, dtype=float) / n_obs
        weights = _pb_tk_probability_weight(probability, gamma) - _pb_tk_probability_weight(previous_probability, gamma)
        utility = np.power(gain_values, alpha)
        tk_value[columns] += (utility * weights[:, None]).sum(axis=0)
    result[valid] = tk_value
    return result

def _pb_tk_get_pb_wide(order_book_ids, dates, pb_field='pb_ratio_lf', batch_size=500):
    """下载 PB 因子并整理为 日期 × 股票 宽表。"""
    parts = []
    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]
        raw = get_factor(order_book_ids=batch, factor=pb_field, start_date=dates[0], end_date=dates[-1], expect_df=True)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.index, pd.MultiIndex):
            pb_wide = raw[pb_field].unstack('order_book_id')
        else:
            pb_wide = raw[[pb_field]].rename(columns={pb_field: batch[0]})
        pb_wide.index = pd.to_datetime(pb_wide.index)
        parts.append(pb_wide.reindex(index=dates, columns=batch))
    if not parts:
        raise RuntimeError('未取得有效 PB 因子数据。')
    result = pd.concat(parts, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result.reindex(index=dates, columns=order_book_ids)

def calculate_pb_tk_daily(start_date, end_date, window=60, candidate_ids=None, pb_field='pb_ratio_lf', alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69, batch_size=500):
    """
    基于 PB 估值变化率的前景价值（TK）

    估值变化率：
        r_pb,t = PB_t / PB_(t-1) - 1

    输出：
        index  : [date, order_book_id]
        column : TK_PB

    window=60：
        使用过去 60 个交易日的 PB 估值变化率构造前景价值。
    """
    if window < 2:
        raise ValueError('window 至少应为 2。')
    if candidate_ids is None:
        candidate_ids = all_instruments('CS')['order_book_id'].dropna().drop_duplicates().tolist()
    elif isinstance(candidate_ids, str):
        candidate_ids = [candidate_ids]
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
        raise ValueError('历史交易日不足，无法计算 TK_PB。')
    pb_dates = calendar[first_loc - window:last_loc + 1]
    pb_wide = _pb_tk_get_pb_wide(order_book_ids=candidate_ids, dates=pb_dates, pb_field=pb_field, batch_size=batch_size)
    pb_wide = pb_wide.where(pb_wide > 0)
    pb_change = pb_wide.pct_change(fill_method=None).iloc[1:]
    pb_change_values = pb_change.values.astype(float)
    target_locs = pb_change.index.get_indexer(target_dates)
    factor_wide = pd.DataFrame(index=target_dates, columns=candidate_ids, dtype=float)
    for output_loc, change_loc in enumerate(target_locs):
        history_changes = pb_change_values[change_loc - window + 1:change_loc + 1, :]
        factor_wide.iloc[output_loc] = _pb_tk_value_from_window(value_changes=history_changes, alpha=alpha, loss_aversion=loss_aversion, gamma=gamma, delta=delta)
    result = factor_wide.stack(dropna=True).rename('TK_PB').to_frame()
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
    return calculate_pb_tk_daily(start_date=start_date, end_date=end_date, candidate_ids=None, window=60, pb_field='pb_ratio_lf', alpha=0.88, loss_aversion=2.25, gamma=0.61, delta=0.69, batch_size=500)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='基于 PB 的前景价值 TK', factor_field='TK_PB')
