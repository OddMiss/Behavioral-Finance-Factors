import rqdatac
rqdatac.init()
import pandas as pd
from rqdatac import all_instruments, get_trading_dates, get_turnover_rate
import rqdatac

def calculate_turnavg_daily(start_date, end_date, window=20, candidate_ids=None, batch_size=500, output='wide'):
    """
    平均换手率（TURNAVG）。

    每个交易日 t 使用 [t-window+1, t] 的日换手率均值。

    参数
    ----
    window : int
        滚动交易日窗口，默认 20，约为一个月。

    output : str
        "wide"  : date × order_book_id 宽表，可直接传入 factor_analysis；
        "panel" : [date, order_book_id] 两层索引、单列 TURNAVG。
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError('window 必须是正整数。')
    if output not in ('wide', 'panel'):
        raise ValueError('output 仅支持 "wide" 或 "panel"。')
    if candidate_ids is None:
        candidate_ids = all_instruments('CS')['order_book_id'].dropna().drop_duplicates().tolist()
    else:
        candidate_ids = list(pd.Index(candidate_ids).dropna().unique())
    if not candidate_ids:
        raise ValueError('candidate_ids 为空。')
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    calendar = pd.DatetimeIndex(pd.to_datetime(get_trading_dates(start_date=start_date - pd.Timedelta(days=max(window * 3, 90)), end_date=end_date)))
    target_dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if len(target_dates) == 0:
        raise ValueError('目标区间内没有交易日。')
    first_target_position = calendar.get_loc(target_dates[0])
    if first_target_position < window - 1:
        raise ValueError('历史交易日不足，无法计算完整 TURNAVG。')
    fetch_dates = calendar[first_target_position - window + 1:]
    turnover_parts = []
    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[start:start + batch_size]
        raw = get_turnover_rate(order_book_ids=batch_ids, start_date=fetch_dates[0], end_date=fetch_dates[-1], fields='today', expect_df=True)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.index, pd.MultiIndex):
            turnover = raw['today'].unstack('order_book_id')
        else:
            turnover = raw[['today']].rename(columns={'today': batch_ids[0]})
        turnover.index = pd.to_datetime(turnover.index)
        turnover_parts.append(turnover.reindex(index=fetch_dates, columns=batch_ids))
    if not turnover_parts:
        raise RuntimeError('未取得有效换手率数据，请检查接口权限。')
    turnover_wide = pd.concat(turnover_parts, axis=1)
    turnover_wide = turnover_wide.loc[:, ~turnover_wide.columns.duplicated()]
    turnover_wide = turnover_wide.reindex(index=fetch_dates, columns=candidate_ids).astype(float)
    turnavg_wide = turnover_wide.rolling(window=window, min_periods=window).mean()
    factor_df = turnavg_wide.reindex(target_dates)
    factor_df.index = pd.DatetimeIndex(factor_df.index)
    factor_df.index.name = 'date'
    factor_df.columns.name = 'order_book_id'
    if output == 'wide':
        return factor_df
    factor_panel = factor_df.stack(dropna=True).to_frame('TURNAVG')
    factor_panel.index.names = ['date', 'order_book_id']
    return factor_panel
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
    return calculate_turnavg_daily(start_date=start_date, end_date=end_date, candidate_ids=None, window=20, batch_size=500, output='wide')
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='平均换手率（TURNAVG）', factor_field='TURNAVG')
