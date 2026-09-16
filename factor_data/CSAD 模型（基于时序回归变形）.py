import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates
from rqdatac import index_components
import pandas as pd
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates

def _csad_ratio_ids(stock_ids):
    if stock_ids is None:
        return all_instruments('CS')['order_book_id'].dropna().drop_duplicates().tolist()
    if isinstance(stock_ids, str):
        return [stock_ids]
    return list(dict.fromkeys(stock_ids))

def _csad_ratio_get_returns_wide(order_book_ids, price_dates, batch_size=500):
    """下载复权收盘价并生成 日收益率宽表。"""
    parts = []
    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]
        raw = get_price(order_book_ids=batch, start_date=price_dates[0], end_date=price_dates[-1], frequency='1d', fields=['close', 'volume'], adjust_type='pre', skip_suspended=False, expect_df=True)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.index, pd.MultiIndex):
            close = raw['close'].unstack('order_book_id')
            volume = raw['volume'].unstack('order_book_id')
        else:
            close = raw[['close']].rename(columns={'close': batch[0]})
            volume = raw[['volume']].rename(columns={'volume': batch[0]})
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        close = close.reindex(index=price_dates, columns=batch)
        volume = volume.reindex(index=price_dates, columns=batch)
        stock_return = close.pct_change(fill_method=None).iloc[1:]
        stock_return = stock_return.where(volume.iloc[1:].gt(0))
        parts.append(stock_return)
    if not parts:
        raise RuntimeError('未取得有效价格数据。')
    returns = pd.concat(parts, axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()]
    return returns.reindex(index=price_dates[1:], columns=order_book_ids)

def _csad_ratio_month_end_dates(dates):
    """取每个月的最后一个交易日。"""
    month_index = pd.Index(dates.to_period('M'))
    return dates[~month_index.duplicated(keep='last')]

def _csad_ratio_select_boards(returns, rebalance_date, candidate_ids, corr_window=120, max_inactive_days=20, group_size=10):
    """
    在月末为每个中心股票选择 group_size - 1 个最高相关股票。
    """
    history = returns.loc[:rebalance_date].tail(corr_window)
    min_observations = corr_window - max_inactive_days
    eligible = history.notna().sum(axis=0) >= min_observations
    eligible = eligible & history.loc[rebalance_date].notna()
    eligible_ids = eligible[eligible].index.tolist()
    if len(eligible_ids) < group_size:
        return {}
    corr_matrix = history[eligible_ids].corr(min_periods=min_observations)
    boards = {}
    for stock in candidate_ids:
        if stock not in corr_matrix.columns:
            continue
        correlations = corr_matrix[stock].dropna()
        correlations = correlations[correlations.index != stock]
        peers = correlations.nlargest(group_size - 1).index.tolist()
        if len(peers) == group_size - 1:
            boards[stock] = [stock] + peers
    return boards

def _csad_ratio_for_one_board(board_returns, center_stock, short_window=20, long_window=120, min_group_active=10):
    """
    计算固定小型板块的：
    ratio_short / ratio_long
    """
    center_return = board_returns[center_stock]
    active_count = board_returns.notna().sum(axis=1)
    csad = board_returns.sub(center_return, axis=0).abs().mean(axis=1, skipna=True).where(active_count >= min_group_active)
    short_ratio = csad.rolling(short_window, min_periods=short_window).std(ddof=1) / center_return.rolling(short_window, min_periods=short_window).std(ddof=1)
    long_ratio = csad.rolling(long_window, min_periods=long_window).std(ddof=1) / center_return.rolling(long_window, min_periods=long_window).std(ddof=1)
    factor = short_ratio / long_ratio
    return factor.replace([np.inf, -np.inf], np.nan)

def calculate_center_board_csad_ratio_daily(start_date, end_date, candidate_ids=None, peer_universe_ids=None, corr_window=120, max_inactive_days=20, group_size=10, short_window=20, long_window=120, min_group_active=10, batch_size=500):
    """
    CSAD 模型（基于时序回归变形）

    每月末：
    1. 为每只中心股票选取 9 只最高相关股票；
    2. 形成中心股票 + 9 个同伴的小型板块；
    3. 计算 CSAD 相对中心股收益率的短期、长期波动率比值；
    4. 输出 short_ratio / long_ratio。

    输出：
        index  : [date, order_book_id]
        column : CSAD_TS_RATIO
    """
    if group_size < 2:
        raise ValueError('group_size 至少应为 2。')
    if min_group_active > group_size:
        raise ValueError('min_group_active 不能大于 group_size。')
    if short_window > long_window:
        raise ValueError('short_window 不应大于 long_window。')
    candidate_ids = _csad_ratio_ids(candidate_ids)
    peer_universe_ids = _csad_ratio_ids(peer_universe_ids)
    all_ids = list(dict.fromkeys(candidate_ids + peer_universe_ids))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    required_history = max(corr_window, long_window) + 25
    calendar = pd.DatetimeIndex(get_trading_dates(start_date=start_date - pd.Timedelta(days=max(required_history * 3, 365)), end_date=end_date))
    target_dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
    if len(target_dates) == 0:
        raise ValueError('所选区间内没有交易日。')
    first_loc = calendar.get_loc(target_dates[0])
    last_loc = calendar.get_loc(target_dates[-1])
    if first_loc < required_history:
        raise ValueError('历史交易日不足，无法构建小型板块。')
    price_dates = calendar[first_loc - required_history:last_loc + 1]
    returns = _csad_ratio_get_returns_wide(order_book_ids=all_ids, price_dates=price_dates, batch_size=batch_size)
    all_month_ends = _csad_ratio_month_end_dates(returns.index)
    valid_month_ends = []
    for date in all_month_ends:
        if returns.index.get_loc(date) >= corr_window - 1:
            valid_month_ends.append(date)
    valid_month_ends = pd.DatetimeIndex(valid_month_ends)
    valid_month_ends = valid_month_ends[valid_month_ends <= target_dates[-1]]
    prior_rebalance_dates = valid_month_ends[valid_month_ends <= target_dates[0]]
    if len(prior_rebalance_dates) == 0:
        raise ValueError('开始日前缺少可用于建组的月末交易日。')
    first_rebalance_date = prior_rebalance_dates[-1]
    rebalance_dates = valid_month_ends[valid_month_ends >= first_rebalance_date]
    factor_wide = pd.DataFrame(index=target_dates, columns=candidate_ids, dtype=float)
    for pos, rebalance_date in enumerate(rebalance_dates):
        next_rebalance_date = rebalance_dates[pos + 1] if pos + 1 < len(rebalance_dates) else None
        if next_rebalance_date is None:
            segment_dates = target_dates[target_dates >= rebalance_date]
        else:
            segment_dates = target_dates[(target_dates >= rebalance_date) & (target_dates < next_rebalance_date)]
        if len(segment_dates) == 0:
            continue
        boards = _csad_ratio_select_boards(returns=returns, rebalance_date=rebalance_date, candidate_ids=candidate_ids, corr_window=corr_window, max_inactive_days=max_inactive_days, group_size=group_size)
        segment_start_loc = returns.index.get_loc(segment_dates[0])
        segment_end_loc = returns.index.get_loc(segment_dates[-1])
        history_start_loc = max(0, segment_start_loc - long_window + 1)
        calculation_dates = returns.index[history_start_loc:segment_end_loc + 1]
        for stock, board in boards.items():
            board_returns = returns.loc[calculation_dates, board]
            factor = _csad_ratio_for_one_board(board_returns=board_returns, center_stock=stock, short_window=short_window, long_window=long_window, min_group_active=min_group_active)
            factor_wide.loc[segment_dates, stock] = factor.reindex(segment_dates)
    result = factor_wide.stack(dropna=True).rename('CSAD_TS_RATIO').to_frame()
    result.index = result.index.set_names(['date', 'order_book_id'])
    return result
from _factor_io import save_from_cli

def build_factor(start_date, end_date, data_dir):
    return calculate_center_board_csad_ratio_daily(start_date=start_date, end_date=end_date, candidate_ids=None, peer_universe_ids=None, corr_window=120, max_inactive_days=20, group_size=10, short_window=20, long_window=120, min_group_active=10, batch_size=500)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='CSAD 模型（基于时序回归变形）', factor_field='CSAD_TS_RATIO')
