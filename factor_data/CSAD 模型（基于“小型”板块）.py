import rqdatac
rqdatac.init()
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates
import numpy as np
import pandas as pd
from rqdatac import get_trading_dates
import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats
from rqdatac import get_price
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
BENCHMARK = '000985.XSHG'
from rqdatac import get_price
BENCHMARK = '000985.XSHG'
import numpy as np
import pandas as pd
from rqdatac import all_instruments, get_price, get_trading_dates

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

def _get_returns(close, volume):
    """只保留前后两个交易日均正常成交的日收益。"""
    returns = close.pct_change(fill_method=None).iloc[1:]
    traded_today = volume.iloc[1:].gt(0)
    traded_yesterday = volume.shift(1).iloc[1:].gt(0)
    valid_price = close.iloc[1:].gt(0) & close.shift(1).iloc[1:].gt(0)
    return returns.where(traded_today & traded_yesterday & valid_price)

def _stack_factor(factor_wide, factor_name):
    """日期 × 股票宽表转为 date、order_book_id 两层索引。"""
    panel = factor_wide.stack(dropna=True).rename(factor_name).to_frame()
    panel.index = panel.index.set_names(['date', 'order_book_id'])
    return panel

def _download_returns_wide(order_book_ids, price_dates, batch_size=500):
    """
    分批下载收盘价与成交量，返回“日期 × 股票”的日收益率宽表。

    停牌或无成交日的收益率会保留为 NaN。
    """
    return_parts = []
    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]
        close, volume = _get_batch_close_volume(batch=batch, price_dates=price_dates)
        if close is None:
            continue
        return_parts.append(_get_returns(close, volume))
    if not return_parts:
        raise RuntimeError('未取得有效价格数据。')
    returns = pd.concat(return_parts, axis=1)
    return returns.reindex(index=price_dates[1:], columns=order_book_ids).sort_index()

def calculate_small_board_csad(start_date, end_date, candidate_ids=None, peer_universe_ids=None, corr_window=120, max_inactive_days=20, group_size=10, zscore_window=120, smooth_window=20, min_group_active=10, batch_size=500):
    """
    基于“小型”板块的 CSAD 羊群效应因子。

    返回
    ----
    DataFrame
        MultiIndex：date、order_book_id
        Column：SMALL_BOARD_CSAD

    因子值越大：
        表示近期板块 CSAD 越低、股票走势越趋同、羊群效应越强。
    """
    if group_size < 2:
        raise ValueError('group_size 至少为 2。')
    if min_group_active > group_size:
        raise ValueError('min_group_active 不能大于 group_size。')
    if corr_window <= max_inactive_days:
        raise ValueError('corr_window 必须大于 max_inactive_days。')
    required_lookback = corr_window + zscore_window + smooth_window + 25
    target_dates, price_dates = _prepare_dates(start_date=start_date, end_date=end_date, lookback_days=required_lookback)
    candidate_ids = _get_ids(candidate_ids)
    peer_universe_ids = _get_ids(peer_universe_ids)
    peer_universe_ids = list(dict.fromkeys(peer_universe_ids + candidate_ids))
    returns = _download_returns_wide(order_book_ids=peer_universe_ids, price_dates=price_dates, batch_size=batch_size)
    work_dates = returns.index
    target_date_set = set(target_dates)
    factor_wide = pd.DataFrame(np.nan, index=target_dates, columns=candidate_ids)
    month_end_dates = pd.DatetimeIndex(pd.Series(work_dates, index=work_dates.to_period('M')).groupby(level=0).max().values)
    min_corr_observations = corr_window - max_inactive_days
    for rebalance_pos, rebalance_date in enumerate(month_end_dates):
        date_pos = work_dates.get_loc(rebalance_date)
        if date_pos + 1 < corr_window:
            continue
        correlation_returns = returns.iloc[date_pos - corr_window + 1:date_pos + 1]
        valid_count = correlation_returns.notna().sum(axis=0)
        eligible_ids = valid_count[valid_count >= min_corr_observations].index.tolist()
        if len(eligible_ids) < group_size:
            continue
        corr_matrix = correlation_returns[eligible_ids].corr(min_periods=min_corr_observations)
        current_candidates = [stock for stock in candidate_ids if stock in eligible_ids and pd.notna(returns.loc[rebalance_date, stock])]
        if rebalance_pos < len(month_end_dates) - 1:
            next_rebalance_date = month_end_dates[rebalance_pos + 1]
            effective_dates = work_dates[(work_dates >= rebalance_date) & (work_dates < next_rebalance_date)]
        else:
            effective_dates = work_dates[work_dates >= rebalance_date]
        output_dates = [date for date in effective_dates if date in target_date_set]
        if not output_dates:
            continue
        for stock in current_candidates:
            peer_corr = corr_matrix[stock].drop(index=stock, errors='ignore').dropna().sort_values(ascending=False)
            peers = peer_corr.head(group_size - 1).index.tolist()
            if len(peers) < group_size - 1:
                continue
            small_board = [stock] + peers
            board_returns = returns[small_board]
            board_mean_return = board_returns.mean(axis=1, skipna=True)
            active_count = board_returns.notna().sum(axis=1)
            csad = board_returns.sub(board_mean_return, axis=0).abs().mean(axis=1, skipna=True).where(active_count >= min_group_active)
            rolling_mean = csad.rolling(window=zscore_window, min_periods=zscore_window).mean()
            rolling_std = csad.rolling(window=zscore_window, min_periods=zscore_window).std(ddof=0)
            zscore = (csad - rolling_mean) / rolling_std.replace(0, np.nan)
            factor_series = -zscore.rolling(window=smooth_window, min_periods=smooth_window).mean()
            factor_wide.loc[output_dates, stock] = factor_series.reindex(output_dates).values
    return _stack_factor(factor_wide, 'SMALL_BOARD_CSAD')
from _factor_io import save_from_cli

def build_factor(start_date, end_date, data_dir):
    return calculate_small_board_csad(start_date=start_date, end_date=end_date, candidate_ids=None, peer_universe_ids=None, corr_window=120, max_inactive_days=20, group_size=10, zscore_window=120, smooth_window=20, min_group_active=10, batch_size=500)
if __name__ == '__main__':
    save_from_cli(build_factor=build_factor, output_stem='CSAD 模型（基于“小型”板块）', factor_field='SMALL_BOARD_CSAD')
