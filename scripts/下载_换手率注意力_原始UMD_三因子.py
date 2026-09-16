"""米筐云端下载脚本：换手率注意力因子的 UMD 衍生因子。

输出：
1. INDUS_TURN_BETA_MONTHLY_EQ_UMD    = |beta_{i,t}|，换手率注意力暴露；
2. INDUS_TURN_ALPHA_MONTHLY_EQ_UMD   = alpha_{i,t}，20 日滚动回归截距；
3. INDUS_TURN_EPSILON_MONTHLY_EQ_UMD = epsilon_{i,d}，滚动窗口末日的回归残差。

UMD_STYLE = "monthly" 时：每月按 t-12 至 t-2 的 11 个完整自然月累计收益
分出赢家前 30% 与输家后 30%，月内持有固定名单；每日 UMD 为两腿等权日收益之差。

UMD_STYLE = "rolling_22d" 时：每个交易日 T 均重新形成股票池，使用
P[T-22] / P[T-22*12] - 1 排序（即约第 12 至第 2 个滞后月的动量，跳过最近
22 个交易日），每日 UMD 为当日赢家组等权收益减输家组等权收益。

UMD_STYLE = "monthly_value_weighted" 时：股票池仍为月度固定的原始定义，
但赢家/输家组合日收益按滞后市值加权。此模式输出 abs(beta)、alpha 与 epsilon，
用于以统一股票池重建原始的换手率注意力暴露。

UMD_STYLE = "rolling_22d_value_weighted" 时：每个交易日 T 按
P[T-22] / P[T-264] - 1 重新形成赢家与输家池；两腿 T 日收益均按
T-1 市值加权。此模式输出 |beta|、alpha 与 epsilon。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac
from rqdatac import (
    all_instruments,
    get_factor,
    get_instrument_industry,
    get_price,
    get_trading_dates,
    get_turnover_rate,
    get_yield_curve,
)


# ============================ 可修改配置 ============================
START_DATE = "2019-01-02"
END_DATE = "2026-08-31"
OUTPUT_DIR = Path("indus_turn_monthly_value_weighted_umd_factors")

REGRESSION_WINDOW = 20
INDUSTRY_SOURCE = "citics_2019"
INDUSTRY_LEVEL = 1
INDUSTRY_REFRESH = "M"       # M：月初更新行业标签；D：每日更新
MIN_INDUSTRY_STOCKS = 5
MIN_MARKET_STOCKS = 100
MIN_PORTFOLIO_STOCKS = 10
BREAKPOINT_UNIVERSE = "main_board"  # SMB/HML 断点来源
BATCH_SIZE = 500
INDUSTRY_BATCH_SIZE = 2000
LOOKBACK_CALENDAR_DAYS = 1100

# "monthly"：自然月固定股票池，等权两腿；
# "rolling_22d"：每天重新形成股票池，等权两腿；
# "monthly_value_weighted"：自然月固定股票池，市值加权两腿（本次使用）。
# "rolling_22d_value_weighted"：每天重新形成股票池，市值加权两腿（本次使用）。
# 本脚本默认重建“原始”月度固定、市值加权 UMD 的全部三项回归产物。
# 其中 BETA_MONTHLY_VW_UMD 可替换旧股票池版本
# INDUS_TURN_BETA_MONTHLY_VW_UMD_LEGACY_UNIVERSE.parquet。
UMD_STYLE = "monthly_value_weighted"
UMD_FORMATION_LOOKBACK = 22 * 12
UMD_SKIP_RECENT_DAYS = 22


def chunks(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def first_dates_by_month(dates):
    months = dates.to_period("M")
    return dates[~months.duplicated()]


def first_dates_by_july(dates):
    dates = first_dates_by_month(dates)
    return dates[dates.month == 7]


def a_share_ids(instruments):
    """沪深 A 股；排除沪市 900 与深市 200 开头的 B 股。"""
    ids = instruments["order_book_id"].astype(str)
    codes = ids.str.split(".").str[0]
    exchange = (
        instruments["exchange"].astype(str)
        if "exchange" in instruments.columns
        else ids.str.split(".").str[-1]
    )
    is_sh_sz = exchange.isin(["XSHG", "XSHE"])
    is_b_share = ((exchange == "XSHG") & codes.str.startswith("900")) | (
        (exchange == "XSHE") & codes.str.startswith("200")
    )
    return instruments.loc[
        is_sh_sz & ~is_b_share, "order_book_id"
    ].dropna().drop_duplicates().tolist()


def as_wide(raw, ids, dates, field):
    if raw is None or raw.empty:
        return pd.DataFrame(index=dates, columns=ids, dtype=float)
    if isinstance(raw.index, pd.MultiIndex):
        wide = raw[field].unstack("order_book_id")
    else:
        wide = raw[[field]].rename(columns={field: ids[0]})
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(index=dates, columns=ids)


def download_price_wide(ids, dates, fields):
    output = {field: [] for field in fields}
    for number, batch in enumerate(chunks(ids, BATCH_SIZE), start=1):
        print("下载行情：第 {}/{} 批".format(number, int(np.ceil(len(ids) / BATCH_SIZE))))
        raw = get_price(
            order_book_ids=batch,
            start_date=dates[0],
            end_date=dates[-1],
            frequency="1d",
            fields=fields,
            adjust_type="pre",
            skip_suspended=False,
            expect_df=True,
        )
        for field in fields:
            output[field].append(as_wide(raw, batch, dates, field))
    return {
        field: pd.concat(parts, axis=1)
        .loc[:, lambda x: ~x.columns.duplicated()]
        .reindex(index=dates, columns=ids)
        .astype(float)
        for field, parts in output.items()
    }


def download_turnover_wide(ids, dates):
    blocks = []
    for number, batch in enumerate(chunks(ids, BATCH_SIZE), start=1):
        print("下载换手率：第 {}/{} 批".format(number, int(np.ceil(len(ids) / BATCH_SIZE))))
        raw = get_turnover_rate(
            order_book_ids=batch,
            start_date=dates[0],
            end_date=dates[-1],
            fields="today",
            expect_df=True,
        )
        blocks.append(as_wide(raw, batch, dates, "today"))
    return pd.concat(blocks, axis=1).loc[:, lambda x: ~x.columns.duplicated()].reindex(index=dates, columns=ids).astype(float)


def download_factor_wide(ids, dates, factor_name):
    blocks = []
    for number, batch in enumerate(chunks(ids, BATCH_SIZE), start=1):
        print("下载 {}：第 {}/{} 批".format(factor_name, number, int(np.ceil(len(ids) / BATCH_SIZE))))
        raw = get_factor(
            order_book_ids=batch,
            factor=factor_name,
            start_date=dates[0],
            end_date=dates[-1],
            expect_df=True,
        )
        blocks.append(as_wide(raw, batch, dates, factor_name))
    return pd.concat(blocks, axis=1).loc[:, lambda x: ~x.columns.duplicated()].reindex(index=dates, columns=ids).astype(float)


def get_daily_rf(dates, tenor="1M", day_count=365.0):
    curve = get_yield_curve(
        start_date=dates[0] - pd.Timedelta(days=10),
        end_date=dates[-1],
        tenor=tenor,
    )
    if curve is None or len(curve) == 0:
        raise RuntimeError("未取得中债收益率曲线。")
    annual = curve if isinstance(curve, pd.Series) else curve[tenor]
    annual.index = pd.to_datetime(annual.index)
    annual = pd.to_numeric(annual, errors="coerce").groupby(level=0).last()
    annual = annual.reindex(dates, method="ffill")
    if annual.isna().any():
        raise RuntimeError("无风险利率无法覆盖全部交易日。")
    return pd.Series(
        np.expm1(np.log1p(annual.to_numpy(float)) / day_count),
        index=dates,
        name="RF",
    )


def main_board_mask(ids, instruments):
    if BREAKPOINT_UNIVERSE == "all_a":
        return np.ones(len(ids), dtype=bool)
    board = instruments.drop_duplicates("order_book_id").set_index("order_book_id")["board_type"]
    return board.reindex(ids).eq("MainBoard").fillna(False).to_numpy(bool)


def value_weighted_return(returns, weights, mask, min_stocks):
    valid = mask & np.isfinite(returns) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < min_stocks:
        return np.nan
    return np.dot(returns[valid], weights[valid]) / weights[valid].sum()


def build_equal_weighted_umd(raw_returns):
    """原始 UMD：月度形成、等权收益、无市值筛选或市值加权。"""
    dates = raw_returns.index
    values = raw_returns.to_numpy(float)
    n_dates, n_stocks = values.shape
    labels = np.full((n_dates, n_stocks), -1, dtype=np.int8)
    months = dates.to_period("M")
    rebalance_positions = [dates.get_loc(x) for x in first_dates_by_month(dates)]

    for sequence, position in enumerate(rebalance_positions):
        next_position = rebalance_positions[sequence + 1] if sequence + 1 < len(rebalance_positions) else n_dates
        current_month = dates[position].to_period("M")
        # t-12 至 t-2：11 个完整自然月，跳过最近的 t-1 月。
        history_mask = (months >= current_month - 12) & (months <= current_month - 2)
        history = values[history_mask]
        if history.shape[0] < 150:
            continue
        valid_history = np.isfinite(history).all(axis=0)
        cumulative = np.full(n_stocks, np.nan)
        cumulative[valid_history] = np.prod(1.0 + history[:, valid_history], axis=0) - 1.0
        eligible = np.flatnonzero(np.isfinite(cumulative))
        if len(eligible) < MIN_MARKET_STOCKS:
            continue
        momentum = cumulative[eligible]
        loser_cut, winner_cut = np.quantile(momentum, [0.30, 0.70])
        labels[position:next_position, eligible[momentum <= loser_cut]] = 0
        labels[position:next_position, eligible[momentum >= winner_cut]] = 1

    umd = np.full(n_dates, np.nan)
    for date_position in range(n_dates):
        daily = values[date_position]
        winners = daily[(labels[date_position] == 1) & np.isfinite(daily)]
        losers = daily[(labels[date_position] == 0) & np.isfinite(daily)]
        if len(winners) >= MIN_PORTFOLIO_STOCKS and len(losers) >= MIN_PORTFOLIO_STOCKS:
            umd[date_position] = winners.mean() - losers.mean()
    return pd.Series(umd, index=dates, name="UMD")


def build_monthly_value_weighted_umd(raw_returns, lag_cap):
    """原始月度形成、市值加权两腿的 UMD。

    月初以 t-12 至 t-2 的完整自然月累计收益排序，赢家前 30%、
    输家后 30%，名单保持至下月再平衡。组合日收益以当日可获得的
    ``lag_cap``（上一交易日市值）加权，避免使用当日收盘后才确定的市值。
    """
    dates = raw_returns.index
    values = raw_returns.to_numpy(float)
    caps = lag_cap.reindex(index=dates, columns=raw_returns.columns).to_numpy(float)
    n_dates, n_stocks = values.shape
    labels = np.full((n_dates, n_stocks), -1, dtype=np.int8)
    months = dates.to_period("M")
    rebalance_positions = [dates.get_loc(x) for x in first_dates_by_month(dates)]

    for sequence, position in enumerate(rebalance_positions):
        next_position = (
            rebalance_positions[sequence + 1]
            if sequence + 1 < len(rebalance_positions) else n_dates
        )
        current_month = dates[position].to_period("M")
        history_mask = (
            (months >= current_month - 12)
            & (months <= current_month - 2)
        )
        history = values[history_mask]
        if history.shape[0] < 150:
            continue

        formation_cap = caps[position]
        valid_history = np.isfinite(history).all(axis=0)
        cumulative = np.full(n_stocks, np.nan)
        cumulative[valid_history] = (
            np.prod(1.0 + history[:, valid_history], axis=0) - 1.0
        )
        eligible = (
            np.isfinite(cumulative)
            & np.isfinite(formation_cap)
            & (formation_cap > 0)
        )
        if eligible.sum() < MIN_MARKET_STOCKS:
            continue

        eligible_idx = np.flatnonzero(eligible)
        momentum = cumulative[eligible_idx]
        loser_cut, winner_cut = np.quantile(momentum, [0.30, 0.70])
        labels[position:next_position, eligible_idx[momentum <= loser_cut]] = 0
        labels[position:next_position, eligible_idx[momentum >= winner_cut]] = 1

    umd = np.full(n_dates, np.nan)
    for position in range(n_dates):
        winner_return = value_weighted_return(
            values[position], caps[position], labels[position] == 1,
            MIN_PORTFOLIO_STOCKS,
        )
        loser_return = value_weighted_return(
            values[position], caps[position], labels[position] == 0,
            MIN_PORTFOLIO_STOCKS,
        )
        if np.isfinite(winner_return) and np.isfinite(loser_return):
            umd[position] = winner_return - loser_return
    return pd.Series(umd, index=dates, name="UMD")


def build_rolling_equal_weighted_umd(raw_returns):
    """每日滚动形成、等权持有的 UMD。

    对任一交易日 T，形成期动量为::

        P[T - 22] / P[T - 22 * 12] - 1

    即使用约第 12 至第 2 个滞后月的收益排序，并跳过最近 22 个交易日。
    股票池仅依赖 T-22 及更早的信息；形成后立刻以 T 日收益计算 UMD[T]，
    因而在 T 日收盘后可得，不含未来数据。
    """
    dates = raw_returns.index
    values = raw_returns.to_numpy(float)
    n_dates, n_stocks = values.shape
    lookback = UMD_FORMATION_LOOKBACK
    skip = UMD_SKIP_RECENT_DAYS

    if lookback <= skip:
        raise ValueError("UMD_FORMATION_LOOKBACK 必须大于 UMD_SKIP_RECENT_DAYS。")

    umd = np.full(n_dates, np.nan)
    for t in range(lookback, n_dates):
        # P[T-skip] / P[T-lookback] 对应的逐日收益为
        # (T-lookback+1) 至 (T-skip)，端点均包含。
        history = values[t - lookback + 1:t - skip + 1]
        valid_history = np.isfinite(history).all(axis=0)
        if valid_history.sum() < MIN_MARKET_STOCKS:
            continue

        cumulative = np.full(n_stocks, np.nan)
        cumulative[valid_history] = (
            np.prod(1.0 + history[:, valid_history], axis=0) - 1.0
        )
        eligible = np.flatnonzero(np.isfinite(cumulative))
        momentum = cumulative[eligible]
        loser_cut, winner_cut = np.quantile(momentum, [0.30, 0.70])

        winners = values[t, eligible[momentum >= winner_cut]]
        losers = values[t, eligible[momentum <= loser_cut]]
        winners = winners[np.isfinite(winners)]
        losers = losers[np.isfinite(losers)]
        if len(winners) >= MIN_PORTFOLIO_STOCKS and len(losers) >= MIN_PORTFOLIO_STOCKS:
            umd[t] = winners.mean() - losers.mean()

    return pd.Series(umd, index=dates, name="UMD")


def build_rolling_value_weighted_umd(raw_returns, lag_cap):
    """每日滚动形成、滞后市值加权两腿的 UMD。

    对交易日 T，先以 ``P[T-22] / P[T-264] - 1`` 形成前/后 30% 股票池。
    股票是否具备市值数据以 T-22 的滞后市值判断；实际 UMD[T] 的赢家与
    输家日收益均以 T-1 市值加权。因此股票池形成与组合权重均没有使用
    T 收盘后才知道的信息。
    """
    dates = raw_returns.index
    values = raw_returns.to_numpy(float)
    caps = lag_cap.reindex(index=dates, columns=raw_returns.columns).to_numpy(float)
    n_dates, n_stocks = values.shape
    lookback = UMD_FORMATION_LOOKBACK
    skip = UMD_SKIP_RECENT_DAYS
    if lookback <= skip:
        raise ValueError("UMD_FORMATION_LOOKBACK 必须大于 UMD_SKIP_RECENT_DAYS。")

    umd = np.full(n_dates, np.nan)
    for t in range(lookback, n_dates):
        # P[T-skip] / P[T-lookback] 对应的逐日收益：
        # (T-lookback+1) 至 (T-skip)，两端均包含。
        history = values[t - lookback + 1:t - skip + 1]
        valid_history = np.isfinite(history).all(axis=0)
        formation_cap = caps[t - skip]
        eligible = (
            valid_history
            & np.isfinite(formation_cap)
            & (formation_cap > 0)
        )
        if eligible.sum() < MIN_MARKET_STOCKS:
            continue

        cumulative = np.full(n_stocks, np.nan)
        cumulative[eligible] = (
            np.prod(1.0 + history[:, eligible], axis=0) - 1.0
        )
        eligible_idx = np.flatnonzero(np.isfinite(cumulative))
        momentum = cumulative[eligible_idx]
        loser_cut, winner_cut = np.quantile(momentum, [0.30, 0.70])
        winner_mask = np.zeros(n_stocks, dtype=bool)
        loser_mask = np.zeros(n_stocks, dtype=bool)
        winner_mask[eligible_idx[momentum >= winner_cut]] = True
        loser_mask[eligible_idx[momentum <= loser_cut]] = True

        winner_return = value_weighted_return(
            values[t], caps[t], winner_mask, MIN_PORTFOLIO_STOCKS
        )
        loser_return = value_weighted_return(
            values[t], caps[t], loser_mask, MIN_PORTFOLIO_STOCKS
        )
        if np.isfinite(winner_return) and np.isfinite(loser_return):
            umd[t] = winner_return - loser_return

    return pd.Series(umd, index=dates, name="UMD")


def build_controls(raw_returns, lag_cap, lag_btm, rf, breakpoint_mask):
    """MKT_RF、SMB、HML 保持原定义；UMD 由 UMD_STYLE 控制。"""
    dates = raw_returns.index
    returns = raw_returns.to_numpy(float)
    caps = lag_cap.to_numpy(float)
    btms = lag_btm.to_numpy(float)
    n_dates, n_stocks = returns.shape
    size_labels = np.full((n_dates, n_stocks), -1, dtype=np.int8)
    value_labels = np.full((n_dates, n_stocks), -1, dtype=np.int8)

    july_positions = [dates.get_loc(x) for x in first_dates_by_july(dates)]
    for sequence, position in enumerate(july_positions):
        next_position = july_positions[sequence + 1] if sequence + 1 < len(july_positions) else n_dates
        cap, btm = caps[position], btms[position]
        eligible_all = np.isfinite(cap) & np.isfinite(btm) & (cap > 0) & (btm > 0)
        eligible_breakpoint = eligible_all & breakpoint_mask
        if eligible_breakpoint.sum() < max(MIN_MARKET_STOCKS, 30):
            continue
        size_cut = np.median(cap[eligible_breakpoint])
        btm_low, btm_high = np.quantile(btm[eligible_breakpoint], [0.30, 0.70])
        valid = np.flatnonzero(eligible_all)
        size_labels[position:next_position, valid[cap[valid] <= size_cut]] = 0
        size_labels[position:next_position, valid[cap[valid] > size_cut]] = 1
        value_labels[position:next_position, valid[btm[valid] <= btm_low]] = 0
        value_labels[position:next_position, valid[(btm[valid] > btm_low) & (btm[valid] <= btm_high)]] = 1
        value_labels[position:next_position, valid[btm[valid] > btm_high]] = 2

    controls = np.full((n_dates, 4), np.nan)
    for position in range(n_dates):
        daily, daily_cap = returns[position], caps[position]
        market = value_weighted_return(daily, daily_cap, np.ones(n_stocks, dtype=bool), MIN_MARKET_STOCKS)
        if np.isfinite(market) and np.isfinite(rf.iloc[position]):
            controls[position, 0] = market - rf.iloc[position]
        six = {}
        for size in (0, 1):
            for value in (0, 1, 2):
                six[size, value] = value_weighted_return(
                    daily, daily_cap,
                    (size_labels[position] == size) & (value_labels[position] == value),
                    MIN_PORTFOLIO_STOCKS,
                )
        sl, sn, sh = six[0, 0], six[0, 1], six[0, 2]
        bl, bn, bh = six[1, 0], six[1, 1], six[1, 2]
        if np.isfinite([sl, sn, sh, bl, bn, bh]).all():
            controls[position, 1] = (sl + sn + sh) / 3.0 - (bl + bn + bh) / 3.0
            controls[position, 2] = (sh + bh) / 2.0 - (sl + bl) / 2.0
    if UMD_STYLE == "monthly":
        umd = build_equal_weighted_umd(raw_returns)
    elif UMD_STYLE == "rolling_22d":
        umd = build_rolling_equal_weighted_umd(raw_returns)
    elif UMD_STYLE == "monthly_value_weighted":
        umd = build_monthly_value_weighted_umd(raw_returns, lag_cap)
    elif UMD_STYLE == "rolling_22d_value_weighted":
        umd = build_rolling_value_weighted_umd(raw_returns, lag_cap)
    else:
        raise ValueError(
            "UMD_STYLE 仅支持 'monthly'、'rolling_22d' 或 "
            "'monthly_value_weighted'、'rolling_22d_value_weighted'。"
        )
    controls[:, 3] = umd.to_numpy(float)
    return pd.DataFrame(controls, index=dates, columns=["MKT_RF", "SMB", "HML", "UMD"])


def build_industry_labels(ids, dates):
    if INDUSTRY_REFRESH == "M":
        refresh_dates = first_dates_by_month(dates)
    elif INDUSTRY_REFRESH == "D":
        refresh_dates = dates
    else:
        raise ValueError("INDUSTRY_REFRESH 仅支持 M 或 D。")
    labels = np.full((len(dates), len(ids)), -1, dtype=np.int16)
    id_positions = {stock_id: pos for pos, stock_id in enumerate(ids)}
    code_labels = {}
    refresh_positions = [dates.get_loc(x) for x in refresh_dates]
    for sequence, date in enumerate(refresh_dates):
        print("下载行业：{}/{}".format(sequence + 1, len(refresh_dates)))
        next_position = refresh_positions[sequence + 1] if sequence + 1 < len(refresh_positions) else len(dates)
        one_day = np.full(len(ids), -1, dtype=np.int16)
        for batch in chunks(ids, INDUSTRY_BATCH_SIZE):
            raw = get_instrument_industry(batch, source=INDUSTRY_SOURCE, level=INDUSTRY_LEVEL, date=date)
            if raw is None or len(raw) == 0:
                continue
            data = raw.reset_index() if isinstance(raw.index, pd.MultiIndex) or raw.index.name is not None else raw.copy()
            field = "first_industry_code"
            for stock_id, code in data.set_index("order_book_id")[field].items():
                if stock_id not in id_positions or pd.isna(code):
                    continue
                code = str(code)
                code_labels.setdefault(code, len(code_labels))
                one_day[id_positions[stock_id]] = code_labels[code]
        labels[refresh_positions[sequence]:next_position] = one_day
    return labels, code_labels


def calculate_three_factors():
    instruments = all_instruments("CS")
    ids = a_share_ids(instruments)
    target_dates = pd.DatetimeIndex(pd.to_datetime(get_trading_dates(START_DATE, END_DATE)))
    history_start = target_dates[0] - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)
    dates = pd.DatetimeIndex(pd.to_datetime(get_trading_dates(history_start, END_DATE)))

    price = download_price_wide(ids, dates, ["close", "volume"])
    raw_returns = price["close"].pct_change(fill_method=None).iloc[1:]
    return_dates = raw_returns.index
    turnover = download_turnover_wide(ids, dates).reindex(return_dates)
    active = price["volume"].iloc[1:].reindex(return_dates).gt(0) & turnover.notna()
    raw_returns = raw_returns.where(active)

    market_cap = download_factor_wide(ids, dates, "market_cap_3")
    book_to_market = download_factor_wide(ids, dates, "book_to_market_ratio_lyr")
    lag_cap = market_cap.shift(1).reindex(return_dates)
    lag_btm = book_to_market.shift(1).reindex(return_dates)
    rf = get_daily_rf(return_dates)
    controls = build_controls(raw_returns, lag_cap, lag_btm, rf, main_board_mask(ids, instruments))
    industry_labels, _ = build_industry_labels(ids, return_dates)

    turnover_values, active_values = turnover.to_numpy(float), active.to_numpy(bool)
    industry_turn = np.full((len(return_dates), max(industry_labels.max() + 1, 0)), np.nan)
    for t in range(len(return_dates)):
        for industry_id in np.unique(industry_labels[t][industry_labels[t] >= 0]):
            values = turnover_values[t][active_values[t] & (industry_labels[t] == industry_id)]
            if len(values) >= max(MIN_INDUSTRY_STOCKS, 2):
                industry_turn[t, industry_id] = np.std(values, ddof=1)

    beta = np.full_like(raw_returns.to_numpy(float), np.nan)
    alpha = np.full_like(beta, np.nan)
    epsilon = np.full_like(beta, np.nan)
    y_all = raw_returns.sub(rf, axis=0).to_numpy(float)
    x_controls = controls[["MKT_RF", "SMB", "HML", "UMD"]].to_numpy(float)
    target_positions = [return_dates.get_loc(d) for d in target_dates if d in return_dates]

    for t in target_positions:
        left = t - REGRESSION_WINDOW + 1
        if left < 0 or not np.isfinite(x_controls[left:t + 1]).all():
            continue
        for industry_id in np.unique(industry_labels[t][industry_labels[t] >= 0]):
            stocks = np.flatnonzero(industry_labels[t] == industry_id)
            attention = industry_turn[left:t + 1, industry_id]
            if not np.isfinite(attention).all():
                continue
            x = np.column_stack([np.ones(REGRESSION_WINDOW), attention, x_controls[left:t + 1]])
            if np.linalg.matrix_rank(x) < x.shape[1]:
                continue
            y = y_all[left:t + 1, :][:, stocks]
            valid = np.isfinite(y).all(axis=0)
            if not valid.any():
                continue
            coefficients = np.linalg.lstsq(x, y[:, valid], rcond=None)[0]
            fitted_today = x[-1] @ coefficients
            valid_stocks = stocks[valid]
            beta[t, valid_stocks] = np.abs(coefficients[1])
            alpha[t, valid_stocks] = coefficients[0]
            epsilon[t, valid_stocks] = y[-1, valid] - fitted_today

    index = return_dates[target_positions]
    columns = ids
    if UMD_STYLE == "monthly_value_weighted":
        # 与新增衍生因子完全使用同一股票池和计算口径；完整输出三项，
        # 使 |beta| 不再依赖历史较窄股票池的旧文件。
        return {
            "INDUS_TURN_BETA_MONTHLY_VW_UMD": pd.DataFrame(
                beta[target_positions], index=index, columns=columns
            ),
            "INDUS_TURN_ALPHA_MONTHLY_VW_UMD": pd.DataFrame(
                alpha[target_positions], index=index, columns=columns
            ),
            "INDUS_TURN_EPSILON_MONTHLY_VW_UMD": pd.DataFrame(
                epsilon[target_positions], index=index, columns=columns
            ),
        }

    if UMD_STYLE == "rolling_22d_value_weighted":
        return {
            "INDUS_TURN_BETA_ROLLING_VW_UMD": pd.DataFrame(
                beta[target_positions], index=index, columns=columns
            ),
            "INDUS_TURN_ALPHA_ROLLING_VW_UMD": pd.DataFrame(
                alpha[target_positions], index=index, columns=columns
            ),
            "INDUS_TURN_EPSILON_ROLLING_VW_UMD": pd.DataFrame(
                epsilon[target_positions], index=index, columns=columns
            ),
        }

    suffix = "MONTHLY_EQ_UMD" if UMD_STYLE == "monthly" else "ROLLING_EQ_UMD"
    return {
        "INDUS_TURN_BETA_{}".format(suffix): pd.DataFrame(beta[target_positions], index=index, columns=columns),
        "INDUS_TURN_ALPHA_{}".format(suffix): pd.DataFrame(alpha[target_positions], index=index, columns=columns),
        "INDUS_TURN_EPSILON_{}".format(suffix): pd.DataFrame(epsilon[target_positions], index=index, columns=columns),
    }


if __name__ == "__main__":
    rqdatac.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("UMD 模式：{}".format(UMD_STYLE))
    if UMD_STYLE == "rolling_22d":
        print(
            "每日股票池按 P[T-{}] / P[T-{}] - 1 形成；"
            "跳过最近 {} 个交易日。".format(
                UMD_SKIP_RECENT_DAYS,
                UMD_FORMATION_LOOKBACK,
                UMD_SKIP_RECENT_DAYS,
            )
        )
    elif UMD_STYLE == "monthly_value_weighted":
        print(
            "月度固定赢家/输家池；赢家与输家每日收益均按上一交易日市值加权；"
            "输出 abs(beta)、alpha 与 epsilon。"
        )
    elif UMD_STYLE == "rolling_22d_value_weighted":
        print(
            "每日按 P[T-{}] / P[T-{}] - 1 重新形成赢家/输家池；"
            "两腿每日收益按 T-1 市值加权。".format(
                UMD_SKIP_RECENT_DAYS,
                UMD_FORMATION_LOOKBACK,
            )
        )
    factors = calculate_three_factors()
    for name, factor in factors.items():
        factor.index.name = "date"
        factor.columns.name = "order_book_id"
        path = OUTPUT_DIR / (name + ".parquet")
        factor.to_parquet(path)
        print("已保存：{}；形状：{}".format(path, factor.shape))
