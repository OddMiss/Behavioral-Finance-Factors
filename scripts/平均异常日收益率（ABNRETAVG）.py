# 20日滑动窗口
print("脚本开始")

import numpy as np
import pandas as pd

from rqdatac import all_instruments, get_price, get_trading_dates
import rqdatac
rqdatac.init()

def _get_ids(stock_ids=None):
    """默认返回米筐 CS 股票全集。"""
    if stock_ids is not None:
        return list(dict.fromkeys(stock_ids))

    return (
        all_instruments("CS")
        .order_book_id
        .dropna()
        .drop_duplicates()
        .tolist()
    )

def _prepare_dates(start_date, end_date, lookback_days):
    """
    返回：
    target_dates：最终需要输出因子的交易日；
    price_dates：为计算滚动收益而额外向前展开后的价格日期。
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    calendar = pd.DatetimeIndex(
        get_trading_dates(
            start_date=start_date - pd.Timedelta(days=max(lookback_days * 3, 90)),
            end_date=end_date,
        )
    )

    target_dates = calendar[
        (calendar >= start_date) & (calendar <= end_date)
    ]

    if len(target_dates) == 0:
        raise ValueError("所选区间内没有交易日。")

    first_loc = calendar.get_loc(target_dates[0])
    last_loc = calendar.get_loc(target_dates[-1])

    if first_loc < lookback_days:
        raise ValueError("历史交易日不足，无法满足滚动窗口。")

    price_dates = calendar[first_loc - lookback_days:last_loc + 1]
    return target_dates, price_dates


def _get_batch_close_volume(batch, price_dates):
    """
    米筐多标的查询不能使用 skip_suspended=True；
    因此下载成交量，并用 volume > 0 排除当日停牌股票。
    """
    raw = get_price(
        order_book_ids=batch,
        start_date=price_dates[0],
        end_date=price_dates[-1],
        frequency="1d",
        fields=["close", "volume"],
        adjust_type="pre",
        skip_suspended=False,
        expect_df=True,
    )

    if raw is None or raw.empty:
        return None, None

    if isinstance(raw.index, pd.MultiIndex):
        close = raw["close"].unstack("order_book_id")
        volume = raw["volume"].unstack("order_book_id")
    else:
        close = raw[["close"]].rename(columns={"close": batch[0]})
        volume = raw[["volume"]].rename(columns={"volume": batch[0]})

    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)

    return (
        close.reindex(index=price_dates, columns=batch),
        volume.reindex(index=price_dates, columns=batch),
    )

# 旧函数
# def _get_returns(close, volume):
#     """仅保留当日实际有成交的股票收益率。"""
#     returns = close.pct_change(fill_method=None).iloc[1:]
#     traded_today = volume.iloc[1:].gt(0)
#     return returns.where(traded_today)

# 新函数，只保留前后两个交易日均正常成交的日收益，避免复牌日后的异常涨幅
def _get_returns(close, volume):
    """只保留前后两个交易日均正常成交的日收益。"""
    returns = close.pct_change(fill_method=None).iloc[1:]

    traded_today = volume.iloc[1:].gt(0)
    traded_yesterday = volume.shift(1).iloc[1:].gt(0)

    valid_price = (
        close.iloc[1:].gt(0)
        & close.shift(1).iloc[1:].gt(0)
    )

    return returns.where(traded_today & traded_yesterday & valid_price)


def _stack_factor(factor_wide, factor_name):
    """日期 × 股票宽表转为 date、order_book_id 两层索引。"""
    panel = factor_wide.stack(dropna=True).rename(factor_name).to_frame()
    panel.index = panel.index.set_names(["date", "order_book_id"])
    return panel


def calculate_abnretavg_daily(
    start_date,
    end_date,
    window=20,
    candidate_ids=None,
    market_ids=None,
    batch_size=500,
    min_market_stocks=500,
):
    """
    计算日频滚动平均异常日收益率（ABNRETAVG）。

    ABNRETAVG_i,T =
        mean_{t=T-window+1,...,T}[(r_i,t - r_bar,t)^2]

    返回
    ----
    DataFrame
        MultiIndex：date、order_book_id
        Column：ABNRETAVG
    """
    target_dates, price_dates = _prepare_dates(
        start_date=start_date,
        end_date=end_date,
        lookback_days=window,
    )

    market_ids = _get_ids(market_ids)
    candidate_ids = _get_ids(candidate_ids)

    return_dates = price_dates[1:]

    # 第一遍：计算全市场每日等权平均收益 r_bar,t。
    return_sum = pd.Series(0.0, index=return_dates)
    return_count = pd.Series(0, index=return_dates, dtype="int64")

    for start in range(0, len(market_ids), batch_size):
        batch = market_ids[start:start + batch_size]

        close, volume = _get_batch_close_volume(batch, price_dates)
        if close is None:
            continue

        returns = _get_returns(close, volume)

        return_sum += returns.sum(axis=1, skipna=True)
        return_count += returns.notna().sum(axis=1).astype("int64")

    market_returns = (
        return_sum / return_count.replace(0, np.nan)
    ).where(return_count >= min_market_stocks)

    # 第二遍：计算每只股票的滚动平均异常收益平方。
    factor_parts = []

    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]

        close, volume = _get_batch_close_volume(batch, price_dates)
        if close is None:
            continue

        returns = _get_returns(close, volume)

        abnormal_return_squared = (
            returns.sub(market_returns, axis=0)
            .pow(2)
        )

        # 每日以过去 window 个交易日为窗口。
        factor_wide = abnormal_return_squared.rolling(
            window=window,
            min_periods=window,
        ).mean()

        factor_wide = factor_wide.reindex(target_dates)
        factor_parts.append(
            _stack_factor(factor_wide, "ABNRETAVG")
        )

    if not factor_parts:
        return pd.DataFrame(columns=["ABNRETAVG"])

    return pd.concat(factor_parts).sort_index()

def exclude_recent_listings(abnretavg_panel, min_trading_days=60):
    """剔除上市未满指定交易日的股票因子值。

    以股票上市日为第 0 个交易日；当其后的 ``min_trading_days`` 个交易日均已
    过去，才保留因子。例如 ``min_trading_days=60`` 时，上市后的第 61 个交易日
    起才允许使用 ABNRETAVG。该规则可避免新股上市初期的无涨跌幅限制波动主导排序。

    本函数只过滤最终因子面板；因此可以直接接在
    ``calculate_abnretavg_daily(...)`` 后使用。
    """
    if min_trading_days < 0:
        raise ValueError("min_trading_days 不能为负数。")
    if not isinstance(abnretavg_panel.index, pd.MultiIndex) or abnretavg_panel.index.nlevels != 2:
        raise ValueError("索引必须为两层 MultiIndex：(date, order_book_id)。")

    # 延迟导入，使单纯的数据检查不依赖已初始化的 rqdatac。
    from rqdatac import all_instruments, get_trading_dates

    panel = abnretavg_panel.copy()
    panel.index = panel.index.set_names(["date", "order_book_id"])
    signal_dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    stock_ids = pd.Index(panel.index.get_level_values("order_book_id").unique())

    instruments_df = all_instruments("CS").set_index("order_book_id")
    listed_dates = pd.to_datetime(
        instruments_df["listed_date"].reindex(stock_ids), errors="coerce"
    )
    if listed_dates.isna().any():
        missing_ids = listed_dates[listed_dates.isna()].index.tolist()[:10]
        raise ValueError("以下股票缺少 listed_date：{}".format(missing_ids))

    calendar = pd.DatetimeIndex(
        get_trading_dates(
            start_date=listed_dates.min(),
            end_date=signal_dates.max(),
        )
    )
    listing_locs = calendar.searchsorted(listed_dates.to_numpy(), side="left")
    eligible_locs = listing_locs + min_trading_days

    eligible_from_values = np.full(
        len(stock_ids), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    within_calendar = eligible_locs < len(calendar)
    eligible_from_values[within_calendar] = calendar.to_numpy()[eligible_locs[within_calendar]]
    eligible_from = pd.Series(pd.to_datetime(eligible_from_values), index=stock_ids)

    row_ids = panel.index.get_level_values("order_book_id")
    row_eligible_from = eligible_from.reindex(row_ids).to_numpy()
    keep = signal_dates.to_numpy() >= row_eligible_from
    return panel.loc[keep].sort_index()

abnretavg_panel = calculate_abnretavg_daily(
    start_date="2020-01-01",
    end_date="2025-12-31",
    window=20,
)

# 上市后的第 61 个交易日起才保留因子
abnretavg_panel = exclude_recent_listings(
    abnretavg_panel,
    min_trading_days=60,
)

# 行：交易日；列：order_book_id；值：ABNRETD
abnretavg_panel_df = (
    abnretavg_panel["ABNRETAVG"]
    .unstack("order_book_id")
    .sort_index()
)

abnretavg_panel_df.index = pd.to_datetime(abnretavg_panel_df.index)
abnretavg_panel_df.columns.name = "order_book_id"

from rqfactor.notebook import factor_analysis

result_abnretavg = factor_analysis(
    f=abnretavg_panel_df,
    period=1,
    shift=1,
    rank_ic=True,
    quantile=10,
    ascending=True,
    winzorization="mad",
    normalization=False,
    neutralization="none",   # 先关闭：这是日频测试最常见的慢点
    include_st=False,
    include_new=False,
    benchmark="000985.XSHG",
)

from pathlib import Path
import json
import pandas as pd

result = result_abnretavg
folder_name = "平均异常日收益率（ABNRETAVG）"
name = "abnretavg_daily_none_q10"
save_dir = Path(f"factor_results/{folder_name}")
save_dir.mkdir(parents=True, exist_ok=True)

def save_result_item(name, value):
    """将米筐因子回测结果中的可用数据保存为 CSV。"""
    if value is None:
        print(f"跳过 {name}：结果为空")
        return

    if isinstance(value, pd.Series):
        value.rename(value.name or name).to_csv(
            save_dir / f"{name}.csv",
            index_label="date",
        )

    elif isinstance(value, pd.DataFrame):
        value.to_csv(save_dir / f"{name}.csv", index=True)

    elif isinstance(value, dict):
        pd.Series(value, name="value").to_csv(
            save_dir / f"{name}.csv",
            index_label="metric",
        )

    else:
        print(f"跳过 {name}：类型为 {type(value)}，不是表格数据")
        return

    print(f"已保存：{name}.csv")


# 米筐因子回测的核心原始结果
for attr_name in [
    "ic",
    "ic_summary",
    "ic_decay",
    "ic_industry_distribute",
    "quantile_factor_returns",
    "quantile_turnover",
]:
    save_result_item(attr_name, getattr(result, attr_name, None))

# 保存本次回测配置
params = {
    "period": 1,
    "shift": 1,
    "rank_ic": True,
    "quantile": 10,
    "ascending": True,
    "winzorization": "mad",
    "normalization": False,
    "neutralization": "none",
    "include_st": False,
    "include_new": False,
    "benchmark": "000985.XSHG",
}

with open(save_dir / "params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)

result.save_figs(mode='column', to=str(save_dir / f'{name}.html'), sizing_mode='stretch_both')
    
print("\n全部图标和数据已保存至：", save_dir.resolve())

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 替换为你的五年因子测试结果变量
ic = result.ic.dropna()

rolling_window = 10  # 可改为 12，与米筐默认图保持一致
rolling_ic = ic.rolling(rolling_window).mean()
cum_ic = ic.cumsum()

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#222222",
    "grid.color": "#D0D0D0",
})

fig, ax1 = plt.subplots(figsize=(16, 7))

# 左轴：每日 IC 柱状图 + 滚动 IC
bars = ax1.bar(
    ic.index,
    ic,
    width=0.8,
    color="#4C78A8",
    alpha=0.65,
    label="Daily Rank IC",
)

line_roll, = ax1.plot(
    rolling_ic.index,
    rolling_ic,
    color="#C23B22",
    linewidth=2,
    label=f"{rolling_window}days rolling Rank IC",
)

ax1.axhline(0, color="gray", linewidth=1, linestyle="--")
ax1.set_ylabel("Daily / Rolling Rank IC")
ax1.set_title("Daily IC, Rolling IC and Cumulative Rank IC")

# 右轴：累计 IC
ax2 = ax1.twinx()

line_cum, = ax2.plot(
    cum_ic.index,
    cum_ic,
    color="#2CA02C",
    linewidth=2.2,
    label="Cumulative Rank IC",
)

ax2.set_ylabel("Cumulative Rank IC")

# 横轴日期格式
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.grid(axis="y", alpha=0.25)

# 合并双轴图例
ax1.legend(
    [bars, line_roll, line_cum],
    ["Daily Rank IC", f"{rolling_window}days rolling Rank IC", "Cumulative Rank IC"],
    loc="upper left",
)

fig.tight_layout()
plt.savefig(save_dir / f'{name}_ic.png')

print("本脚本任务结束")