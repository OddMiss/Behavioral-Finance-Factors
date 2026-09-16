from rqdatac import get_turnover_rate
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


def _get_batch_turnover(batch, dates):
    """
    下载日换手率，返回：日期 × 股票 的宽表。
    字段 today 的单位通常是百分数，例如 0.519 表示 0.519%。
    """
    raw = get_turnover_rate(
        order_book_ids=batch,
        start_date=dates[0],
        end_date=dates[-1],
        fields="today",
        expect_df=True,
    )

    if raw is None or raw.empty:
        return None

    if isinstance(raw.index, pd.MultiIndex):
        turnover = raw["today"].unstack("order_book_id")
    else:
        turnover = raw[["today"]].rename(columns={"today": batch[0]})

    turnover.index = pd.to_datetime(turnover.index)

    return turnover.reindex(index=dates, columns=batch)

def calculate_cgo_daily(
    start_date,
    end_date,
    lookback_window=60,
    candidate_ids=None,
    turnover_is_percent=True,
    batch_size=500,
):
    """
    计算日频未实现盈利量（CGO）。

    RP_t = sum_n[w_t,n * P_t-n] / sum_n[w_t,n]

    w_t,n = V_t-n * product(1 - V_t-tau)
    其中 V 为换手率，后续换手越高，较早成交价格的权重越低。

    CGO_t = (P_t - RP_t) / RP_t

    返回
    ----
    DataFrame
        MultiIndex：date、order_book_id
        Column：CGO
    """
    target_dates, price_dates = _prepare_dates(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_window,
    )

    candidate_ids = _get_ids(candidate_ids)
    factor_parts = []

    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]

        # 前复权收盘价用于保证跨除权日的价格可比。
        close, volume = _get_batch_close_volume(batch, price_dates)
        turnover = _get_batch_turnover(batch, price_dates)

        if close is None or turnover is None:
            continue

        # CGO 公式中的 V 必须是 0～1 的比例。
        if turnover_is_percent:
            turnover = turnover / 100.0

        turnover = turnover.clip(lower=0.0, upper=1.0)

        # 停牌／缺失换手率视为未发生筹码交换。
        turnover = turnover.fillna(0.0)

        weight_sum = pd.DataFrame(
            0.0,
            index=price_dates,
            columns=batch,
        )

        weighted_price_sum = pd.DataFrame(
            0.0,
            index=price_dates,
            columns=batch,
        )

        # survival 表示：从历史成交日至今，筹码仍未换手的概率。
        survival = pd.DataFrame(
            1.0,
            index=price_dates,
            columns=batch,
        )

        for lag in range(1, lookback_window + 1):
            turnover_lag = turnover.shift(lag)
            price_lag = close.shift(lag)

            # w_t,n = V_t-n × Π(1 - V_t-tau)
            weight = turnover_lag * survival

            # 价格缺失时不将该观察计入参考价格。
            weight = weight.where(price_lag.notna(), 0.0)

            weighted_price_sum += weight * price_lag.fillna(0.0)
            weight_sum += weight

            # 为下一期更早的历史价格更新留存概率。
            survival *= (1.0 - turnover_lag)

        reference_price = (
            weighted_price_sum
            / weight_sum.replace(0.0, np.nan)
        )

        cgo = (
            (close - reference_price)
            / reference_price
        )

        # CGO 输出要求：
        # 1）当前日有实际成交；
        # 2）此前拥有完整 lookback_window 日价格历史。
        # 3) 防止极少数零价格记录进入窗口
        current_active = volume.gt(0) & close.gt(0)

        history_complete = (
            close.gt(0)
            .rolling(
                window=lookback_window,
                min_periods=lookback_window,
            )
            .sum()
            .shift(1)
            .eq(lookback_window)
        )

        cgo = cgo.where(current_active & history_complete)
        cgo = cgo.reindex(target_dates)

        factor_parts.append(_stack_factor(cgo, "CGO"))

    if not factor_parts:
        return pd.DataFrame(columns=["CGO"])

    return pd.concat(factor_parts).sort_index()

cgo_panel = calculate_cgo_daily(
    start_date="2020-01-01",
    end_date="2025-12-31",
    lookback_window=60,   # 可改为 40
    turnover_is_percent=True,
)

# 行：交易日；列：order_book_id；值：ABNRETD
cgo_panel_df = (
    cgo_panel["CGO"]
    .unstack("order_book_id")
    .sort_index()
)

cgo_panel_df.index = pd.to_datetime(cgo_panel_df.index)
cgo_panel_df.columns.name = "order_book_id"

from rqfactor.notebook import factor_analysis

result_cgo = factor_analysis(
    f=cgo_panel_df,
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

result = result_cgo
folder_name = "未实现盈利量（CGO）"
name = "cgo_daily_none_q10"
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