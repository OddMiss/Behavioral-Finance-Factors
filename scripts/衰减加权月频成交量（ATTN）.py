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

def _stack_factor(factor_wide, factor_name):
    """日期 × 股票宽表转为 date、order_book_id 两层索引。"""
    panel = factor_wide.stack(dropna=True).rename(factor_name).to_frame()
    panel.index = panel.index.set_names(["date", "order_book_id"])
    return panel

def calculate_attn_daily(
    start_date,
    end_date,
    completed_months=12,
    candidate_ids=None,
    batch_size=500,
):
    """
    计算衰减加权月频成交量（ATTN），并映射为日频因子。

    对当前输出月 M，使用此前 12 个已完成月：
    M-1、M-2、...、M-12。

    权重为：
    12/78、11/78、...、1/78。

    因子在每月第一个交易日更新，月内保持不变。
    """
    if completed_months != 12:
        raise ValueError("当前版本严格对应 12 个月线性衰减，completed_months 必须为 12。")

    # 12 个完整月历史，加上数据冗余。
    required_lookback = 400

    target_dates, volume_dates = _prepare_dates(
        start_date=start_date,
        end_date=end_date,
        lookback_days=required_lookback,
    )

    candidate_ids = _get_ids(candidate_ids)
    factor_parts = []

    # 12 + 11 + ... + 1 = 78
    weight_denominator = completed_months * (completed_months + 1) / 2.0

    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]

        _, volume = _get_batch_close_volume(batch, volume_dates)

        if volume is None:
            continue

        month_index = volume.index.to_period("M")

        monthly_volume = volume.groupby(month_index).sum()

        monthly_trade_days = (
            volume.gt(0)
            .groupby(month_index)
            .sum()
        )

        monthly_attn = pd.DataFrame(
            0.0,
            index=monthly_volume.index,
            columns=batch,
        )

        valid_history = pd.DataFrame(
            True,
            index=monthly_volume.index,
            columns=batch,
        )

        # 输出月 M 使用 M-1 至 M-12 的完整月数据。
        for lag in range(1, completed_months + 1):
            weight = (
                completed_months + 1 - lag
            ) / weight_denominator

            monthly_attn += (
                weight * monthly_volume.shift(lag)
            )

            valid_history &= monthly_trade_days.shift(lag).gt(0)

        monthly_attn = monthly_attn.where(valid_history)

        # 映射到日频：每月首个交易日开始使用最新已完成月的数据。
        daily_attn = monthly_attn.reindex(month_index)
        daily_attn.index = volume.index

        # 仅在股票当日实际交易时保留因子。
        daily_attn = daily_attn.where(volume.gt(0))

        factor_wide = daily_attn.reindex(target_dates)

        factor_parts.append(
            _stack_factor(factor_wide, "ATTN")
        )

    if not factor_parts:
        return pd.DataFrame(columns=["ATTN"])

    return pd.concat(factor_parts).sort_index()

attn_panel = calculate_attn_daily(
    start_date="2020-01-01",
    end_date="2025-12-31",
)

# 行：交易日；列：order_book_id；值：ABNRETD
attn_panel_df = (
    attn_panel["ATTN"]
    .unstack("order_book_id")
    .sort_index()
)

attn_panel_df.index = pd.to_datetime(attn_panel_df.index)
attn_panel_df.columns.name = "order_book_id"

from rqfactor.notebook import factor_analysis

factor_df = attn_panel_df
result = factor_analysis(
    f=factor_df,
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

print("回测已完成")

from pathlib import Path
import json
import pandas as pd

folder_name = "衰减加权月频成交量（ATTN）"
name = "attn_panel_daily_none_q10"
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