import rqdatac
rqdatac.init()

import numpy as np
import pandas as pd

from rqdatac import (
    all_instruments,
    get_factor,
    get_instrument_industry,
    get_trading_dates,
    get_turnover_rate,
)


def _turn_spill_get_ids(stock_ids):
    if stock_ids is None:
        return (
            all_instruments("CS")["order_book_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    if isinstance(stock_ids, str):
        return [stock_ids]

    return list(dict.fromkeys(stock_ids))


def _turn_spill_get_turnover_wide(
    order_book_ids,
    dates,
    batch_size=500,
):
    """下载日换手率，整理为 日期 × 股票 宽表。"""
    parts = []

    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]

        raw = get_turnover_rate(
            order_book_ids=batch,
            start_date=dates[0],
            end_date=dates[-1],
            fields="today",
            expect_df=True,
        )

        if raw is None or raw.empty:
            continue

        if isinstance(raw.index, pd.MultiIndex):
            turnover = raw["today"].unstack("order_book_id")
        else:
            turnover = raw[["today"]].rename(
                columns={"today": batch[0]}
            )

        turnover.index = pd.to_datetime(turnover.index)

        parts.append(
            turnover.reindex(index=dates, columns=batch)
        )

    if not parts:
        raise RuntimeError("未取得有效换手率数据。")

    turnover_wide = pd.concat(parts, axis=1)
    turnover_wide = turnover_wide.loc[
        :, ~turnover_wide.columns.duplicated()
    ]

    return turnover_wide.reindex(
        index=dates,
        columns=order_book_ids,
    )


def _turn_spill_get_market_cap_wide(
    order_book_ids,
    dates,
    batch_size=500,
):
    """下载总市值 market_cap_3，整理为 日期 × 股票 宽表。"""
    parts = []

    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]

        raw = get_factor(
            order_book_ids=batch,
            factor="market_cap_3",
            start_date=dates[0],
            end_date=dates[-1],
            expect_df=True,
        )

        if raw is None or raw.empty:
            continue

        if isinstance(raw.index, pd.MultiIndex):
            market_cap = raw["market_cap_3"].unstack(
                "order_book_id"
            )
        else:
            market_cap = raw[["market_cap_3"]].rename(
                columns={"market_cap_3": batch[0]}
            )

        market_cap.index = pd.to_datetime(market_cap.index)

        parts.append(
            market_cap.reindex(index=dates, columns=batch)
        )

    if not parts:
        raise RuntimeError("未取得有效市值数据。")

    market_cap_wide = pd.concat(parts, axis=1)
    market_cap_wide = market_cap_wide.loc[
        :, ~market_cap_wide.columns.duplicated()
    ]

    return market_cap_wide.reindex(
        index=dates,
        columns=order_book_ids,
    )


def _turn_spill_get_industry_codes(
    order_book_ids,
    asof_date,
    industry_source="citics_2019",
    industry_level=1,
    batch_size=500,
):
    """取得指定日期的行业代码。"""
    field_map = {
        1: "first_industry_code",
        2: "second_industry_code",
        3: "third_industry_code",
    }

    industry_field = field_map[industry_level]
    parts = []

    for start in range(0, len(order_book_ids), batch_size):
        batch = order_book_ids[start:start + batch_size]

        raw = get_instrument_industry(
            order_book_ids=batch,
            source=industry_source,
            level=industry_level,
            date=asof_date,
        )

        if raw is not None and not raw.empty:
            parts.append(raw[industry_field])

    if not parts:
        return pd.Series(index=order_book_ids, dtype=object)

    industry_codes = pd.concat(parts)
    industry_codes = industry_codes[
        ~industry_codes.index.duplicated(keep="last")
    ]

    return industry_codes.reindex(order_book_ids)


def calculate_turnover_spill_daily(
    start_date,
    end_date,
    attention_window=20,
    candidate_ids=None,
    peer_ids=None,
    industry_source="citics_2019",
    industry_level=1,
    industry_refresh="M",
    min_group_stocks=2,
    batch_size=500,
):
    """
    基于换手率的注意力溢出因子

    输出：
        index  : [date, order_book_id]
        column : SPILL_TURN

    注意：
        candidate_ids 是输出股票池；
        peer_ids 是计算同行业、同市值组均值的股票池。
        两者均为 None 时，使用全 A 股。
    """
    if attention_window < 1:
        raise ValueError("attention_window 必须为正整数。")

    if industry_level not in (1, 2, 3):
        raise ValueError("industry_level 仅支持 1、2、3。")

    if industry_refresh not in ("M", "Q", "D"):
        raise ValueError(
            "industry_refresh 仅支持 'M'、'Q'、'D'。"
        )

    candidate_ids = _turn_spill_get_ids(candidate_ids)
    peer_ids = _turn_spill_get_ids(peer_ids)

    # 保证待输出股票也属于同组比较范围
    peer_ids = list(dict.fromkeys(peer_ids + candidate_ids))

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    calendar = pd.DatetimeIndex(
        get_trading_dates(
            start_date=start_date - pd.Timedelta(
                days=max(attention_window * 3, 90)
            ),
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

    if first_loc < attention_window - 1:
        raise ValueError("历史交易日不足，无法计算平均换手率。")

    attention_dates = calendar[
        first_loc - attention_window + 1:last_loc + 1
    ]

    # 1. ATTN：过去 attention_window 个交易日的平均换手率
    turnover_wide = _turn_spill_get_turnover_wide(
        order_book_ids=peer_ids,
        dates=attention_dates,
        batch_size=batch_size,
    )

    attention_wide = (
        turnover_wide
        .rolling(
            window=attention_window,
            min_periods=attention_window,
        )
        .mean()
        .reindex(target_dates)
    )

    # 2. 每日市值；行业归属在月初、季初或每日更新
    market_cap_wide = _turn_spill_get_market_cap_wide(
        order_book_ids=peer_ids,
        dates=target_dates,
        batch_size=batch_size,
    )

    industry_wide = pd.DataFrame(
        index=target_dates,
        columns=peer_ids,
        dtype=object,
    )

    if industry_refresh == "D":
        refresh_mask = np.ones(len(target_dates), dtype=bool)
    else:
        periods = target_dates.to_period(industry_refresh)
        refresh_mask = ~periods.duplicated()

    refresh_locs = np.flatnonzero(refresh_mask)

    for pos, begin_loc in enumerate(refresh_locs):
        end_loc = (
            refresh_locs[pos + 1]
            if pos + 1 < len(refresh_locs)
            else len(target_dates)
        )
        block_dates = target_dates[begin_loc:end_loc]

        industry_codes = _turn_spill_get_industry_codes(
            order_book_ids=peer_ids,
            asof_date=block_dates[0],
            industry_source=industry_source,
            industry_level=industry_level,
            batch_size=batch_size,
        )

        industry_wide.loc[block_dates, peer_ids] = np.tile(
            industry_codes.values,
            (len(block_dates), 1),
        )

    # 3. 每日按 行业 × 市值 3:4:3 分组，计算剔除自身后的均值
    factor_wide = pd.DataFrame(
        index=target_dates,
        columns=candidate_ids,
        dtype=float,
    )

    for date in target_dates:
        day = pd.DataFrame({
            "attention": attention_wide.loc[date, peer_ids],
            "market_cap": market_cap_wide.loc[date, peer_ids],
            "industry": industry_wide.loc[date, peer_ids],
        })

        day = day.dropna(
            subset=["attention", "market_cap", "industry"]
        )
        day = day[day["market_cap"] > 0]

        if day.empty:
            continue

        # 行业内按市值由大至小排序
        day["rank"] = day.groupby("industry")[
            "market_cap"
        ].rank(
            method="first",
            ascending=False,
        )

        day["industry_count"] = day.groupby("industry")[
            "market_cap"
        ].transform("count")

        # 3:4:3 市值分组
        large_cutoff = np.ceil(
            day["industry_count"] * 0.3
        )
        middle_cutoff = np.ceil(
            day["industry_count"] * 0.7
        )

        day["size_group"] = np.where(
            day["rank"] <= large_cutoff,
            "L",
            np.where(day["rank"] <= middle_cutoff, "M", "S"),
        )

        day["group_key"] = (
            day["industry"].astype(str)
            + "_"
            + day["size_group"]
        )

        group_sum = day.groupby("group_key")[
            "attention"
        ].transform("sum")

        group_count = day.groupby("group_key")[
            "attention"
        ].transform("count")

        peer_attention_mean = (
            (group_sum - day["attention"])
            / (group_count - 1).replace(0, np.nan)
        )

        day["SPILL_TURN"] = (
            day["attention"] - peer_attention_mean
        ).where(group_count >= min_group_stocks)

        output = day.loc[
            day.index.isin(candidate_ids),
            "SPILL_TURN",
        ]

        factor_wide.loc[date, output.index] = output

    result = (
        factor_wide
        .stack(dropna=True)
        .rename("SPILL_TURN")
        .to_frame()
    )
    result.index = result.index.set_names(
        ["date", "order_book_id"]
    )

    return result

turnover_spill = calculate_turnover_spill_daily(
    start_date="2020-01-01",
    end_date="2025-12-31",
    attention_window=20,
    candidate_ids=None,
    peer_ids=None,
    industry_source="citics_2019",
    industry_level=1,
    industry_refresh="M",
    min_group_stocks=2,
    batch_size=500,
)

turnover_spill_df = (
    turnover_spill["SPILL_TURN"]
    .unstack("order_book_id")
    .sort_index()
)

turnover_spill_df.index = pd.to_datetime(turnover_spill_df.index)
turnover_spill_df.columns.name = "order_book_id"

from rqfactor.notebook import factor_analysis

factor_df = turnover_spill_df
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

folder_name = "基于换手率的注意力溢出（SPILL）"
name = "turnover_spill_daily_none_q10"
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