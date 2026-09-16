import pandas as pd

from rqdatac import (
    all_instruments,
    get_trading_dates,
    get_turnover_rate,
)

import rqdatac
rqdatac.init()

def calculate_turnavg_daily(
    start_date,
    end_date,
    window=20,
    candidate_ids=None,
    batch_size=500,
    output="wide",
):
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
        raise ValueError("window 必须是正整数。")

    if output not in ("wide", "panel"):
        raise ValueError('output 仅支持 "wide" 或 "panel"。')

    if candidate_ids is None:
        candidate_ids = (
            all_instruments("CS")["order_book_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
    else:
        candidate_ids = list(
            pd.Index(candidate_ids).dropna().unique()
        )

    if not candidate_ids:
        raise ValueError("candidate_ids 为空。")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # 额外获取历史交易日，以保证起始日也能计算完整 20 日均值。
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            get_trading_dates(
                start_date=start_date - pd.Timedelta(
                    days=max(window * 3, 90)
                ),
                end_date=end_date,
            )
        )
    )

    target_dates = calendar[
        (calendar >= start_date)
        & (calendar <= end_date)
    ]

    if len(target_dates) == 0:
        raise ValueError("目标区间内没有交易日。")

    first_target_position = calendar.get_loc(target_dates[0])

    if first_target_position < window - 1:
        raise ValueError("历史交易日不足，无法计算完整 TURNAVG。")

    fetch_dates = calendar[
        first_target_position - window + 1:
    ]

    turnover_parts = []

    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[start:start + batch_size]

        raw = get_turnover_rate(
            order_book_ids=batch_ids,
            start_date=fetch_dates[0],
            end_date=fetch_dates[-1],
            fields="today",
            expect_df=True,
        )

        if raw is None or raw.empty:
            continue

        if isinstance(raw.index, pd.MultiIndex):
            turnover = raw["today"].unstack("order_book_id")
        else:
            # 单只股票请求时的兼容处理
            turnover = raw[["today"]].rename(
                columns={"today": batch_ids[0]}
            )

        turnover.index = pd.to_datetime(turnover.index)

        turnover_parts.append(
            turnover.reindex(
                index=fetch_dates,
                columns=batch_ids,
            )
        )

    if not turnover_parts:
        raise RuntimeError("未取得有效换手率数据，请检查接口权限。")

    turnover_wide = pd.concat(turnover_parts, axis=1)
    turnover_wide = turnover_wide.loc[
        :,
        ~turnover_wide.columns.duplicated()
    ]

    turnover_wide = turnover_wide.reindex(
        index=fetch_dates,
        columns=candidate_ids,
    ).astype(float)

    # 截至 t 的过去 window 个交易日平均换手率
    turnavg_wide = turnover_wide.rolling(
        window=window,
        min_periods=window,
    ).mean()

    factor_df = turnavg_wide.reindex(target_dates)
    factor_df.index = pd.DatetimeIndex(factor_df.index)
    factor_df.index.name = "date"
    factor_df.columns.name = "order_book_id"

    if output == "wide":
        return factor_df

    factor_panel = factor_df.stack(dropna=True).to_frame(
        "TURNAVG"
    )
    factor_panel.index.names = ["date", "order_book_id"]

    return factor_panel

turnavg_df = calculate_turnavg_daily(
    start_date="2020-01-01",
    end_date="2025-12-31",
    window=20,
    candidate_ids=None,
    batch_size=500,
    output="wide",
)

from rqfactor.notebook import factor_analysis

result_factor = factor_analysis(
    f=turnavg_df,
    period=1,
    shift=1,
    rank_ic=True,
    quantile=10,
    ascending=True,
    winzorization="mad",
    normalization=False,
    neutralization="none",
    include_st=False,
    include_new=False,
    benchmark="000985.XSHG",
)

from pathlib import Path
import json
import pandas as pd

result = result_factor
folder_name = "平均换手率（TURNAVG）"
name = "turnavg_daily_none_q10"
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