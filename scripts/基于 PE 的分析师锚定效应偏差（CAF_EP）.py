from pathlib import Path

import numpy as np
import pandas as pd

import rqdatac
rqdatac.init()

FPE_FIELD = "comp_con_pe_ftm"
INDUSTRY_FIELD = "first_industry_code"

def _get_dated_files(directory):
    """
    读取 YYYY-MM-DD.parquet 文件并返回：
    {Timestamp: Path}
    """
    directory = Path(directory)
    result = {}

    for file_path in directory.glob("*.parquet"):
        try:
            result[pd.Timestamp(file_path.stem)] = file_path
        except ValueError:
            continue

    if not result:
        raise FileNotFoundError(
            "目录中未找到日期命名的 parquet 文件：{}".format(
                directory
            )
        )

    return result


def _read_positive_fpe(file_path, calculation_ids=None):
    """
    读取单日预测 PE。

    非正预测 PE 无经济含义，按缺失值处理；
    后续会以前一有效值向前填充。
    """
    data = pd.read_parquet(
        file_path,
        columns=["order_book_id", FPE_FIELD],
    )

    required_columns = {"order_book_id", FPE_FIELD}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "{} 缺少字段：{}".format(
                file_path.name,
                sorted(missing_columns),
            )
        )

    data[FPE_FIELD] = pd.to_numeric(
        data[FPE_FIELD],
        errors="coerce",
    )

    data = data.loc[
        data[FPE_FIELD] > 0,
        ["order_book_id", FPE_FIELD],
    ]

    if calculation_ids is not None:
        data = data[
            data["order_book_id"].isin(calculation_ids)
        ]

    return (
        data
        .drop_duplicates(
            subset=["order_book_id"],
            keep="last",
        )
        .set_index("order_book_id")[FPE_FIELD]
    )


def _read_industry(file_path):
    """
    读取单个行业分类快照。

    本地数据中使用一级中信行业：
    first_industry_code
    """
    data = pd.read_parquet(
        file_path,
        columns=["order_book_id", INDUSTRY_FIELD],
    )

    required_columns = {
        "order_book_id",
        INDUSTRY_FIELD,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "{} 缺少字段：{}".format(
                file_path.name,
                sorted(missing_columns),
            )
        )

    return (
        data
        .dropna(subset=["order_book_id"])
        .drop_duplicates(
            subset=["order_book_id"],
            keep="last",
        )
        .set_index("order_book_id")[INDUSTRY_FIELD]
    )


def _normalize_ids(ids):
    if ids is None:
        return None

    if isinstance(ids, str):
        return {ids}

    return set(dict.fromkeys(ids))


def calculate_caf_ep_daily_from_local(
    start_date,
    end_date,
    candidate_ids=None,
    industry_universe_ids=None,
    data_dir="data",
    min_industry_stocks=5,
    initial_lookback_days=370,
):
    """
    基于本地数据构建日频 CAF_EP 因子。

    Parameters
    ----------
    start_date, end_date : str
        输出因子的起止日期。

    candidate_ids : list[str] or str, optional
        因子输出股票池。None 表示所有可计算股票。

    industry_universe_ids : list[str] or str, optional
        计算行业预测 PE 中位数的股票池。
        None 表示全部可取得有效预测 PE 的股票。

    data_dir : str
        本地数据目录，默认 data。

    min_industry_stocks : int
        行业内当日有效预测 PE 股票数下限。
        少于该数量的行业不输出因子值。

    initial_lookback_days : int
        向前读取预测 PE 的自然日窗口，用于前向填充。

    Returns
    -------
    pd.DataFrame
        MultiIndex: [date, order_book_id]
        Column: CAF_EP
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date。")

    if min_industry_stocks < 1:
        raise ValueError(
            "min_industry_stocks 至少应为 1。"
        )

    data_dir = Path(data_dir)

    fpe_dir = (
        data_dir
        / "get_comp_indicators"
        / FPE_FIELD
    )

    industry_dir = (
        data_dir
        / "get_instrument_industry_monthly"
    )

    fpe_files = _get_dated_files(fpe_dir)
    industry_files = _get_dated_files(industry_dir)

    fpe_dates = pd.DatetimeIndex(
        sorted(fpe_files.keys())
    )

    industry_dates = pd.DatetimeIndex(
        sorted(industry_files.keys())
    )

    candidate_set = _normalize_ids(candidate_ids)
    industry_universe_set = _normalize_ids(
        industry_universe_ids
    )

    # 确保待输出股票也纳入行业中位数的计算范围
    if (
        candidate_set is None
        and industry_universe_set is None
    ):
        calculation_ids = None
    else:
        calculation_ids = (
            (candidate_set or set())
            | (industry_universe_set or set())
        )

    lookback_start = (
        start_date
        - pd.Timedelta(days=initial_lookback_days)
    )

    process_dates = fpe_dates[
        (fpe_dates >= lookback_start)
        & (fpe_dates <= end_date)
    ]

    target_dates = process_dates[
        process_dates >= start_date
    ]

    if len(target_dates) == 0:
        raise ValueError(
            "所选区间内没有可用的预测 PE 交易日。"
        )

    # 保存每只股票最近一次有效的正预测 PE。
    # 对应原始代码中：FPE > 0 后再 ffill。
    last_valid_fpe = {}

    current_industry = pd.Series(dtype=object)
    current_industry_position = None

    result_parts = []

    for date in process_dates:
        # 更新当日有效预测 PE
        daily_fpe = _read_positive_fpe(
            file_path=fpe_files[date],
            calculation_ids=calculation_ids,
        )

        if not daily_fpe.empty:
            last_valid_fpe.update(
                daily_fpe.to_dict()
            )

        # lookback 期只用于建立 FPE 前值，不输出因子
        if date < start_date or not last_valid_fpe:
            continue

        # 使用当前日期之前最近一期的月度行业分类
        industry_position = (
            industry_dates.searchsorted(
                date,
                side="right",
            )
            - 1
        )

        if industry_position < 0:
            continue

        if industry_position != current_industry_position:
            current_industry_position = industry_position

            current_industry = _read_industry(
                industry_files[
                    industry_dates[industry_position]
                ]
            )

        # 当日可用的、前向填充后的 FPE
        fpe = pd.Series(
            last_valid_fpe,
            name="FPE",
            dtype=float,
        )

        frame = pd.DataFrame({
            "FPE": fpe,
            "industry": current_industry.reindex(
                fpe.index
            ),
        })

        frame = frame.dropna(
            subset=["FPE", "industry"]
        )

        # 计算行业预测 PE 中位数时使用的股票池
        if industry_universe_set is None:
            median_universe = frame
        else:
            median_universe = frame.loc[
                frame.index.isin(industry_universe_set)
            ]

        if median_universe.empty:
            continue

        industry_median_fpe = (
            median_universe
            .groupby("industry")["FPE"]
            .median()
        )

        industry_fpe_count = (
            median_universe
            .groupby("industry")["FPE"]
            .count()
        )

        eligible_industries = industry_fpe_count[
            industry_fpe_count >= min_industry_stocks
        ].index

        if len(eligible_industries) == 0:
            continue

        frame = frame.loc[
            frame["industry"].isin(eligible_industries)
        ].copy()

        frame["industry_median_fpe"] = (
            frame["industry"].map(industry_median_fpe)
        )

        # CAF_EP
        # = (1 / 个股 FPE - 1 / 行业 FPE 中位数)
        #   / (1 / 行业 FPE 中位数)
        # = 行业 FPE 中位数 / 个股 FPE - 1
        frame["CAF_EP"] = (
            frame["industry_median_fpe"]
            / frame["FPE"]
            - 1.0
        )

        # 输出股票池筛选
        if candidate_set is not None:
            frame = frame.loc[
                frame.index.isin(candidate_set)
            ]

        if frame.empty:
            continue

        result_part = (
            frame[["CAF_EP"]]
            .reset_index()
            .rename(
                columns={"index": "order_book_id"}
            )
        )

        result_part.insert(0, "date", date)

        result_parts.append(result_part)

    if not result_parts:
        raise RuntimeError(
            "未生成 CAF_EP 因子，请检查数据路径、"
            "行业分类覆盖与 min_industry_stocks 设置。"
        )

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(["date", "order_book_id"])
        .set_index(["date", "order_book_id"])
    )

    result.index = result.index.set_names(
        ["date", "order_book_id"]
    )

    return result

# 全市场日频 CAF_EP
caf_ep = calculate_caf_ep_daily_from_local(
    start_date="2020-01-01",
    end_date="2025-12-31",
    candidate_ids=None,
    industry_universe_ids=None,
    data_dir="data",
    min_industry_stocks=5,
    initial_lookback_days=370,
)

caf_ep_df = (
    caf_ep["CAF_EP"]
    .unstack("order_book_id")
    .sort_index()
)

caf_ep_df.index = pd.to_datetime(caf_ep_df.index)
caf_ep_df.columns.name = "order_book_id"

from rqfactor.notebook import factor_analysis

factor_df = caf_ep_df
result = factor_analysis(
    f=factor_df,
    period=1,
    shift=1,
    rank_ic=True,
    quantile=10,
    ascending=False,
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

folder_name = "基于 PE 的分析师锚定效应偏差（CAF_EP）"
name = "caf_ep_daily_none_q10"
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