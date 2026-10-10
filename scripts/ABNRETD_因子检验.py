"""ABNRETD（最大异常日收益率）的计算与研究检验。

本文件可以在 Ricequant 研究环境或已初始化 ``rqdatac`` 的本地环境运行。
它把因子研究分为三层：

1. 数据质量：样本覆盖、重复值、缺失、非有限值、异常范围、市场样本数；
2. 公式/经济逻辑：用可手算的模拟数据验证公式，并检查它与 MAXRET、波动率的
   机械关系是否符合定义；
3. 有效性：以 T 日收盘后可获得的因子预测 T+1 开盘至未来第 H 日收盘的收益，
   输出 Rank IC、分组收益和前后半样本稳定性。

重要：研究股票池必须使用每个历史时点实际可投资的股票池。例如沪深 300 应按
历史成分股过滤；不能用今天的成分股回填历史，否则会产生幸存者偏差。

仅用于研究，不构成投资建议。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from rqdatac import all_instruments, get_price, get_trading_dates, index_components
except ImportError:  # 允许在没有米筐环境时运行纯公式单测。
    all_instruments = get_price = get_trading_dates = index_components = None


FACTOR_NAME = "ABNRETD"
DATE_LEVEL = "date"
ID_LEVEL = "order_book_id"


@dataclass
class ABNRETDCalculation:
    """因子计算结果及可用于审计的附加信息。"""

    factor: pd.DataFrame
    market_diagnostics: pd.DataFrame
    components: pd.DataFrame | None = None


@dataclass
class ABNRETDValidation:
    """完整研究流程返回的表格。"""

    factor: pd.DataFrame
    forward_returns: pd.DataFrame
    data_quality_summary: pd.DataFrame
    data_quality_by_date: pd.DataFrame
    mechanism_summary: pd.DataFrame
    ic_by_date: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_returns: pd.DataFrame
    quantile_summary: pd.DataFrame


def _require_rqdatac() -> None:
    if any(item is None for item in (all_instruments, get_price, get_trading_dates)):
        raise ImportError(
            "未检测到 rqdatac。请在米筐研究环境运行，或先安装并初始化 rqdatac。"
        )


def _get_ids(stock_ids: Iterable[str] | None = None) -> list[str]:
    """返回去重后的股票代码；未指定时取米筐 CS 股票全集。"""
    if stock_ids is not None:
        return list(dict.fromkeys(stock_ids))

    _require_rqdatac()
    return (
        all_instruments("CS")
        .order_book_id.dropna()
        .drop_duplicates()
        .tolist()
    )


def get_historical_index_universe(
    index_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """生成指数的逐日历史成分股面板，避免把今天的成分股用于历史回测。

    返回的索引为 ``(date, order_book_id)``，列 ``eligible`` 恒为 True。函数逐日
    查询是为了保留临时调入调出等变化；研究较长区间时请求会较多，但不会引入
    成分股的未来信息。
    """
    _require_rqdatac()
    if index_components is None:
        raise ImportError("当前 rqdatac 版本不支持 index_components。")
    dates = pd.DatetimeIndex(get_trading_dates(start_date=start_date, end_date=end_date))
    parts: list[pd.DataFrame] = []
    for date in dates:
        ids = list(dict.fromkeys(index_components(index_id, date=date)))
        if not ids:
            continue
        index = pd.MultiIndex.from_product([[date], ids], names=[DATE_LEVEL, ID_LEVEL])
        parts.append(pd.DataFrame({"eligible": True}, index=index))
    if not parts:
        raise ValueError(f"{index_id} 在所选日期内没有返回历史成分股。")
    return pd.concat(parts).sort_index()


def filter_factor_to_universe(
    factor_panel: pd.DataFrame,
    expected_universe: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """仅保留当日属于历史可投资股票池的因子观测。"""
    if isinstance(expected_universe, pd.Series):
        eligible = expected_universe
    else:
        if "eligible" in expected_universe:
            eligible = expected_universe["eligible"]
        elif len(expected_universe.columns) == 1:
            eligible = expected_universe.iloc[:, 0]
        else:
            raise ValueError("expected_universe 需要名为 eligible 的列或仅含一列。")
    if not isinstance(eligible.index, pd.MultiIndex) or eligible.index.nlevels != 2:
        raise ValueError("expected_universe 索引必须为 (date, order_book_id)。")
    eligible = eligible[eligible.astype(bool)]
    eligible.index = eligible.index.set_names([DATE_LEVEL, ID_LEVEL])
    return factor_panel.loc[factor_panel.index.isin(eligible.index)].sort_index()


def select_rebalance_dates(
    panel: pd.DataFrame,
    frequency: str = "M",
) -> pd.DataFrame:
    """从日频面板选择信号/调仓日。

    ``frequency='M'`` 取每个自然月最后一个可用交易日，和“月内最大异常日收益”
    的常见研究口径一致；``'D'`` 保留每日滚动 20 日版本。该函数只选择已形成的
    信号，不会改变每个信号所使用的历史窗口。
    """
    frequency = frequency.upper()
    if frequency == "D":
        return panel.sort_index()
    if frequency != "M":
        raise ValueError("frequency 仅支持 'D'（日频）或 'M'（月末）。")
    dates = pd.DatetimeIndex(panel.index.get_level_values(DATE_LEVEL).unique()).sort_values()
    selected = pd.Series(dates, index=dates.to_period("M")).groupby(level=0).max()
    return panel.loc[panel.index.get_level_values(DATE_LEVEL).isin(selected.to_numpy())].sort_index()


def _prepare_dates(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    lookback_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """取得输出日与为滚动窗口额外展开的价格日期。"""
    _require_rqdatac()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date 不能早于 start_date。")

    calendar = pd.DatetimeIndex(
        get_trading_dates(
            start_date=start - pd.Timedelta(days=max(lookback_days * 3, 90)),
            end_date=end,
        )
    )
    target_dates = calendar[(calendar >= start) & (calendar <= end)]
    if target_dates.empty:
        raise ValueError("所选区间内没有交易日。")

    first = calendar.get_loc(target_dates[0])
    last = calendar.get_loc(target_dates[-1])
    if first < lookback_days:
        raise ValueError("历史交易日不足，无法满足滚动窗口。")
    return target_dates, calendar[first - lookback_days : last + 1]


def _get_batch_prices(
    batch: Sequence[str],
    price_dates: pd.DatetimeIndex,
    fields: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """下载一批股票的日线字段，统一为 日期 × 股票 的宽表。"""
    _require_rqdatac()
    raw = get_price(
        order_book_ids=list(batch),
        start_date=price_dates[0],
        end_date=price_dates[-1],
        frequency="1d",
        fields=list(fields),
        adjust_type="pre",
        skip_suspended=False,
        expect_df=True,
    )
    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.index, pd.MultiIndex):
        for field in fields:
            result[field] = raw[field].unstack(ID_LEVEL)
    else:  # 米筐对单股票请求返回普通 DatetimeIndex。
        for field in fields:
            result[field] = raw[[field]].rename(columns={field: batch[0]})

    for field, values in result.items():
        values.index = pd.to_datetime(values.index)
        result[field] = values.reindex(index=price_dates, columns=batch)
    return result


def _get_returns(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """计算前复权收盘收益，并仅保留当日确有成交的股票。"""
    returns = close.pct_change(fill_method=None).iloc[1:]
    return returns.where(volume.iloc[1:].gt(0))


def _stack_factor(factor_wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """日期 × 股票宽表转为 date、order_book_id 两层索引。"""
    panel = factor_wide.stack(dropna=True).rename(name).to_frame()
    panel.index = panel.index.set_names([DATE_LEVEL, ID_LEVEL])
    return panel


def abnretd_from_returns(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """纯计算函数：ABNRETD = rolling_max(abs(r_i - r_m))。

    ``returns`` 必须是日期 × 股票、且 index 与 ``market_returns`` 对齐的日收益率。
    使用 ``min_periods=window``，确保每个因子值都由完整的窗口形成。
    """
    if window <= 0:
        raise ValueError("window 必须为正整数。")
    abnormal = returns.sub(market_returns.reindex(returns.index), axis=0).abs()
    return abnormal.rolling(window=window, min_periods=window).max()


def calculate_abnretd_daily(
    start_date: str,
    end_date: str,
    window: int = 20,
    candidate_ids: Iterable[str] | None = None,
    market_ids: Iterable[str] | None = None,
    batch_size: int = 500,
    min_market_stocks: int = 500,
    return_details: bool = False,
) -> pd.DataFrame | ABNRETDCalculation:
    """计算日频 ABNRETD。

    因子日期为 T，使用截至 T 收盘的过去 ``window`` 个交易日数据。市场收益为
    同日所有有成交股票的等权平均收益。若当日有效市场股票数少于
    ``min_market_stocks``，该日市场收益及受影响的滚动因子值均置为缺失。

    ``return_details=True`` 会额外返回市场样本数，以及用于公式合理性检查的
    MAXRET 和收益波动率。默认仅返回因子面板，保持与一般因子函数的使用习惯一致。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数。")
    if min_market_stocks <= 0:
        raise ValueError("min_market_stocks 必须为正整数。")

    target_dates, price_dates = _prepare_dates(start_date, end_date, window)
    market_ids_list = _get_ids(market_ids)
    candidate_ids_list = _get_ids(candidate_ids)
    if not market_ids_list or not candidate_ids_list:
        raise ValueError("market_ids 与 candidate_ids 均不能为空。")

    return_dates = price_dates[1:]
    market_return_sum = pd.Series(0.0, index=return_dates)
    market_return_count = pd.Series(0, index=return_dates, dtype="int64")

    # 第一遍：全市场每日等权收益率。不能用 skip_suspended=True，因为多标的查询
    # 不支持该参数；故使用 volume > 0 明确剔除当日停牌股票。
    for begin in range(0, len(market_ids_list), batch_size):
        batch = market_ids_list[begin : begin + batch_size]
        prices = _get_batch_prices(batch, price_dates, fields=["close", "volume"])
        if not prices:
            continue
        returns = _get_returns(prices["close"], prices["volume"])
        market_return_sum = market_return_sum.add(returns.sum(axis=1, skipna=True), fill_value=0.0)
        market_return_count = market_return_count.add(
            returns.notna().sum(axis=1).astype("int64"), fill_value=0
        ).astype("int64")

    market_returns = (market_return_sum / market_return_count.replace(0, np.nan)).where(
        market_return_count >= min_market_stocks
    )

    factor_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    for begin in range(0, len(candidate_ids_list), batch_size):
        batch = candidate_ids_list[begin : begin + batch_size]
        prices = _get_batch_prices(batch, price_dates, fields=["close", "volume"])
        if not prices:
            continue
        returns = _get_returns(prices["close"], prices["volume"])
        factor_wide = abnretd_from_returns(returns, market_returns, window=window)
        factor_wide = factor_wide.reindex(target_dates)
        factor_parts.append(_stack_factor(factor_wide, FACTOR_NAME))

        if return_details:
            # 两个量均与 ABNRETD 使用相同窗口及缺失要求，便于检查因子是否符合
            # “极端收益/波动”这一经济直觉，而非由数据错误驱动。
            maxret = returns.abs().rolling(window, min_periods=window).max().reindex(target_dates)
            volatility = returns.rolling(window, min_periods=window).std(ddof=0).reindex(target_dates)
            component_parts.append(
                pd.concat(
                    [
                        _stack_factor(maxret, "MAX_ABS_RETURN"),
                        _stack_factor(volatility, "RETURN_VOLATILITY"),
                    ],
                    axis=1,
                )
            )

    factor = (
        pd.concat(factor_parts).sort_index()
        if factor_parts
        else pd.DataFrame(columns=[FACTOR_NAME])
    )
    market_diagnostics = pd.DataFrame(
        {
            "market_return": market_returns.reindex(target_dates),
            "market_stock_count": market_return_count.reindex(target_dates),
        }
    )
    market_diagnostics.index.name = DATE_LEVEL
    market_diagnostics["factor_stock_count"] = factor.groupby(level=DATE_LEVEL).size().reindex(
        target_dates, fill_value=0
    )

    if not return_details:
        return factor
    components = pd.concat(component_parts).sort_index() if component_parts else None
    return ABNRETDCalculation(
        factor=factor,
        market_diagnostics=market_diagnostics,
        components=components,
    )


def calculate_forward_returns(
    start_date: str,
    end_date: str,
    stock_ids: Iterable[str],
    holding_period: int = 20,
    batch_size: int = 500,
) -> pd.DataFrame:
    """构造无未来函数的可交易前瞻收益。

    对 T 日收盘后才计算出的因子，收益定义为 ``close[T+H] / open[T+1] - 1``。
    即最早在下一交易日开盘建仓，持有至信号日之后第 H 个交易日收盘。入场或出场
    日停牌、无价格或无成交时记为缺失；不会以停牌价格虚构可交易收益。
    """
    _require_rqdatac()
    if holding_period <= 0 or batch_size <= 0:
        raise ValueError("holding_period 和 batch_size 必须为正整数。")
    stock_ids_list = _get_ids(stock_ids)
    if not stock_ids_list:
        raise ValueError("stock_ids 不能为空。")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    calendar = pd.DatetimeIndex(
        get_trading_dates(
            start_date=start,
            end_date=end + pd.Timedelta(days=max(holding_period * 3, 30)),
        )
    )
    factor_dates = calendar[(calendar >= start) & (calendar <= end)]
    if factor_dates.empty:
        raise ValueError("所选区间内没有交易日。")
    last_loc = calendar.get_loc(factor_dates[-1])
    if last_loc + holding_period >= len(calendar):
        raise ValueError("前瞻价格日期不足；请扩大数据结束日期。")
    price_dates = calendar[: last_loc + holding_period + 1]

    parts: list[pd.DataFrame] = []
    for begin in range(0, len(stock_ids_list), batch_size):
        batch = stock_ids_list[begin : begin + batch_size]
        prices = _get_batch_prices(batch, price_dates, fields=["open", "close", "volume"])
        if not prices:
            continue
        entry_open = prices["open"].shift(-1)
        exit_close = prices["close"].shift(-holding_period)
        entry_traded = prices["volume"].shift(-1).gt(0)
        exit_traded = prices["volume"].shift(-holding_period).gt(0)
        forward = (exit_close / entry_open - 1).where(entry_traded & exit_traded)
        parts.append(_stack_factor(forward.reindex(factor_dates), f"FWD_{holding_period}D"))

    return pd.concat(parts).sort_index() if parts else pd.DataFrame(columns=[f"FWD_{holding_period}D"])


def _as_factor_frame(panel: pd.DataFrame | pd.Series, value_name: str) -> pd.DataFrame:
    """校验并标准化成两层索引、单列值的因子面板。"""
    if isinstance(panel, pd.Series):
        panel = panel.to_frame(value_name)
    if value_name not in panel.columns:
        if len(panel.columns) != 1:
            raise ValueError(f"面板中找不到列 {value_name!r}。")
        panel = panel.rename(columns={panel.columns[0]: value_name})
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("面板索引必须为两层 MultiIndex：(date, order_book_id)。")
    result = panel[[value_name]].copy()
    result.index = result.index.set_names([DATE_LEVEL, ID_LEVEL])
    result.index = result.index.set_levels(
        pd.to_datetime(result.index.levels[0]), level=DATE_LEVEL
    )
    return result.sort_index()


def _summary_row(
    check: str,
    status: str,
    value: object,
    threshold: object,
    meaning: str,
) -> dict[str, object]:
    return {
        "check": check,
        "status": status,
        "value": value,
        "threshold": threshold,
        "meaning": meaning,
    }


def audit_factor_data(
    factor_panel: pd.DataFrame,
    market_diagnostics: pd.DataFrame | None = None,
    expected_universe: pd.DataFrame | pd.Series | None = None,
    factor_name: str = FACTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 ABNRETD 数据质量审计表。

    ``expected_universe`` 可传入 date、order_book_id 两层索引的布尔面板，取值为
    True 表示该股票在当日应当可被纳入因子池。它使覆盖率检验能发现历史成分股
    处理错误和大量缺失；未提供时仍会检查因子自身的完整性。
    """
    factor = _as_factor_frame(factor_panel, factor_name)
    values = factor[factor_name]
    n_dates = values.index.get_level_values(DATE_LEVEL).nunique()
    n_stocks = values.index.get_level_values(ID_LEVEL).nunique()
    n_obs = len(values)
    duplicate_count = int(factor.index.duplicated().sum())
    non_finite_count = int((~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))).sum())
    negative_count = int((values < 0).sum())

    rows = [
        _summary_row("重复 date-stock", "PASS" if duplicate_count == 0 else "FAIL", duplicate_count, 0,
                     "同一日期和股票只能有一个因子值。"),
        _summary_row("非有限值", "PASS" if non_finite_count == 0 else "FAIL", non_finite_count, 0,
                     "NaN 不应写入最终面板；inf 往往表示价格或除零错误。"),
        _summary_row("负值", "PASS" if negative_count == 0 else "FAIL", negative_count, 0,
                     "ABNRETD 为绝对异常收益的最大值，理论上不能为负。"),
        _summary_row("样本交易日", "PASS" if n_dates >= 60 else "WARN", n_dates, ">= 60",
                     "少于约三个月日频样本，统计检验的稳定性有限。"),
        _summary_row("因子观测数", "PASS" if n_obs > 0 else "FAIL", n_obs, "> 0",
                     "最终用于横截面检验的有效 date-stock 观测数。"),
        _summary_row("覆盖股票数", "PASS" if n_stocks > 0 else "FAIL", n_stocks, "> 0",
                     "样本期至少出现过一次有效因子值的股票数。"),
    ]

    by_date = values.groupby(level=DATE_LEVEL).agg(
        factor_count="count",
        mean="mean",
        std="std",
        min="min",
        median="median",
        max="max",
    )
    quantiles = values.groupby(level=DATE_LEVEL).quantile([0.01, 0.05, 0.95, 0.99]).unstack()
    quantiles.columns = ["p01", "p05", "p95", "p99"]
    by_date = by_date.join(quantiles)
    zero_share = values.eq(0).groupby(level=DATE_LEVEL).mean().rename("zero_share")
    by_date = by_date.join(zero_share)

    if not by_date.empty:
        rows.extend(
            [
                _summary_row("单日横截面覆盖（P5）", "PASS" if by_date["factor_count"].quantile(0.05) >= 50 else "WARN",
                             round(float(by_date["factor_count"].quantile(0.05)), 1), ">= 50",
                             "每日股票数过少会使 IC 和分组收益不稳定。"),
                _summary_row("横截面无离散度日数", "PASS" if int(by_date["std"].fillna(0).eq(0).sum()) == 0 else "FAIL",
                             int(by_date["std"].fillna(0).eq(0).sum()), 0,
                             "若同日因子全相同，横截面排序没有信息。"),
                _summary_row("极端因子最大值", "WARN" if float(by_date["max"].max()) > 1 else "PASS",
                             round(float(by_date["max"].max()), 6), "<= 1（经验预警）",
                             "超过 100% 的单日异常收益不必然错误，但应逐笔核对复权、停牌复牌和错误价格。"),
            ]
        )

    if expected_universe is not None:
        if isinstance(expected_universe, pd.Series):
            expected = expected_universe.rename("eligible").to_frame()
        else:
            expected = expected_universe.copy()
            if "eligible" not in expected:
                if len(expected.columns) != 1:
                    raise ValueError("expected_universe 需要名为 eligible 的列或仅含一列。")
                expected.columns = ["eligible"]
        expected = _as_factor_frame(expected, "eligible")
        expected = expected[expected["eligible"].astype(bool)]
        expected_count = expected.groupby(level=DATE_LEVEL).size().rename("expected_count")
        observed_count = factor.groupby(level=DATE_LEVEL).size().rename("factor_count")
        coverage = pd.concat([expected_count, observed_count], axis=1).fillna(0)
        coverage["coverage"] = coverage["factor_count"] / coverage["expected_count"].replace(0, np.nan)
        by_date = by_date.drop(columns=["factor_count"], errors="ignore").join(coverage)
        median_coverage = float(coverage["coverage"].median())
        rows.append(
            _summary_row("股票池覆盖率（中位数）", "PASS" if median_coverage >= 0.8 else "WARN",
                         round(median_coverage, 4), ">= 0.80",
                         "分母仅含当日应投资的股票；低覆盖率常见于停牌、新股或股票池定义错误。")
        )

    if market_diagnostics is not None:
        market = market_diagnostics.copy()
        market.index = pd.to_datetime(market.index)
        market.index.name = DATE_LEVEL
        by_date = by_date.join(market, how="left")
        if "market_stock_count" in market:
            min_count = int(market["market_stock_count"].min())
            rows.append(
                _summary_row("市场有效股票数（最小值）", "PASS" if min_count > 0 else "FAIL", min_count, "> 0",
                             "市场收益必须由实际有成交的股票计算。")
            )

    return pd.DataFrame(rows), by_date.sort_index()


def _daily_spearman(left: pd.Series, right: pd.Series, min_obs: int = 50) -> pd.Series:
    """按日期计算 Spearman 相关，观测数不足时返回 NaN。"""
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()

    def one_day(frame: pd.DataFrame) -> float:
        if len(frame) < min_obs or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
            return np.nan
        return frame["left"].corr(frame["right"], method="spearman")

    return joined.groupby(level=DATE_LEVEL, group_keys=False).apply(one_day).rename("spearman")


def audit_factor_mechanism(
    factor_panel: pd.DataFrame,
    components: pd.DataFrame | None,
    factor_name: str = FACTOR_NAME,
) -> pd.DataFrame:
    """检验 ABNRETD 是否体现“极端日收益/高波动”的定义。

    高 ABNRETD 通常应与滚动 MAXRET、收益波动率正相关。这只是公式及经济解释的
    合理性检查，不是预测收益的证据。
    """
    if components is None or components.empty:
        return pd.DataFrame(
            [{
                "relation": "未提供组件数据",
                "mean_daily_spearman": np.nan,
                "median_daily_spearman": np.nan,
                "positive_day_share": np.nan,
                "n_days": 0,
                "interpretation": "计算因子时设置 return_details=True 才能生成该检验。",
            }]
        )

    factor = _as_factor_frame(factor_panel, factor_name)[factor_name]
    rows: list[dict[str, object]] = []
    for column, text in [
        ("MAX_ABS_RETURN", "与未市场调整的最大绝对日收益"),
        ("RETURN_VOLATILITY", "与滚动日收益波动率"),
    ]:
        if column not in components:
            continue
        corr = _daily_spearman(factor, components[column], min_obs=50).dropna()
        rows.append(
            {
                "relation": text,
                "mean_daily_spearman": corr.mean(),
                "median_daily_spearman": corr.median(),
                "positive_day_share": (corr > 0).mean(),
                "n_days": len(corr),
                "interpretation": "应大体为正；若显著为负，请核对收益率、复权和市场收益的对齐方式。",
            }
        )
    return pd.DataFrame(rows)


def _newey_west_mean_t(values: pd.Series, max_lag: int) -> float:
    """均值的 Newey-West t 值；对重叠持有期的日序列使用 HAC 标准误。"""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    lag = min(max(max_lag, 0), n - 1)
    residual = x - x.mean()
    gamma0 = np.dot(residual, residual) / n
    long_run_variance = gamma0
    for j in range(1, lag + 1):
        gamma = np.dot(residual[j:], residual[:-j]) / n
        long_run_variance += 2 * (1 - j / (lag + 1)) * gamma
    standard_error = np.sqrt(long_run_variance / n)
    return np.nan if standard_error == 0 else float(x.mean() / standard_error)


def _assign_quantiles(values: pd.Series, quantiles: int, min_stocks: int) -> pd.Series:
    """在单个截面按因子从低到高分组；样本不足时整日不分组。"""
    groups = pd.Series(np.nan, index=values.index, dtype="float64")
    usable = values.dropna()
    if len(usable) < min_stocks or usable.nunique() < 2:
        return groups
    # method='first' 只用于处理完全相同的取值，保证每组尽量等人数。
    rank_pct = usable.rank(method="first", pct=True)
    groups.loc[usable.index] = np.ceil(rank_pct * quantiles).clip(1, quantiles)
    return groups


def _cross_sectional_standardized_slope(frame: pd.DataFrame, min_obs: int) -> float:
    """单日截面回归：前瞻收益对截面 z-score 因子的斜率。"""
    frame = frame.dropna()
    if len(frame) < min_obs or frame[FACTOR_NAME].nunique() < 2:
        return np.nan
    x = frame[FACTOR_NAME]
    z = (x - x.mean()) / x.std(ddof=0)
    return float(np.dot(z, frame["forward_return"] - frame["forward_return"].mean()) / np.dot(z, z))


def evaluate_factor_effectiveness(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    holding_period: int,
    quantiles: int = 5,
    min_stocks_per_date: int = 50,
    factor_name: str = FACTOR_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算 Rank IC、截面回归斜率、分组收益和稳定性摘要。

    返回依次为：每日 IC 表、IC 摘要、每日分组收益、分组收益摘要。高组为 QN，
    低组为 Q1；同时报告 ``Q1_MINUS_QN``，与“高注意力溢价后反转、做多低因子”
    的假设方向一致。
    """
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2。")
    factor = _as_factor_frame(factor_panel, factor_name)
    return_name = forward_returns.columns[0] if len(forward_returns.columns) == 1 else f"FWD_{holding_period}D"
    forward = _as_factor_frame(forward_returns, return_name).rename(columns={return_name: "forward_return"})
    data = factor.join(forward, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("因子与前瞻收益没有可匹配的有效观测。")

    rank_ic = _daily_spearman(data[factor_name], data["forward_return"], min_obs=min_stocks_per_date)
    slopes = data.groupby(level=DATE_LEVEL, group_keys=False).apply(
        _cross_sectional_standardized_slope, min_obs=min_stocks_per_date
    ).rename("fm_slope")
    n_obs = data.groupby(level=DATE_LEVEL).size().rename("n_obs")
    ic_by_date = pd.concat([rank_ic.rename("rank_ic"), slopes, n_obs], axis=1)
    ic_by_date.index.name = DATE_LEVEL

    def ic_stats(sample: pd.DataFrame, label: str) -> dict[str, object]:
        ic = sample["rank_ic"].dropna()
        slope = sample["fm_slope"].dropna()
        return {
            "sample": label,
            "n_dates": len(ic),
            "rank_ic_mean": ic.mean(),
            "rank_ic_median": ic.median(),
            "rank_ic_std": ic.std(ddof=1),
            "rank_ic_ir": ic.mean() / ic.std(ddof=1) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan,
            "rank_ic_positive_share": (ic > 0).mean(),
            "rank_ic_newey_west_t": _newey_west_mean_t(ic, max_lag=holding_period - 1),
            "fm_slope_mean": slope.mean(),
            "fm_slope_newey_west_t": _newey_west_mean_t(slope, max_lag=holding_period - 1),
        }

    midpoint = ic_by_date.index.sort_values()[len(ic_by_date) // 2]
    ic_summary = pd.DataFrame(
        [
            ic_stats(ic_by_date, "全样本"),
            ic_stats(ic_by_date.loc[ic_by_date.index <= midpoint], "前半样本"),
            ic_stats(ic_by_date.loc[ic_by_date.index > midpoint], "后半样本"),
        ]
    )

    data = data.copy()
    data["quantile"] = data.groupby(level=DATE_LEVEL, group_keys=False)[factor_name].apply(
        _assign_quantiles, quantiles=quantiles, min_stocks=min_stocks_per_date
    )
    grouped = data.dropna(subset=["quantile"]).groupby([pd.Grouper(level=DATE_LEVEL), "quantile"])[
        "forward_return"
    ].mean().unstack("quantile")
    grouped = grouped.rename(columns={q: f"Q{int(q)}" for q in grouped.columns})
    for q in range(1, quantiles + 1):
        if f"Q{q}" not in grouped:
            grouped[f"Q{q}"] = np.nan
    grouped = grouped[[f"Q{q}" for q in range(1, quantiles + 1)]]
    grouped["QN_MINUS_Q1"] = grouped[f"Q{quantiles}"] - grouped["Q1"]
    grouped["Q1_MINUS_QN"] = -grouped["QN_MINUS_Q1"]
    grouped.index.name = DATE_LEVEL

    group_rows: list[dict[str, object]] = []
    for name in list(grouped.columns):
        series = grouped[name].dropna()
        group_rows.append(
            {
                "portfolio": name,
                "mean_forward_return": series.mean(),
                "std_forward_return": series.std(ddof=1),
                "positive_day_share": (series > 0).mean(),
                "newey_west_t": _newey_west_mean_t(series, max_lag=holding_period - 1),
                "n_dates": len(series),
            }
        )
    quantile_summary = pd.DataFrame(group_rows)
    return ic_by_date, ic_summary, grouped, quantile_summary


def run_formula_tests() -> None:
    """用可手算样本检验公式和无未来函数属性；失败会抛出 AssertionError。"""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    market = pd.Series([0.01, -0.02, 0.00, 0.03, -0.01], index=dates)
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.04, 0.02, -0.02],
            "B": [0.01, -0.02, 0.00, 0.03, -0.01],
        },
        index=dates,
    )
    actual = abnretd_from_returns(returns, market, window=3)
    expected_a = pd.Series([np.nan, np.nan, 0.04, 0.04, 0.04], index=dates)
    expected_b = pd.Series([np.nan, np.nan, 0.0, 0.0, 0.0], index=dates)
    np.testing.assert_allclose(actual["A"], expected_a, equal_nan=True)
    np.testing.assert_allclose(actual["B"], expected_b, equal_nan=True)

    # 修改 T+1 之后的数据，T 日已形成的因子不应改变。
    changed = returns.copy()
    changed.loc[dates[-1], "A"] = 0.90
    unchanged_before_last = abnretd_from_returns(changed, market, window=3).loc[dates[:-1], "A"]
    np.testing.assert_allclose(unchanged_before_last, actual.loc[dates[:-1], "A"], equal_nan=True)


def save_validation_report(result: ABNRETDValidation, output_dir: str | Path) -> Path:
    """将所有研究表保存为 UTF-8 CSV 与运行说明 JSON。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "factor.csv": result.factor.reset_index(),
        "forward_returns.csv": result.forward_returns.reset_index(),
        "data_quality_summary.csv": result.data_quality_summary,
        "data_quality_by_date.csv": result.data_quality_by_date.reset_index(),
        "mechanism_summary.csv": result.mechanism_summary,
        "rank_ic_by_date.csv": result.ic_by_date.reset_index(),
        "rank_ic_summary.csv": result.ic_summary,
        "quantile_returns_by_date.csv": result.quantile_returns.reset_index(),
        "quantile_summary.csv": result.quantile_summary,
    }
    for filename, table in tables.items():
        table.to_csv(directory / filename, index=False, encoding="utf-8-sig")
    metadata = {
        "factor": FACTOR_NAME,
        "forward_return_definition": "close[T+H] / open[T+1] - 1",
        "interpretation": "QN_MINUS_Q1 为高 ABNRETD 组合减低 ABNRETD 组合；Q1_MINUS_QN 为反转假设对应的低因子减高因子组合。",
        "warning": "分组收益仍未扣除交易成本、冲击成本、涨跌停和无法成交的约束；需在回测中复核。",
    }
    (directory / "README.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory


def run_abnretd_validation(
    start_date: str,
    end_date: str,
    window: int = 20,
    holding_period: int = 20,
    candidate_ids: Iterable[str] | None = None,
    market_ids: Iterable[str] | None = None,
    batch_size: int = 500,
    min_market_stocks: int = 500,
    quantiles: int = 5,
    min_stocks_per_date: int = 50,
    rebalance_frequency: str = "M",
    expected_universe: pd.DataFrame | pd.Series | None = None,
    output_dir: str | Path | None = "data/abnretd_validation",
) -> ABNRETDValidation:
    """运行一次完整的 ABNRETD 数据、逻辑与有效性检验。

    示例
    ----
    result = run_abnretd_validation(
        start_date="2018-01-01", end_date="2025-12-31",
        candidate_ids=hs300_ids, market_ids=all_a_ids,
        min_market_stocks=500, holding_period=20, rebalance_frequency="M",
    )
    print(result.ic_summary.round(4))
    print(result.quantile_summary.round(4))

    ``candidate_ids`` 只是待下载的代码上限。若研究的是随时间变化的指数成分股，
    可先用 ``get_historical_index_universe`` 取得历史成员，再将其同时传给
    ``expected_universe``（完整流程会自动过滤因子）及其股票代码并集传给
    ``candidate_ids``；不要传入当前成分股后直接回测历史。

    ``rebalance_frequency`` 默认 ``"M"``，在月末用当月最后一个已形成的信号做
    检验；传入 ``"D"`` 可检验每日滚动 20 日版本，但应额外评估其交易成本和
    信号重叠带来的影响。
    """
    run_formula_tests()
    calculation = calculate_abnretd_daily(
        start_date=start_date,
        end_date=end_date,
        window=window,
        candidate_ids=candidate_ids,
        market_ids=market_ids,
        batch_size=batch_size,
        min_market_stocks=min_market_stocks,
        return_details=True,
    )
    assert isinstance(calculation, ABNRETDCalculation)
    if expected_universe is not None:
        calculation.factor = filter_factor_to_universe(calculation.factor, expected_universe)
        if calculation.components is not None:
            calculation.components = filter_factor_to_universe(
                calculation.components, expected_universe
            )
    calculation.factor = select_rebalance_dates(
        calculation.factor, frequency=rebalance_frequency
    )
    if calculation.components is not None:
        calculation.components = select_rebalance_dates(
            calculation.components, frequency=rebalance_frequency
        )
    forward_returns = calculate_forward_returns(
        start_date=start_date,
        end_date=end_date,
        stock_ids=calculation.factor.index.get_level_values(ID_LEVEL).unique(),
        holding_period=holding_period,
        batch_size=batch_size,
    )
    quality_summary, quality_by_date = audit_factor_data(
        calculation.factor,
        market_diagnostics=calculation.market_diagnostics,
        expected_universe=expected_universe,
    )
    mechanism_summary = audit_factor_mechanism(calculation.factor, calculation.components)
    ic_by_date, ic_summary, quantile_returns, quantile_summary = evaluate_factor_effectiveness(
        calculation.factor,
        forward_returns,
        holding_period=holding_period,
        quantiles=quantiles,
        min_stocks_per_date=min_stocks_per_date,
    )
    result = ABNRETDValidation(
        factor=calculation.factor,
        forward_returns=forward_returns,
        data_quality_summary=quality_summary,
        data_quality_by_date=quality_by_date,
        mechanism_summary=mechanism_summary,
        ic_by_date=ic_by_date,
        ic_summary=ic_summary,
        quantile_returns=quantile_returns,
        quantile_summary=quantile_summary,
    )
    if output_dir is not None:
        save_validation_report(result, output_dir)
    return result


if __name__ == "__main__":
    # 不连接数据源也能先确认公式单测通过。
    run_formula_tests()
    print("ABNRETD 公式与无未来函数单测通过。")
    print("请在已初始化 rqdatac 的环境中调用 run_abnretd_validation() 运行全量研究。")
