"""米筐云端下载：换手率注意力的 10 个扩展衍生因子。

依赖：本文件须与 ``下载_换手率注意力_原始UMD_三因子.py`` 放在同一目录。
后者提供已经核对过的 A 股股票池、行情下载、行业分类、无风险利率、
FF 三因子以及“月度固定 + 市值加权 UMD”的构造。两份脚本共同使用：

    r_{i,d} - RF_d = alpha_{i,T} + beta_{i,T} * indus_turn_{g(i),d}
                      + rho' * (MKT_RF_d, SMB_d, HML_d, UMD_d) + e_{i,d}

对任意信号日 T，全部统计量仅用 [T-19, T] 的数据；因子在 T 收盘后可得，
后续回测应采用 shift=1。输出均为 date x order_book_id 的 Parquet 宽表。

十个新增因子
------------
1. SIGNED_BETA：有符号 beta，区分正、负的行业换手率敏感性。
2. BETA_TSTAT：beta 的 t 值，强调统计可靠的换手率暴露。
3. ATTN_DELTA_R2：加入行业换手率后相对“仅控制变量”模型的增量 R²。
4. ALPHA_TSTAT：截距 alpha 的 t 值，强调持续的风险调整异常收益。
5. RESID_VOL：回归残差的滚动标准差，衡量未解释的特质波动。
6. RESID_Z：窗口末日残差相对本窗口残差波动的标准化意外收益。
7. RESID_TAIL_RATIO：窗口内最大绝对残差 / 残差波动，衡量尾部注意力事件。
8. ATTENTION_LEVEL：当前行业换手率横截面标准差（原始注意力水平）。
9. ATTENTION_ZSCORE：当前行业换手率相对前 19 日的 z 分数（注意力冲击）。
10. ATTENTION_TREND：行业换手率在 20 日窗口内的标准化线性趋势。

说明：第 8--10 项是行业层面信号，同一行业股票在同日取相同值；这是设计
本意，而非重复赋值错误。后续若进行行业中性化，它们的行业间成分会被移除，
应同时保留“不做行业中性化”的结果作经济解释。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac

# 此模块名包含中文，但 Python 3 可正常导入；要求两文件位于同一目录。
import 下载_换手率注意力_原始UMD_三因子 as base


# ============================ 可修改配置 ============================
START_DATE = "2019-01-02"
END_DATE = "2026-08-31"
OUTPUT_DIR = Path("indus_turn_extended_monthly_value_weighted_umd_factors")

# 20 日与已经下载的 BETA / ALPHA / EPSILON 月度市值加权 UMD 因子完全一致。
REGRESSION_WINDOW = 20


def configure_base_module():
    """将基准模块固定到与既有原始因子完全一致的口径。"""
    base.START_DATE = START_DATE
    base.END_DATE = END_DATE
    base.REGRESSION_WINDOW = REGRESSION_WINDOW
    base.UMD_STYLE = "monthly_value_weighted"


def calculate_extended_factors():
    """构造十个不使用未来数据的扩展衍生因子。"""
    configure_base_module()

    instruments = base.all_instruments("CS")
    ids = base.a_share_ids(instruments)
    target_dates = pd.DatetimeIndex(
        pd.to_datetime(base.get_trading_dates(START_DATE, END_DATE))
    )
    history_start = target_dates[0] - pd.Timedelta(
        days=base.LOOKBACK_CALENDAR_DAYS
    )
    dates = pd.DatetimeIndex(
        pd.to_datetime(base.get_trading_dates(history_start, END_DATE))
    )

    price = base.download_price_wide(ids, dates, ["close", "volume"])
    raw_returns = price["close"].pct_change(fill_method=None).iloc[1:]
    return_dates = raw_returns.index
    turnover = base.download_turnover_wide(ids, dates).reindex(return_dates)
    active = (
        price["volume"].iloc[1:].reindex(return_dates).gt(0)
        & turnover.notna()
    )
    raw_returns = raw_returns.where(active)

    market_cap = base.download_factor_wide(ids, dates, "market_cap_3")
    book_to_market = base.download_factor_wide(
        ids, dates, "book_to_market_ratio_lyr"
    )
    lag_cap = market_cap.shift(1).reindex(return_dates)
    lag_btm = book_to_market.shift(1).reindex(return_dates)
    rf = base.get_daily_rf(return_dates)
    controls = base.build_controls(
        raw_returns,
        lag_cap,
        lag_btm,
        rf,
        base.main_board_mask(ids, instruments),
    )
    industry_labels, _ = base.build_industry_labels(ids, return_dates)

    # 原回归中的 indus_turn：行业内活跃股票换手率的横截面样本标准差。
    turnover_values = turnover.to_numpy(float)
    active_values = active.to_numpy(bool)
    n_dates, n_stocks = raw_returns.shape
    n_industries = max(industry_labels.max() + 1, 0)
    industry_turn = np.full((n_dates, n_industries), np.nan)
    for t in range(n_dates):
        for industry_id in np.unique(
            industry_labels[t][industry_labels[t] >= 0]
        ):
            values = turnover_values[t][
                active_values[t] & (industry_labels[t] == industry_id)
            ]
            if len(values) >= max(base.MIN_INDUSTRY_STOCKS, 2):
                industry_turn[t, industry_id] = np.std(values, ddof=1)

    # 每个数组的行均为 return_dates，列均为 ids。
    shapes = (n_dates, n_stocks)
    signed_beta = np.full(shapes, np.nan)
    beta_tstat = np.full(shapes, np.nan)
    attention_delta_r2 = np.full(shapes, np.nan)
    alpha_tstat = np.full(shapes, np.nan)
    resid_vol = np.full(shapes, np.nan)
    resid_z = np.full(shapes, np.nan)
    resid_tail_ratio = np.full(shapes, np.nan)
    attention_level = np.full(shapes, np.nan)
    attention_zscore = np.full(shapes, np.nan)
    attention_trend = np.full(shapes, np.nan)

    y_all = raw_returns.sub(rf, axis=0).to_numpy(float)
    x_controls = controls[["MKT_RF", "SMB", "HML", "UMD"]].to_numpy(float)
    target_positions = [
        return_dates.get_loc(d) for d in target_dates if d in return_dates
    ]
    k_full = 6  # 截距、行业换手率和 4 个收益控制变量
    k_reduced = 5  # 仅截距和 4 个收益控制变量
    time_index = np.arange(REGRESSION_WINDOW, dtype=float)

    for count, t in enumerate(target_positions, start=1):
        if count % 50 == 0:
            print("回归进度：{}/{}".format(count, len(target_positions)))
        left = t - REGRESSION_WINDOW + 1
        if left < 0 or not np.isfinite(x_controls[left:t + 1]).all():
            continue

        for industry_id in np.unique(
            industry_labels[t][industry_labels[t] >= 0]
        ):
            stocks = np.flatnonzero(industry_labels[t] == industry_id)
            attention = industry_turn[left:t + 1, industry_id]
            if not np.isfinite(attention).all():
                continue

            x_full = np.column_stack(
                [
                    np.ones(REGRESSION_WINDOW),
                    attention,
                    x_controls[left:t + 1],
                ]
            )
            x_reduced = np.column_stack(
                [np.ones(REGRESSION_WINDOW), x_controls[left:t + 1]]
            )
            if (
                np.linalg.matrix_rank(x_full) < k_full
                or np.linalg.matrix_rank(x_reduced) < k_reduced
            ):
                continue

            y = y_all[left:t + 1, :][:, stocks]
            valid = np.isfinite(y).all(axis=0)
            if not valid.any():
                continue

            y_valid = y[:, valid]
            coef_full = np.linalg.lstsq(x_full, y_valid, rcond=None)[0]
            fitted = x_full @ coef_full
            residuals = y_valid - fitted
            sse_full = np.sum(residuals ** 2, axis=0)
            dof_full = REGRESSION_WINDOW - k_full
            if dof_full <= 0:
                raise ValueError("REGRESSION_WINDOW 必须大于 6。")
            mse_full = sse_full / dof_full
            residual_sigma = np.sqrt(mse_full)

            coef_reduced = np.linalg.lstsq(x_reduced, y_valid, rcond=None)[0]
            residuals_reduced = y_valid - x_reduced @ coef_reduced
            sse_reduced = np.sum(residuals_reduced ** 2, axis=0)

            # beta / alpha 的 t 值：同一 X 下，每只股票按自身残差方差估计标准误。
            xtx_inv = np.linalg.inv(x_full.T @ x_full)
            beta_se = np.sqrt(mse_full * xtx_inv[1, 1])
            alpha_se = np.sqrt(mse_full * xtx_inv[0, 0])
            beta_t = np.divide(
                coef_full[1], beta_se,
                out=np.full_like(beta_se, np.nan), where=beta_se > 0
            )
            alpha_t = np.divide(
                coef_full[0], alpha_se,
                out=np.full_like(alpha_se, np.nan), where=alpha_se > 0
            )
            delta_r2 = np.divide(
                sse_reduced - sse_full, sse_reduced,
                out=np.full_like(sse_full, np.nan), where=sse_reduced > 0
            )
            latest_residual = residuals[-1]
            z_residual = np.divide(
                latest_residual, residual_sigma,
                out=np.full_like(residual_sigma, np.nan),
                where=residual_sigma > 0,
            )
            tail_ratio = np.divide(
                np.max(np.abs(residuals), axis=0), residual_sigma,
                out=np.full_like(residual_sigma, np.nan),
                where=residual_sigma > 0,
            )

            # 注意力本身的三个行业层面派生量。
            level = attention[-1]
            prior = attention[:-1]
            prior_std = prior.std(ddof=1)
            z_attention = (
                (level - prior.mean()) / prior_std
                if prior_std > 0 else np.nan
            )
            # 无量纲趋势：每日线性斜率除以窗口内样本标准差。
            attention_std = attention.std(ddof=1)
            trend = (
                np.polyfit(time_index, attention, 1)[0] / attention_std
                if attention_std > 0 else np.nan
            )

            valid_stocks = stocks[valid]
            signed_beta[t, valid_stocks] = coef_full[1]
            beta_tstat[t, valid_stocks] = beta_t
            attention_delta_r2[t, valid_stocks] = delta_r2
            alpha_tstat[t, valid_stocks] = alpha_t
            resid_vol[t, valid_stocks] = residual_sigma
            resid_z[t, valid_stocks] = z_residual
            resid_tail_ratio[t, valid_stocks] = tail_ratio
            attention_level[t, valid_stocks] = level
            attention_zscore[t, valid_stocks] = z_attention
            attention_trend[t, valid_stocks] = trend

    index = return_dates[target_positions]
    return {
        "INDUS_TURN_SIGNED_BETA_MONTHLY_VW_UMD": pd.DataFrame(signed_beta[target_positions], index=index, columns=ids),
        "INDUS_TURN_BETA_TSTAT_MONTHLY_VW_UMD": pd.DataFrame(beta_tstat[target_positions], index=index, columns=ids),
        "INDUS_TURN_ATTN_DELTA_R2_MONTHLY_VW_UMD": pd.DataFrame(attention_delta_r2[target_positions], index=index, columns=ids),
        "INDUS_TURN_ALPHA_TSTAT_MONTHLY_VW_UMD": pd.DataFrame(alpha_tstat[target_positions], index=index, columns=ids),
        "INDUS_TURN_RESID_VOL_MONTHLY_VW_UMD": pd.DataFrame(resid_vol[target_positions], index=index, columns=ids),
        "INDUS_TURN_RESID_Z_MONTHLY_VW_UMD": pd.DataFrame(resid_z[target_positions], index=index, columns=ids),
        "INDUS_TURN_RESID_TAIL_RATIO_MONTHLY_VW_UMD": pd.DataFrame(resid_tail_ratio[target_positions], index=index, columns=ids),
        "INDUS_TURN_ATTENTION_LEVEL_MONTHLY_VW_UMD": pd.DataFrame(attention_level[target_positions], index=index, columns=ids),
        "INDUS_TURN_ATTENTION_ZSCORE_MONTHLY_VW_UMD": pd.DataFrame(attention_zscore[target_positions], index=index, columns=ids),
        "INDUS_TURN_ATTENTION_TREND_MONTHLY_VW_UMD": pd.DataFrame(attention_trend[target_positions], index=index, columns=ids),
    }


if __name__ == "__main__":
    rqdatac.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("口径：月度固定赢家/输家池 + 滞后市值加权 UMD；20 日滚动回归。")
    factors = calculate_extended_factors()
    for name, factor in factors.items():
        factor.index.name = "date"
        factor.columns.name = "order_book_id"
        path = OUTPUT_DIR / (name + ".parquet")
        factor.to_parquet(path)
        print("已保存：{}；形状：{}".format(path, factor.shape))
