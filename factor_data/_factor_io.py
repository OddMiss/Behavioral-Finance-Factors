"""Shared runtime helpers for standalone factor scripts."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-08-31"
_LOCAL_DATA_DIR = None
_INDUSTRY_FILES = None


def configure_local_industry(data_dir):
    """Point industry-dependent factors to the supplied monthly cache."""
    global _LOCAL_DATA_DIR, _INDUSTRY_FILES
    _LOCAL_DATA_DIR = Path(data_dir)
    _INDUSTRY_FILES = None


def _dated_industry_files():
    global _INDUSTRY_FILES
    if _INDUSTRY_FILES is not None:
        return _INDUSTRY_FILES
    files = {}
    if _LOCAL_DATA_DIR is not None:
        directory = _LOCAL_DATA_DIR / "get_instrument_industry_monthly"
        if directory.exists():
            for path in directory.glob("*.parquet"):
                try:
                    files[pd.Timestamp(path.stem)] = path
                except ValueError:
                    pass
    _INDUSTRY_FILES = files
    return files


def get_local_industry_or_fallback(order_book_ids, date, fallback, **kwargs):
    """Return a RiceQuant-compatible industry frame from the local cache."""
    files = _dated_industry_files()
    dates = pd.DatetimeIndex(sorted(files))
    position = dates.searchsorted(pd.Timestamp(date), side="right") - 1
    if position >= 0:
        raw = pd.read_parquet(files[dates[position]])
        if "order_book_id" not in raw.columns:
            raise ValueError("本地行业数据缺少 order_book_id。")
        # 下游因子有两种读取方式：部分通过 reset_index 读取列，
        # 部分直接将行业代码 Series 的索引视为 order_book_id。
        # 因此设为 order_book_id 索引；不要保留同名列，以免
        # 下游 reset_index() 时产生重复列。
        return (
            raw.loc[
                raw["order_book_id"].isin(list(order_book_ids))
            ]
            .copy()
            .set_index("order_book_id", drop=True)
        )
    return fallback(order_book_ids=order_book_ids, date=date, **kwargs)


def to_wide_factor(raw_factor, factor_field):
    """Convert a panel, a wide DataFrame, or (factor, details) to a wide frame."""
    if isinstance(raw_factor, tuple):
        raw_factor = raw_factor[0]
    if isinstance(raw_factor, pd.Series):
        raw_factor = raw_factor.to_frame(name=factor_field)
    if not isinstance(raw_factor, pd.DataFrame):
        raise TypeError("因子函数必须返回 pandas DataFrame 或 Series。")

    if isinstance(raw_factor.index, pd.MultiIndex):
        if factor_field in raw_factor.columns:
            series = raw_factor[factor_field]
        elif raw_factor.shape[1] == 1:
            series = raw_factor.iloc[:, 0]
        else:
            raise ValueError(
                "无法确定面板因子列 {}。".format(factor_field)
            )
        names = list(series.index.names)
        stock_level = (
            names.index("order_book_id")
            if "order_book_id" in names else 1
        )
        wide = series.unstack(level=stock_level)
    else:
        wide = raw_factor.copy()

    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "datetime"
    return wide.sort_index().sort_index(axis=1)


def save_from_cli(build_factor, output_stem, factor_field):
    parser = argparse.ArgumentParser(
        description="计算并保存原始行为金融因子宽表。"
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent)
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
    )
    args = parser.parse_args()

    raw_factor = build_factor(
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=Path(args.data_dir),
    )
    factor_wide = to_wide_factor(raw_factor, factor_field)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (output_stem + ".csv")
    factor_wide.to_csv(
        output_path, index=True, index_label="datetime",
        encoding="utf-8-sig",
    )
    print(
        "saved={} rows={} columns={}".format(
            output_path, factor_wide.shape[0], factor_wide.shape[1]
        )
    )
