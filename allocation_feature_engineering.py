"""
Allocation Multiple Model feature engineering.

Single feature module used by the Jupyter trainer and Streamlit app.
Includes the approved sheet columns plus State, Site, and Line Name.

Core responsibilities:
- Normalize incoming workbook columns to canonical names.
- Detect allocation working-table headers.
- Create numeric, categorical-hash, demand/supply, ranking, context, AK, and Site 802 features.
- Build a deterministic feature matrix from train or inference data.

This module intentionally depends only on numpy and pandas.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

APPROVED_COLUMNS = [
    "Class Name", "Line Name", "Site", "State", "MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM",
    "Supply", "Dc Avail", "Rank", "Proj. Demand", "Alloc. Rec.", "Flag",
]
TARGET_COLUMN = "Final Alloc."
NUMERIC_COLUMNS = ["MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM", "Supply", "Dc Avail", "Rank", "Proj. Demand", "Alloc. Rec."]
TEXT_COLUMNS = ["Class Name", "Line Name", "Site", "State", "Flag"]
AK_SITES = {"248", "159", "212", "145", "121"}

COLUMN_ALIASES = {
    "class": "Class Name", "classname": "Class Name", "class name": "Class Name",
    "line": "Line Name", "linename": "Line Name", "line name": "Line Name",
    "site": "Site", "store": "Site", "store number": "Site", "site id": "Site",
    "state": "State", "st": "State",
    "mil": "MIL", "min inv level": "MIL", "minimum inventory level": "MIL",
    "flm": "FLM", "full load multiple": "FLM", "full-load multiple": "FLM",
    "cost": "Cost", "unit cost": "Cost",
    "l30": "L30", "last 30": "L30", "last 30 days": "L30",
    "d30": "D30", "demand 30": "D30", "30 day demand": "D30",
    "d60": "D60", "demand 60": "D60", "60 day demand": "D60",
    "lw": "LW", "last week": "LW",
    "ttm": "TTM", "trailing 12": "TTM", "trailing twelve": "TTM",
    "supply": "Supply", "qoh supply": "Supply", "supply on hand": "Supply",
    "dc avail": "Dc Avail", "dc available": "Dc Avail", "dc availability": "Dc Avail", "left dc": "Dc Avail", "left in dc": "Dc Avail",
    "rank": "Rank",
    "proj demand": "Proj. Demand", "proj. demand": "Proj. Demand", "projected demand": "Proj. Demand",
    "alloc rec": "Alloc. Rec.", "alloc. rec": "Alloc. Rec.", "allocation rec": "Alloc. Rec.", "allocation recommendation": "Alloc. Rec.",
    "flag": "Flag", "review flag": "Flag",
    "final alloc": "Final Alloc.", "final alloc.": "Final Alloc.", "final allocate": "Final Alloc.", "final allocation": "Final Alloc.",
}

@dataclass
class FeatureConfig:
    version: str = "allocation_multiple_model_v1_state_ranker"
    numeric_feature_names: Optional[List[str]] = None
    extra_feature_names: Optional[List[str]] = None
    final_feature_names: Optional[List[str]] = None
    numeric_mean: Optional[List[float]] = None
    numeric_std: Optional[List[float]] = None
    extra_mean: Optional[List[float]] = None
    extra_std: Optional[List[float]] = None
    hash_dims: Optional[Dict[str, int]] = None
    approved_columns: Optional[List[str]] = None
    target_column: str = TARGET_COLUMN
    ak_sites: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("approved_columns") is None:
            d["approved_columns"] = APPROVED_COLUMNS
        if d.get("ak_sites") is None:
            d["ak_sites"] = sorted(AK_SITES)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureConfig":
        return cls(**d)


def _clean_name(x: object) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _alias_key(x: object) -> str:
    s = _clean_name(x).lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9. ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique column names.

    Some allocation workbooks can contain duplicate headers after Excel header
    detection or alias normalization.  In pandas, ``df["Rank"]`` returns a
    DataFrame when duplicate ``Rank`` columns exist, which then breaks
    ``pd.to_numeric`` and the Streamlit prediction path.

    For model columns, duplicate headers are collapsed row-by-row by taking the
    first nonblank value from left to right.  Non-model duplicate columns keep
    their first occurrence because the model never consumes them directly.
    """
    if df.columns.is_unique:
        return df

    canonical_cols = set(APPROVED_COLUMNS + [TARGET_COLUMN])
    result = {}
    used = set()
    for col in list(df.columns):
        if col in used:
            continue
        used.add(col)
        block = df.loc[:, df.columns == col]
        if not isinstance(block, pd.DataFrame) or block.shape[1] == 1:
            result[col] = block.iloc[:, 0] if isinstance(block, pd.DataFrame) else block
            continue

        if col in canonical_cols:
            tmp = block.copy()
            # Treat blank strings as missing so a later duplicate can supply
            # the value. This preserves the left-most valid source column.
            tmp = tmp.replace(r"^\s*$", np.nan, regex=True)
            result[col] = tmp.bfill(axis=1).iloc[:, 0]
        else:
            result[col] = block.iloc[:, 0]

    return pd.DataFrame(result, index=df.index)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common workbook headers to canonical approved names and dedupe them."""
    out = df.copy()
    ren = {}
    seen = set()
    for col in out.columns:
        key = _alias_key(col)
        canon = COLUMN_ALIASES.get(key)
        if canon and canon not in seen:
            ren[col] = canon
            seen.add(canon)
    out = out.rename(columns=ren)
    out = _coalesce_duplicate_columns(out)
    return out


def detect_header_and_table(raw: pd.DataFrame, min_hits: int = 7, scan_rows: int = 80) -> pd.DataFrame:
    """Detect a working-table header row in messy Excel uploads."""
    raw = raw.copy()
    best_i, best_hits = 0, -1
    max_scan = min(scan_rows, len(raw))
    canonical_set = set(APPROVED_COLUMNS + [TARGET_COLUMN])
    for i in range(max_scan):
        vals = [_clean_name(v) for v in raw.iloc[i].tolist()]
        hits = 0
        for v in vals:
            canon = COLUMN_ALIASES.get(_alias_key(v), v)
            if canon in canonical_set:
                hits += 1
        if hits > best_hits:
            best_i, best_hits = i, hits
    if best_hits >= min_hits:
        header = [_clean_name(v) or f"Unnamed_{j}" for j, v in enumerate(raw.iloc[best_i].tolist())]
        out = raw.iloc[best_i + 1:].copy()
        out.columns = header
        out = out.reset_index(drop=True)
    else:
        out = raw.copy()
    out = normalize_columns(out)
    out = out.dropna(how="all").reset_index(drop=True)
    return out


def ensure_columns(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    out = normalize_columns(df.copy())
    for c in APPROVED_COLUMNS:
        if c not in out.columns:
            out[c] = "" if c in TEXT_COLUMNS else 0.0
    if include_target and TARGET_COLUMN not in out.columns:
        out[TARGET_COLUMN] = 0.0
    return out


def _column_as_series(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Fetch one logical column even if a duplicate header slipped through."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        tmp = obj.replace(r"^\s*$", np.nan, regex=True)
        return tmp.bfill(axis=1).iloc[:, 0]
    return obj


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = _column_as_series(df, col, 0.0)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def text_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a clean string Series without pandas object downcast warnings.

    Pandas 2.2+ warns when fillna silently downcasts object arrays.  Casting to
    the nullable string dtype before filling keeps behavior explicit and stable.
    Duplicate logical columns are collapsed to the first nonblank value.
    """
    s = _column_as_series(df, col, "")
    s = s.astype("string").fillna("")
    return s.astype(str).map(lambda x: _clean_name(x))


def safe_div(a, b, default=0.0, clip=1e6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.divide(a, b, out=np.full_like(a, default, dtype=np.float64), where=np.abs(b) > 1e-9)
    return np.clip(out, -clip, clip)


def pct_rank(values: pd.Series, group: pd.Series) -> np.ndarray:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    g = group.fillna("").astype(str)
    return s.groupby(g).rank(pct=True, method="average").fillna(0.5).to_numpy(dtype=np.float32)


def group_sum(values: pd.Series, group: pd.Series) -> np.ndarray:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    g = group.fillna("").astype(str)
    return g.map(s.groupby(g).sum()).fillna(0.0).to_numpy(dtype=np.float32)


def group_mean(values: pd.Series, group: pd.Series) -> np.ndarray:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    g = group.fillna("").astype(str)
    return g.map(s.groupby(g).mean()).fillna(0.0).to_numpy(dtype=np.float32)


def group_count(group: pd.Series) -> np.ndarray:
    g = group.fillna("").astype(str)
    return g.map(g.value_counts()).fillna(1.0).to_numpy(dtype=np.float32)


def raw_dc_bucket(dc: np.ndarray) -> List[str]:
    labels = []
    for x in dc:
        if x <= 0: labels.append("DC_RAW_EMPTY")
        elif x <= 25: labels.append("DC_RAW_1_25")
        elif x <= 50: labels.append("DC_RAW_26_50")
        elif x <= 100: labels.append("DC_RAW_51_100")
        elif x <= 300: labels.append("DC_RAW_101_300")
        elif x <= 600: labels.append("DC_RAW_301_600")
        elif x <= 1000: labels.append("DC_RAW_601_1000")
        elif x <= 2000: labels.append("DC_RAW_1001_2000")
        else: labels.append("DC_RAW_2000_PLUS")
    return labels


def flm_dc_bucket(dc: np.ndarray, flm: np.ndarray) -> List[str]:
    r = safe_div(dc, np.maximum(flm, 1.0), clip=1e4)
    labels = []
    for x in r:
        if x <= 0: labels.append("DC_EMPTY")
        elif x < 1: labels.append("DC_LT_1_FLM")
        elif x < 2: labels.append("DC_1_FLM")
        elif x < 4: labels.append("DC_2_3_FLM")
        elif x < 7: labels.append("DC_4_6_FLM")
        elif x < 13: labels.append("DC_7_12_FLM")
        elif x < 25: labels.append("DC_13_24_FLM")
        else: labels.append("DC_25_PLUS_FLM")
    return labels


def hash_text_to_matrix(values: Sequence[str], dim: int, prefix: str) -> Tuple[np.ndarray, List[str]]:
    mat = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        token = str(v or "").strip().upper()
        if not token:
            token = "__MISSING__"
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        j = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        mat[i, j] = sign
    names = [f"hash__{prefix}__{i}" for i in range(dim)]
    return mat, names


def build_extra_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build v3.9+ state-aware, ranking-aware engineered features."""
    df = ensure_columns(df)
    n = len(df)

    cls = text_series(df, "Class Name")
    line = text_series(df, "Line Name")
    site = text_series(df, "Site")
    state = text_series(df, "State").str.upper()
    flag = text_series(df, "Flag").str.upper()
    class_line = cls.str.upper() + "||" + line.str.upper()

    mil = numeric_series(df, "MIL").to_numpy(float)
    flm = np.maximum(numeric_series(df, "FLM").to_numpy(float), 1.0)
    cost = numeric_series(df, "Cost").to_numpy(float)
    l30 = numeric_series(df, "L30").to_numpy(float)
    d30 = numeric_series(df, "D30").to_numpy(float)
    d60 = numeric_series(df, "D60").to_numpy(float)
    lw = numeric_series(df, "LW").to_numpy(float)
    ttm = numeric_series(df, "TTM").to_numpy(float)
    supply = numeric_series(df, "Supply").to_numpy(float)
    dc = numeric_series(df, "Dc Avail").to_numpy(float)
    rank_raw = numeric_series(df, "Rank").to_numpy(float)
    proj = numeric_series(df, "Proj. Demand").to_numpy(float)
    rec = numeric_series(df, "Alloc. Rec.").to_numpy(float)

    d60_month = d60 / 2.0
    lw_month = lw * 4.29
    ttm_month = ttm / 12.0
    demand_mat = np.vstack([l30, d30, d60_month, lw_month, ttm_month, proj]).T
    weighted_velocity = (0.22*l30 + 0.20*d30 + 0.16*d60_month + 0.18*lw_month + 0.10*ttm_month + 0.14*proj)
    mean_demand = demand_mat.mean(axis=1)
    max_demand = demand_mat.max(axis=1)
    sheet_need = np.maximum.reduce([mil, proj, weighted_velocity, rec, max_demand])

    shortage_units = np.maximum(0, weighted_velocity - supply)
    shortage_sheet_units = np.maximum(0, sheet_need - supply)
    shortage_flms = safe_div(shortage_units, flm, clip=1e4)
    shortage_sheet_flms = safe_div(shortage_sheet_units, flm, clip=1e4)
    rec_flms = safe_div(rec, flm, clip=1e4)
    proj_flms = safe_div(proj, flm, clip=1e4)
    dc_flms = safe_div(dc, flm, clip=1e4)

    rank_filled = np.nan_to_num(rank_raw, nan=9999.0, posinf=9999.0, neginf=9999.0)
    rank_score = 1.0 / (1.0 + np.maximum(rank_filled, 0.0))
    rank_inverse = np.divide(1.0, rank_filled, out=np.zeros_like(rank_filled, dtype=float), where=rank_filled > 0)
    is_rank_blank = text_series(df, "Rank").replace("", np.nan).isna().to_numpy().astype(float)
    is_site802_by_rank_blank = is_rank_blank
    is_site802_by_site = site.astype(str).str.strip().eq("802").to_numpy().astype(float)
    is_ak = ((state == "AK") | site.astype(str).str.strip().isin(AK_SITES)).to_numpy().astype(float)
    is_allocate = flag.str.contains("ALLOC", na=False).to_numpy().astype(float)
    is_review = flag.str.contains("REVIEW", na=False).to_numpy().astype(float)

    above_supply = demand_mat > supply[:, None]
    above_supply_plus = demand_mat > (supply + flm)[:, None]
    positive = demand_mat > 0
    demand_signal_share = above_supply.mean(axis=1)
    high_conf_demand_share = above_supply_plus.mean(axis=1)
    demand_positive_count = positive.sum(axis=1)
    demand_zero_count = (demand_mat <= 0).sum(axis=1)

    demand_range = demand_mat.max(axis=1) - demand_mat.min(axis=1)
    demand_std = demand_mat.std(axis=1)
    demand_cv = safe_div(demand_std, np.maximum(mean_demand, 1.0), clip=100)
    recent_vs_long = np.abs((l30 + d30 + lw_month) / 3.0 - ttm_month)
    short_vs_med = np.abs((l30 + lw_month) / 2.0 - (d30 + d60_month) / 2.0)

    proj_gap_units = proj - supply
    proj_gap_flms = safe_div(np.maximum(0, proj_gap_units), flm, clip=1e4)
    rec_minus_proj = rec - proj
    rec_minus_proj_flms = safe_div(rec_minus_proj, flm, clip=1e4)
    proj_minus_rec_flms = safe_div(proj - rec, flm, clip=1e4)
    rec_proj_abs_gap = np.abs(rec - proj)
    rec_proj_gap_flms = safe_div(rec_proj_abs_gap, flm, clip=1e4)
    proj_rec_agreement = 1.0 / (1.0 + np.abs(rec_minus_proj_flms))

    post_rec_supply = supply + rec
    post_rec_to_proj = safe_div(post_rec_supply, np.maximum(proj, 1.0), clip=100)
    post_rec_to_velocity = safe_div(post_rec_supply, np.maximum(weighted_velocity, 1.0), clip=100)
    post_rec_to_sheet_need = safe_div(post_rec_supply, np.maximum(sheet_need, 1.0), clip=100)
    dc_after_rec = dc - rec
    dc_after_rec_flms = safe_div(dc_after_rec, flm, clip=1e4)

    supply_after_1 = supply + flm
    supply_after_2 = supply + 2*flm
    supply_after_3 = supply + 3*flm
    need_after_1 = sheet_need - supply_after_1
    need_after_2 = sheet_need - supply_after_2
    gap_after_1_proj = proj - supply_after_1
    gap_after_2_proj = proj - supply_after_2
    gap_after_3_proj = proj - supply_after_3
    gap_after_1_vel = weighted_velocity - supply_after_1
    gap_after_2_vel = weighted_velocity - supply_after_2
    gap_after_3_vel = weighted_velocity - supply_after_3
    oversupply_after_1 = np.maximum(0, supply_after_1 - sheet_need)
    oversupply_after_2 = np.maximum(0, supply_after_2 - sheet_need)
    oversupply_after_3 = np.maximum(0, supply_after_3 - sheet_need)

    class_line_s = pd.Series(class_line, index=df.index)
    site_s = pd.Series(site.astype(str), index=df.index)
    cls_s = pd.Series(cls.astype(str), index=df.index)

    class_line_total_dc = group_sum(pd.Series(dc), class_line_s)
    class_line_total_rec = group_sum(pd.Series(rec), class_line_s)
    class_line_total_short_flms = group_sum(pd.Series(shortage_sheet_flms), class_line_s)
    class_line_row_count = group_count(class_line_s)
    class_line_avg_supply = group_mean(pd.Series(supply), class_line_s)
    class_line_avg_velocity = group_mean(pd.Series(weighted_velocity), class_line_s)
    class_line_positive_rec_rate = group_mean(pd.Series((rec > 0).astype(float)), class_line_s)
    class_line_need_pct = pct_rank(pd.Series(sheet_need), class_line_s)
    class_line_rank_pct = pct_rank(pd.Series(-rank_filled), class_line_s)
    class_line_vel_pct = pct_rank(pd.Series(weighted_velocity), class_line_s)
    class_line_rec_pct = pct_rank(pd.Series(rec), class_line_s)
    class_line_short_pct = pct_rank(pd.Series(shortage_sheet_flms), class_line_s)
    class_line_dc_pressure = safe_div(class_line_total_dc, np.maximum(class_line_total_short_flms * flm, 1.0), clip=100)

    site_total_shortage = group_sum(pd.Series(shortage_sheet_units), site_s)
    site_total_rec = group_sum(pd.Series(rec), site_s)
    site_avg_velocity = group_mean(pd.Series(weighted_velocity), site_s)
    site_avg_supply_gap = group_mean(pd.Series(sheet_need - supply), site_s)
    site_positive_rec_rate = group_mean(pd.Series((rec > 0).astype(float)), site_s)
    site_need_pct = pct_rank(pd.Series(sheet_need), site_s)
    site_velocity_pct = pct_rank(pd.Series(weighted_velocity), site_s)
    site_rec_pct = pct_rank(pd.Series(rec), site_s)
    site_pressure_pct = pct_rank(pd.Series(shortage_sheet_flms), site_s)

    peer_avg_l30 = group_mean(pd.Series(l30), class_line_s)
    peer_avg_d30 = group_mean(pd.Series(d30), class_line_s)
    peer_avg_d60 = group_mean(pd.Series(d60), class_line_s)
    peer_avg_supply = group_mean(pd.Series(supply), class_line_s)
    peer_avg_proj = group_mean(pd.Series(proj), class_line_s)
    peer_avg_rec = group_mean(pd.Series(rec), class_line_s)

    rec_trust_score = (
        0.30 * proj_rec_agreement +
        0.25 * demand_signal_share +
        0.20 * (weighted_velocity + flm >= rec).astype(float) +
        0.15 * (dc >= rec).astype(float) -
        0.10 * safe_div(oversupply_after_1, np.maximum(sheet_need, 1.0), clip=100)
    )
    conservative_need_score = demand_signal_share + high_conf_demand_share + np.minimum(shortage_sheet_flms, 5) - safe_div(oversupply_after_1, flm, clip=100)
    aggressive_need_score = np.minimum(shortage_flms, 10) + np.minimum(rec_flms, 10) + np.minimum(proj_flms, 10) + demand_signal_share
    review_priority_score = (
        0.25*class_line_need_pct + 0.20*demand_signal_share + 0.20*np.minimum(shortage_sheet_flms/5.0, 1.0) +
        0.15*np.minimum(rank_score*100, 1.0) + 0.10*rec_trust_score + 0.10*(1.0/(1.0+class_line_dc_pressure))
    )
    allocate_cut_candidate = ((rec > proj + flm) & (demand_signal_share < 0.50)).astype(float)
    allocate_rescue_candidate = ((rec > 0) & (demand_signal_share >= 0.67) & (shortage_sheet_flms >= 1)).astype(float)

    feats = {
        # Original duplicates / strongly weighted aliases
        "mil": mil, "flm": flm, "cost": cost, "l30": l30, "d30": d30, "d60_month": d60_month, "lw_month": lw_month, "ttm_month": ttm_month,
        "supply": supply, "dc_avail": dc, "rank_score": rank_score, "rank_inverse": rank_inverse, "proj_units": proj, "rec_units": rec,
        # Flags and routing
        "is_allocate_flag": is_allocate, "is_review_flag": is_review, "is_ak_state_or_site": is_ak, "is_site802_by_rank_blank": is_site802_by_rank_blank, "is_site802_by_site": is_site802_by_site,
        # Demand and supply
        "weighted_velocity": weighted_velocity, "sheet_need": sheet_need, "mean_demand_signal": mean_demand, "max_demand": max_demand,
        "demand_signal_share_above_supply": demand_signal_share, "demand_signal_share_above_supply_plus_1flm": high_conf_demand_share,
        "demand_signal_count_above_supply": above_supply.sum(axis=1), "demand_positive_count": demand_positive_count, "demand_zero_count": demand_zero_count,
        "demand_signal_range": demand_range, "demand_signal_std": demand_std, "demand_signal_cv": demand_cv,
        "recent_vs_long_term_disagreement": recent_vs_long, "short_term_vs_medium_term_disagreement": short_vs_med,
        "trend_l30_vs_d30": l30 - d30, "trend_d30_vs_d60_month": d30 - d60_month, "trend_lw_month_vs_l30": lw_month - l30,
        "short_term_acceleration": (l30 + lw_month)/2.0 - (d30 + d60_month)/2.0,
        "supply_gap_velocity": weighted_velocity - supply, "supply_gap_sheet_need": sheet_need - supply, "supply_gap_proj": proj - supply,
        "shortage_units": shortage_units, "shortage_sheet_units": shortage_sheet_units, "shortage_flms": shortage_flms, "shortage_sheet_flms": shortage_sheet_flms,
        "overstock_units": np.maximum(0, supply - sheet_need),
        "supply_to_weighted_velocity": safe_div(supply, np.maximum(weighted_velocity, 1.0), clip=100),
        "supply_to_sheet_need": safe_div(supply, np.maximum(sheet_need, 1.0), clip=100),
        "supply_to_proj": safe_div(supply, np.maximum(proj, 1.0), clip=100),
        # Recommendation / projection
        "rec_flms": rec_flms, "proj_flms": proj_flms, "dc_flms": dc_flms,
        "proj_gap_units": proj_gap_units, "proj_gap_flms": proj_gap_flms, "rec_minus_proj": rec_minus_proj, "rec_minus_proj_flms": rec_minus_proj_flms,
        "proj_minus_rec_flms": proj_minus_rec_flms, "rec_proj_abs_gap": rec_proj_abs_gap, "rec_proj_gap_flms": rec_proj_gap_flms,
        "proj_rec_agreement": proj_rec_agreement, "post_rec_to_proj": post_rec_to_proj, "post_rec_to_velocity": post_rec_to_velocity,
        "post_rec_to_sheet_need": post_rec_to_sheet_need, "dc_after_rec": dc_after_rec, "dc_after_rec_flms": dc_after_rec_flms,
        "proj_supported_by_rec": (rec + flm >= proj).astype(float), "rec_exceeds_proj_1flm": (rec > proj + flm).astype(float),
        "proj_exceeds_rec_1flm": (proj > rec + flm).astype(float), "has_proj_demand": (proj > 0).astype(float), "has_alloc_rec_signal": (rec > 0).astype(float),
        "rec_trust_score": rec_trust_score, "rec_trust_high_flag": (rec_trust_score >= 0.65).astype(float), "rec_trust_low_flag": (rec_trust_score < 0.35).astype(float),
        "rec_cut_candidate": ((rec > proj + flm) & (rec_trust_score < 0.45)).astype(float),
        "rec_follow_candidate": ((rec > 0) & (rec_trust_score >= 0.55)).astype(float),
        "rec_add_candidate": ((proj > rec + flm) & (demand_signal_share >= 0.5)).astype(float),
        # Pack outcomes
        "supply_after_1flm": supply_after_1, "supply_after_2flm": supply_after_2, "supply_after_3flm": supply_after_3, "supply_after_rec": post_rec_supply,
        "need_after_1flm": need_after_1, "need_after_2flm": need_after_2,
        "gap_after_1flm_to_proj": gap_after_1_proj, "gap_after_2flm_to_proj": gap_after_2_proj, "gap_after_3flm_to_proj": gap_after_3_proj,
        "gap_after_1flm_to_velocity": gap_after_1_vel, "gap_after_2flm_to_velocity": gap_after_2_vel, "gap_after_3flm_to_velocity": gap_after_3_vel,
        "oversupply_after_1flm": oversupply_after_1, "oversupply_after_2flm": oversupply_after_2, "oversupply_after_3flm": oversupply_after_3,
        "single_flm_oversupply_risk": safe_div(oversupply_after_1, np.maximum(sheet_need, 1.0), clip=100),
        # Class-line context
        "class_line_total_dc_avail": class_line_total_dc, "class_line_total_rec": class_line_total_rec,
        "class_line_total_shortage_flms": class_line_total_short_flms, "class_line_row_count": class_line_row_count,
        "class_line_avg_supply": class_line_avg_supply, "class_line_avg_velocity": class_line_avg_velocity,
        "class_line_positive_rec_rate": class_line_positive_rec_rate, "class_line_need_percentile": class_line_need_pct,
        "class_line_rank_percentile": class_line_rank_pct, "class_line_velocity_percentile": class_line_vel_pct, "class_line_rec_percentile": class_line_rec_pct,
        "shortage_flm_rank_in_class_line": class_line_short_pct, "class_line_dc_pressure_score": class_line_dc_pressure,
        "class_line_tight_dc_flag": (class_line_dc_pressure < 0.75).astype(float), "class_line_abundant_dc_flag": (class_line_dc_pressure > 1.5).astype(float),
        "top5_need_in_class_line": (class_line_need_pct >= 0.95).astype(float), "top10_need_in_class_line": (class_line_need_pct >= 0.90).astype(float),
        # Site context
        "site_total_shortage": site_total_shortage, "site_total_alloc_rec": site_total_rec, "site_avg_velocity": site_avg_velocity,
        "site_avg_supply_gap": site_avg_supply_gap, "site_positive_rec_rate": site_positive_rec_rate, "site_need_percentile": site_need_pct,
        "site_velocity_percentile": site_velocity_pct, "site_rec_percentile": site_rec_pct, "site_supply_pressure_percentile": site_pressure_pct,
        # Rank / cost / sparse / scores
        "pressure_x_rank": shortage_sheet_flms * rank_score, "rank_x_weighted_velocity": weighted_velocity * rank_score, "rank_x_proj_gap": proj_gap_flms * rank_score,
        "cost_log1p": np.log1p(np.maximum(cost, 0)), "pressure_x_cost": shortage_sheet_flms * np.log1p(np.maximum(cost, 0)),
        "allocation_cost_if_1flm": cost * flm, "allocation_cost_if_rec": cost * rec, "cost_x_rec_flms": cost * rec_flms,
        "zero_l30_flag": (l30 <= 0).astype(float), "zero_d30_flag": (d30 <= 0).astype(float), "zero_d60_flag": (d60 <= 0).astype(float),
        "zero_lw_flag": (lw <= 0).astype(float), "zero_ttm_flag": (ttm <= 0).astype(float), "recent_sales_all_zero": ((l30 <= 0) & (d30 <= 0) & (lw <= 0)).astype(float),
        "sparse_demand_count": (demand_mat <= 0).sum(axis=1), "has_any_sales_flag": ((l30 + d30 + d60 + lw + ttm) > 0).astype(float),
        "review_priority_score": review_priority_score, "review_top_5pct_need": (class_line_need_pct >= 0.95).astype(float),
        "review_top_10pct_need": (class_line_need_pct >= 0.90).astype(float), "review_above_group_median_need": (class_line_need_pct >= 0.50).astype(float),
        "conservative_need_score": conservative_need_score, "aggressive_need_score": aggressive_need_score,
        "allocate_cut_candidate": allocate_cut_candidate, "allocate_rescue_candidate": allocate_rescue_candidate,
        "allocate_follow_rec_candidate": ((rec > 0) & (rec_trust_score >= 0.50)).astype(float),
        "allocate_reduce_to_1flm_candidate": ((rec_flms > 1.5) & (shortage_sheet_flms <= 1.2)).astype(float),
        "allocate_zero_out_candidate": ((rec > 0) & (demand_signal_share < 0.25) & (supply >= sheet_need)).astype(float),
        # Peer deltas
        "peer_avg_l30": peer_avg_l30, "peer_avg_d30": peer_avg_d30, "peer_avg_d60": peer_avg_d60, "peer_avg_supply": peer_avg_supply,
        "peer_avg_proj": peer_avg_proj, "peer_avg_rec": peer_avg_rec, "row_l30_vs_peer_avg": l30 - peer_avg_l30,
        "row_proj_vs_peer_avg": proj - peer_avg_proj, "row_rec_vs_peer_avg": rec - peer_avg_rec, "row_supply_gap_vs_peer_avg": (sheet_need - supply) - (sheet_need.mean() - peer_avg_supply),
        # Specialist helpers
        "site802_high_rec_low_velocity": ((is_site802_by_site + is_site802_by_rank_blank > 0) & (rec_flms > 1) & (weighted_velocity < rec)).astype(float),
        "site802_proj_rec_disagreement": ((is_site802_by_site + is_site802_by_rank_blank > 0) & (np.abs(rec_minus_proj_flms) > 1)).astype(float),
        "site802_single_flm_risk": ((is_site802_by_site + is_site802_by_rank_blank > 0) & (safe_div(oversupply_after_1, flm, clip=100) > 1)).astype(float),
        "site802_strong_demand_support": ((is_site802_by_site + is_site802_by_rank_blank > 0) & (demand_signal_share >= 0.67)).astype(float),
        "ak_allocate_candidate": ((is_ak > 0) & (is_allocate > 0)).astype(float),
        "ak_review_main_model_only": ((is_ak > 0) & (is_review > 0)).astype(float),
    }

    # Raw DC bucket one-hot indicators
    raw_labels = raw_dc_bucket(dc)
    for lab in ["DC_RAW_EMPTY", "DC_RAW_1_25", "DC_RAW_26_50", "DC_RAW_51_100", "DC_RAW_101_300", "DC_RAW_301_600", "DC_RAW_601_1000", "DC_RAW_1001_2000", "DC_RAW_2000_PLUS"]:
        feats[lab.lower()] = np.array([1.0 if x == lab else 0.0 for x in raw_labels])

    extra = pd.DataFrame({k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for k, v in feats.items()}, index=df.index)
    groups = {
        "numeric": NUMERIC_COLUMNS,
        "text": TEXT_COLUMNS,
        "extra": list(extra.columns),
        "routing": ["is_allocate_flag", "is_review_flag", "is_ak_state_or_site", "is_site802_by_rank_blank", "is_site802_by_site"],
    }
    return extra, groups


def build_feature_matrix(df: pd.DataFrame, config: Optional[FeatureConfig | Dict] = None, fit: bool = False) -> Tuple[np.ndarray, FeatureConfig, pd.DataFrame]:
    """Return X, feature_config, canonical_df.

    If fit=True, statistics are learned from df. Otherwise they are read from config.
    """
    canon = ensure_columns(df)
    if isinstance(config, dict):
        config = FeatureConfig.from_dict(config)
    if config is None:
        config = FeatureConfig()

    hash_dims = config.hash_dims or {
        "Class Name": 192, "Line Name": 256, "Site": 192, "State": 32,
        "RankText": 16, "Flag": 16, "DcBucket": 16, "RawDcBucket": 12,
    }

    numeric_df = pd.DataFrame({c: numeric_series(canon, c).astype(np.float32) for c in NUMERIC_COLUMNS}, index=canon.index)
    extra_df, _groups = build_extra_features(canon)

    if fit or config.numeric_mean is None:
        numeric_mean = numeric_df.mean(axis=0).to_numpy(dtype=np.float32)
        numeric_std = numeric_df.std(axis=0).replace(0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
        extra_mean = extra_df.mean(axis=0).to_numpy(dtype=np.float32)
        extra_std = extra_df.std(axis=0).replace(0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
        config.numeric_feature_names = list(numeric_df.columns)
        config.extra_feature_names = list(extra_df.columns)
        config.numeric_mean = numeric_mean.tolist()
        config.numeric_std = numeric_std.tolist()
        config.extra_mean = extra_mean.tolist()
        config.extra_std = extra_std.tolist()
        config.hash_dims = hash_dims
        config.approved_columns = APPROVED_COLUMNS
        config.ak_sites = sorted(AK_SITES)
    else:
        # Align to training features if present
        for c in config.numeric_feature_names or []:
            if c not in numeric_df.columns:
                numeric_df[c] = 0.0
        numeric_df = numeric_df[config.numeric_feature_names or list(numeric_df.columns)]
        for c in config.extra_feature_names or []:
            if c not in extra_df.columns:
                extra_df[c] = 0.0
        extra_df = extra_df[config.extra_feature_names or list(extra_df.columns)]
        numeric_mean = np.array(config.numeric_mean, dtype=np.float32)
        numeric_std = np.array(config.numeric_std, dtype=np.float32)
        extra_mean = np.array(config.extra_mean, dtype=np.float32)
        extra_std = np.array(config.extra_std, dtype=np.float32)

    X_parts = []
    names = []
    Xn = (numeric_df.to_numpy(dtype=np.float32) - numeric_mean) / np.maximum(numeric_std, 1e-6)
    Xe = (extra_df.to_numpy(dtype=np.float32) - extra_mean) / np.maximum(extra_std, 1e-6)
    X_parts += [Xn, Xe]
    names += list(numeric_df.columns) + list(extra_df.columns)

    dc = numeric_series(canon, "Dc Avail").to_numpy(float)
    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    cat_sources = {
        "Class Name": text_series(canon, "Class Name"),
        "Line Name": text_series(canon, "Line Name"),
        "Site": text_series(canon, "Site"),
        "State": text_series(canon, "State"),
        "RankText": text_series(canon, "Rank"),
        "Flag": text_series(canon, "Flag"),
        "DcBucket": pd.Series(flm_dc_bucket(dc, flm), index=canon.index),
        "RawDcBucket": pd.Series(raw_dc_bucket(dc), index=canon.index),
    }
    for field, vals in cat_sources.items():
        dim = int(hash_dims.get(field, 16))
        mat, hn = hash_text_to_matrix(vals.astype(str).tolist(), dim, field)
        X_parts.append(mat)
        names += hn

    X = np.concatenate(X_parts, axis=1).astype(np.float32)
    if fit or config.final_feature_names is None:
        config.final_feature_names = names
    return X, config, canon


def _no_alloc_flag(flag: pd.Series) -> pd.Series:
    """Rows marked Z / No Alloc should never be treated as Allocate rows."""
    return flag.str.contains(r"\bNO\s*ALLOC\b|Z\s*[-–]?\s*NO\s*ALLOC|DO\s*NOT\s*ALLOC", regex=True, na=False)


def eligible_mask(df: pd.DataFrame) -> np.ndarray:
    flag = text_series(df, "Flag").str.upper()
    is_review = flag.str.contains("REVIEW", na=False)
    is_allocate = flag.str.contains("ALLOC", na=False) & ~_no_alloc_flag(flag) & ~is_review
    return (is_allocate | is_review).to_numpy()


def allocate_mask(df: pd.DataFrame) -> np.ndarray:
    flag = text_series(df, "Flag").str.upper()
    is_review = flag.str.contains("REVIEW", na=False)
    return (flag.str.contains("ALLOC", na=False) & ~_no_alloc_flag(flag) & ~is_review).to_numpy()


def review_mask(df: pd.DataFrame) -> np.ndarray:
    return text_series(df, "Flag").str.upper().str.contains("REVIEW", na=False).to_numpy()


def ak_mask(df: pd.DataFrame) -> np.ndarray:
    df = ensure_columns(df)
    state = text_series(df, "State").str.upper()
    site = text_series(df, "Site").str.strip()
    return ((state == "AK") | site.isin(AK_SITES)).to_numpy()


def site802_mask(df: pd.DataFrame) -> np.ndarray:
    df = ensure_columns(df)
    site = text_series(df, "Site").str.strip()
    rank = text_series(df, "Rank").str.strip()
    return ((site == "802") | (rank == "")).to_numpy()


def postprocess_units(raw_units: np.ndarray, df: pd.DataFrame, allow_remainder: bool = True) -> np.ndarray:
    """Round to FLM, cap by DC, keep integer units."""
    raw = np.nan_to_num(np.asarray(raw_units, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    flm = np.maximum(numeric_series(df, "FLM").to_numpy(float), 1.0)
    dc = np.maximum(numeric_series(df, "Dc Avail").to_numpy(float), 0.0)
    out = np.maximum(raw, 0.0)
    rounded = np.round(out / flm) * flm
    if allow_remainder:
        # If prediction is positive but less than one FLM remains, allow the final remainder.
        remainder_mask = (out > 0) & (dc > 0) & (dc < flm)
        rounded[remainder_mask] = dc[remainder_mask]
    out = np.minimum(rounded, dc)
    out = np.maximum(np.round(out), 0).astype(np.int64)
    return out


def save_feature_config(config: FeatureConfig, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def load_feature_config(path: str | Path) -> FeatureConfig:
    return FeatureConfig.from_dict(json.loads(Path(path).read_text()))
