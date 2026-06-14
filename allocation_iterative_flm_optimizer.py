"""Neural iterative item-level FLM allocator for Allocation Model 3.

The allocator is a hard-rule DC spending layer plus a trainable marginal FLM
step scorer.  It groups rows by item, repeatedly evaluates each eligible row as
"should this row receive the next FLM now?", awards one FLM to the highest
scoring row, recomputes that row's state, and continues until DC is exhausted or
no row clears the learned threshold.

The neural step scorer is optional at prediction time.  If no scorer is passed,
the allocator falls back to a deterministic scoring formula.  When present, the
step scorer receives all upstream model outputs plus dynamic cycle state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import pandas as pd

from allocation_feature_engineering import (
    TARGET_COLUMN,
    ensure_columns,
    numeric_series,
    text_series,
)

AUX_NAMES = [
    "aux_cut_rec",
    "aux_follow_rec",
    "aux_one_flm",
    "aux_zero_out",
    "aux_below_flm_remainder",
]

STEP_FEATURE_NAMES = [
    "segment_allocate", "segment_review",
    "classifier_probability", "rank_priority", "pred_flms_raw",
    "shared_demand_score", "target_final_supply_prediction",
    "aux_cut_rec", "aux_follow_rec", "aux_one_flm", "aux_zero_out", "aux_below_flm_remainder",
    "step_units", "step_flms", "allocated_units_so_far", "allocated_flms_so_far",
    "dc_remaining_units", "dc_remaining_flms", "row_cap_remaining_units", "row_cap_remaining_flms",
    "flm", "mil", "supply", "current_supply", "final_supply_after_step",
    "alloc_rec", "proj_demand", "d60_month", "weighted_velocity",
    "demand_target", "demand_gap_flm_before", "demand_gap_flm_after",
    "shared_demand_gap_flm_before", "shared_demand_gap_flm_after",
    "target_supply_gap_flm_before", "target_supply_gap_flm_after",
    "proj_gap_flm_before", "proj_gap_flm_after",
    "velocity_gap_flm_before", "velocity_gap_flm_after",
    "rec_remaining_flm_before", "rec_remaining_flm_after",
    "pred_remaining_flm_before", "pred_remaining_flm_after",
    "over_proj_after", "over_d60_after", "zero_demand_positive_step",
    "candidate_score_formula", "classifier_above_threshold", "review_soft_excess_after",
]

@dataclass
class IterativeFLMConfig:
    # Classifier thresholds feeding the cycle. These are not final allocation
    # predictions; they only decide whether a row can compete for a marginal FLM.
    allocate_threshold: float = 0.32
    review_threshold: float = 0.44

    # Neural step scorer thresholds.  The scorer predicts whether a row should
    # receive the next marginal FLM at its current state.
    use_neural_step_scorer: bool = True
    min_allocate_neural_score: float = 0.48
    min_review_neural_score: float = 0.52
    min_partial_neural_score: float = 0.60

    # Deterministic fallback thresholds if the neural step scorer is absent.
    min_allocate_cycle_score: float = 1.05
    min_review_cycle_score: float = 1.22
    min_partial_cycle_score: float = 1.55

    # Business guardrails.
    max_above_rec_flm: float = 1.0
    # Kept for backward-compatible config loading, but row caps now always use
    # Alloc. Rec. + max_above_rec_flm * FLM.  If Alloc. Rec. is 0, the max is 1 FLM.
    rescue_cap_flm_if_rec_zero: float = 1.0
    allow_final_partial: bool = True
    partial_only_once_per_group: bool = True
    max_cycles_per_group: int = 10000
    # Testing/prediction trace guard.  Cycle traces are useful, but writing every
    # accepted FLM cycle can be slow on large books.  The allocation logic still
    # runs fully; this only limits the diagnostic DataFrame size.
    record_cycle_trace: bool = True
    max_cycle_trace_rows: int = 250000

    # Fallback formula weights and soft penalties.
    review_score_penalty: float = 0.18
    review_soft_max_flm: float = 4.0
    diminishing_return_per_flm: float = 0.32
    direct_sizer_weight: float = 0.72
    shared_demand_weight: float = 0.34
    final_supply_gap_weight: float = 0.42
    rank_weight: float = 1.15
    classifier_weight: float = 1.35
    demand_gap_weight: float = 0.78
    rec_gap_weight: float = 0.50
    zero_out_penalty: float = 1.15
    cut_rec_penalty: float = 0.28
    oversupply_proj_penalty: float = 0.22
    oversupply_d60_penalty: float = 0.18
    zero_demand_penalty: float = 1.05

    # Training sample controls for the neural step scorer.
    # The default compact mode trains from the Allocate/Review source rows
    # directly instead of expanding every item into every historical cycle.
    # That keeps the cached step-scorer object proportional to source rows.
    step_training_mode: str = "compact_source_rows"  # compact_source_rows or full_cycle
    compact_positive_states_per_row: int = 3
    compact_add_stop_examples: bool = True
    max_training_cycles_per_group: int = 2000
    max_negative_candidates_per_cycle: int = 18
    include_all_candidates_when_group_leq: int = 22
    random_negative_candidates_per_cycle: int = 4
    step_training_seed: int = 42


def save_optimizer_config(path: str | Path, config: Dict[str, Any] | None = None) -> None:
    cfg = IterativeFLMConfig(**(config or {}))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def load_optimizer_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return asdict(IterativeFLMConfig())
    loaded = json.loads(p.read_text(encoding="utf-8"))
    base = asdict(IterativeFLMConfig())
    base.update(loaded)
    return base


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return numeric_series(df, col).to_numpy(float)


def _txt(df: pd.DataFrame, col: str) -> pd.Series:
    return text_series(df, col).fillna("").astype(str).str.strip()


def item_group_key(df: pd.DataFrame) -> pd.Series:
    """Build the allocation pool key used by the iterative optimizer.

    The optimizer should cycle within one item *inside one allocation run*.
    Training and testing often concatenate many daily workbooks into a single
    DataFrame, so grouping by Item alone incorrectly lets rows from different
    source files compete for the same DC pool.  When source_file/source_sheet are
    present, include them in the key.  For a single uploaded workbook where those
    columns are absent, this naturally falls back to item/product/class-line.
    """
    item = _txt(df, "Item") if "Item" in df.columns else pd.Series([""] * len(df), index=df.index)
    product = _txt(df, "Product ID") if "Product ID" in df.columns else pd.Series([""] * len(df), index=df.index)
    cls_line = _txt(df, "Class Name") + "||" + _txt(df, "Line Name")
    item_part = item.where(item.str.len() > 0, product)
    item_part = item_part.where(item_part.str.len() > 0, cls_line)
    item_part = item_part.where(item_part.astype(str).str.len() > 0, "__unknown_item__").astype(str)

    source_file = _txt(df, "source_file") if "source_file" in df.columns else pd.Series([""] * len(df), index=df.index)
    source_sheet = _txt(df, "source_sheet") if "source_sheet" in df.columns else pd.Series([""] * len(df), index=df.index)
    has_source = source_file.str.len() > 0
    source_part = source_file.where(source_file.str.len() > 0, "__single_file__")
    # source_sheet only matters when multiple sheets from the same workbook were read.
    source_part = source_part + "||" + source_sheet.where(source_sheet.str.len() > 0, "sheet")
    key = item_part.where(~has_source, source_part + "||" + item_part)
    return key.astype(str)


def item_group_quality(df: pd.DataFrame) -> pd.Series:
    item = _txt(df, "Item") if "Item" in df.columns else pd.Series([""] * len(df), index=df.index)
    product = _txt(df, "Product ID") if "Product ID" in df.columns else pd.Series([""] * len(df), index=df.index)
    q = pd.Series("class_line_fallback", index=df.index, dtype=object)
    q.loc[product.str.len() > 0] = "product_id"
    q.loc[item.str.len() > 0] = "item"
    return q


def weighted_velocity(df: pd.DataFrame) -> np.ndarray:
    l30 = _num(df, "L30")
    d30 = _num(df, "D30")
    d60m = _num(df, "D60") / 2.0
    lwm = _num(df, "LW") * 4.29
    ttmm = _num(df, "TTM") / 12.0
    proj = _num(df, "Proj. Demand")
    return 0.22*l30 + 0.20*d30 + 0.16*d60m + 0.18*lwm + 0.10*ttmm + 0.14*proj


def _prepare_signals(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    s = signals.copy().reset_index(drop=True)
    defaults = {
        "model_segment": "none",
        "classifier_probability": 0.0,
        "rank_priority": 0.0,
        "pred_flms_raw": 0.0,
        "shared_demand_score": 0.0,
        "target_final_supply_prediction": 0.0,
    }
    for col, val in defaults.items():
        if col not in s.columns:
            s[col] = val
    # Ignore any legacy review_pass1 columns if present. The new Review path is single-pass.
    for col in ["classifier_probability", "rank_priority", "pred_flms_raw", "shared_demand_score", "target_final_supply_prediction"] + AUX_NAMES:
        if col not in s.columns:
            s[col] = 0.0
        s[col] = pd.to_numeric(s[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(s) != n:
        raise ValueError(f"signals row count {len(s)} does not match data row count {n}")
    return s



def _eligible_model_indices(signals: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Return only rows owned by the Allocate/Review iterative allocator.

    This intentionally excludes Z - No Alloc / Do Not Allocate / other rows even
    when they share the same item group.  Those rows may still be present in a
    full workbook for auditing, but they cannot create step-scorer examples,
    compete for FLMs, or influence the item DC pool used by the cycle allocator.
    """
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return idx
    seg = signals.loc[idx, "model_segment"].astype(str).to_numpy()
    return idx[np.isin(seg, ["Allocate", "Review"])]


def _group_dc_pool(static: Dict[str, np.ndarray], eligible_idxs: np.ndarray) -> float:
    """DC pool for the cycle, based only on Allocate/Review rows in the item."""
    eligible_idxs = np.asarray(eligible_idxs, dtype=int)
    if len(eligible_idxs) == 0:
        return 0.0
    return max(float(np.nanmax(static["dc"][eligible_idxs])), 0.0)


def _apply_row_caps(allocated: np.ndarray, row_cap: np.ndarray, row_reasons: List[str] | None = None) -> np.ndarray:
    """Clamp row allocations to the hard row cap and optionally explain reductions."""
    capped = np.minimum(np.maximum(allocated, 0.0), np.maximum(row_cap, 0.0))
    if row_reasons is not None:
        changed = np.where(capped < allocated - 1e-9)[0]
        for idx in changed:
            row_reasons[int(idx)] = "repaired: capped at Alloc. Rec. + 1 FLM"
    return capped


def _static_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "flm": np.maximum(_num(df, "FLM"), 1.0),
        "supply": _num(df, "Supply"),
        "rec": _num(df, "Alloc. Rec."),
        "proj": _num(df, "Proj. Demand"),
        "d60m": _num(df, "D60") / 2.0,
        "mil": _num(df, "MIL"),
        "vel": weighted_velocity(df),
        "dc": np.maximum(_num(df, "Dc Avail"), 0.0),
    }


def _row_caps(df: pd.DataFrame, signals: pd.DataFrame, cfg: IterativeFLMConfig, static: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    if static is None:
        static = _static_arrays(df)
    supply = static["supply"]
    flm = static["flm"]
    rec = static["rec"]
    mil = static["mil"]
    proj = static["proj"]
    d60m = static["d60m"]
    vel = static["vel"]
    pred_flms = np.maximum(pd.to_numeric(signals["pred_flms_raw"], errors="coerce").fillna(0).to_numpy(float), 0.0)
    # Hard business maximum: Final Alloc. may not exceed Alloc. Rec. + 1 FLM.
    # This is applied for every row, including rows with Alloc. Rec. = 0, where
    # the maximum allowed allocation becomes exactly one FLM.
    rec_cap = np.maximum(0.0, rec) + cfg.max_above_rec_flm * flm
    return np.nan_to_num(np.maximum(rec_cap, 0.0), nan=0.0, posinf=0.0, neginf=0.0)


def _formula_score(idx: int, step_units: float, allocated: np.ndarray, signals: pd.DataFrame, cfg: IterativeFLMConfig, static: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
    seg = str(signals.at[idx, "model_segment"])
    cp = float(signals.at[idx, "classifier_probability"])
    flm = static["flm"][idx]
    current_alloc = allocated[idx]
    current_supply = static["supply"][idx] + current_alloc
    after_supply = current_supply + step_units
    rec = static["rec"][idx]
    proj = static["proj"][idx]
    d60m = static["d60m"][idx]
    mil = static["mil"][idx]
    vel = static["vel"][idx]
    pred_flms = max(float(signals.at[idx, "pred_flms_raw"]), 0.0)
    rank = float(signals.at[idx, "rank_priority"])
    shared_demand = max(float(signals.at[idx, "shared_demand_score"]), 0.0)
    target_supply_pred = max(float(signals.at[idx, "target_final_supply_prediction"]), 0.0)
    demand_target = max(mil, proj, d60m, vel, shared_demand, target_supply_pred, static["supply"][idx] + pred_flms * flm)
    demand_gap_flm = max(0.0, demand_target - current_supply) / flm
    shared_demand_gap_flm = max(0.0, shared_demand - current_supply) / flm
    final_supply_gap_flm = max(0.0, target_supply_pred - current_supply) / flm
    proj_gap_flm = max(0.0, proj - current_supply) / flm
    vel_gap_flm = max(0.0, vel - current_supply) / flm
    pred_remaining_flm = max(0.0, pred_flms - current_alloc / flm)
    rec_remaining_flm = max(0.0, (max(rec, 0.0) + cfg.max_above_rec_flm * flm - current_alloc) / flm)
    allocated_flm = current_alloc / flm
    aux_cut = float(signals.at[idx, "aux_cut_rec"])
    aux_follow = float(signals.at[idx, "aux_follow_rec"])
    aux_one = float(signals.at[idx, "aux_one_flm"])
    aux_zero = float(signals.at[idx, "aux_zero_out"])
    aux_rem = float(signals.at[idx, "aux_below_flm_remainder"])
    over_proj = 1.0 if proj > 0 and after_supply > proj + flm else 0.0
    over_d60 = 1.0 if d60m > 0 and after_supply > d60m + flm else 0.0
    zero_demand = 1.0 if max(mil, proj, d60m, vel) <= 0 and step_units > 0 else 0.0
    review_penalty = cfg.review_score_penalty if seg == "Review" else 0.0
    review_soft_excess = max(0.0, allocated_flm + step_units / flm - cfg.review_soft_max_flm) if seg == "Review" else 0.0
    score = (
        cfg.classifier_weight * cp + cfg.rank_weight * rank + cfg.direct_sizer_weight * pred_remaining_flm
        + cfg.demand_gap_weight * demand_gap_flm + cfg.shared_demand_weight * shared_demand_gap_flm
        + cfg.final_supply_gap_weight * final_supply_gap_flm + 0.42 * proj_gap_flm + 0.36 * vel_gap_flm
        + cfg.rec_gap_weight * rec_remaining_flm + 0.22 * aux_follow + 0.14 * aux_one + 0.18 * aux_rem
        - cfg.zero_out_penalty * aux_zero - cfg.cut_rec_penalty * aux_cut * max(0.0, allocated_flm)
        - cfg.diminishing_return_per_flm * allocated_flm - cfg.oversupply_proj_penalty * over_proj
        - cfg.oversupply_d60_penalty * over_d60 - cfg.zero_demand_penalty * zero_demand
        - review_penalty - 0.45 * review_soft_excess
    )
    return float(score), {
        "formula_score": float(score),
        "demand_gap_flm": demand_gap_flm,
        "shared_demand_gap_flm": shared_demand_gap_flm,
        "final_supply_gap_flm": final_supply_gap_flm,
        "proj_gap_flm": proj_gap_flm,
        "vel_gap_flm": vel_gap_flm,
        "rec_remaining_flm": rec_remaining_flm,
        "pred_remaining_flm": pred_remaining_flm,
        "allocated_flm_before": allocated_flm,
        "over_proj_flag": over_proj,
        "over_d60_flag": over_d60,
    }


def build_step_feature_matrix(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    allocated: np.ndarray,
    remaining_dc: float,
    step_units: np.ndarray,
    indices: np.ndarray,
    config: Dict[str, Any] | None = None,
    static: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build neural marginal-step features for selected row indexes."""
    cfg = IterativeFLMConfig(**(config or {}))
    if static is None:
        static = _static_arrays(df)
        static["row_cap"] = _row_caps(df, signals, cfg, static)
    idx = np.asarray(indices, dtype=int)
    step = np.asarray(step_units, dtype=float).reshape(-1)
    if len(step) == 1 and len(idx) > 1:
        step = np.repeat(step, len(idx))
    flm = np.maximum(static["flm"][idx], 1.0)
    supply = static["supply"][idx]
    allocated_now = allocated[idx]
    current_supply = supply + allocated_now
    after_supply = current_supply + step
    rec = static["rec"][idx]
    proj = static["proj"][idx]
    d60m = static["d60m"][idx]
    mil = static["mil"][idx]
    vel = static["vel"][idx]
    row_cap = static["row_cap"][idx]
    seg = signals.loc[idx, "model_segment"].astype(str).to_numpy()
    cp = signals.loc[idx, "classifier_probability"].to_numpy(float)
    threshold = np.where(seg == "Review", cfg.review_threshold, cfg.allocate_threshold)
    rank = signals.loc[idx, "rank_priority"].to_numpy(float)
    pred_flms = np.maximum(signals.loc[idx, "pred_flms_raw"].to_numpy(float), 0.0)
    shared_demand = np.maximum(signals.loc[idx, "shared_demand_score"].to_numpy(float), 0.0)
    target_supply = np.maximum(signals.loc[idx, "target_final_supply_prediction"].to_numpy(float), 0.0)
    demand_target = np.maximum.reduce([mil, proj, d60m, vel, shared_demand, target_supply, supply + pred_flms * flm])
    rec_limit = np.maximum(rec, 0.0) + cfg.max_above_rec_flm * flm
    rec_remaining_before = np.maximum(0.0, (rec_limit - allocated_now) / flm)
    rec_remaining_after = np.maximum(0.0, (rec_limit - allocated_now - step) / flm)
    pred_remaining_before = np.maximum(0.0, pred_flms - allocated_now / flm)
    pred_remaining_after = np.maximum(0.0, pred_flms - (allocated_now + step) / flm)
    # Formula score is included as one feature because it encodes the original hand-built allocator signal.
    formula_scores = []
    for local_i, row_idx in enumerate(idx):
        fs, _ = _formula_score(int(row_idx), float(step[local_i]), allocated, signals, cfg, static)
        formula_scores.append(fs)
    formula_scores = np.asarray(formula_scores, dtype=float)
    data = {
        "segment_allocate": (seg == "Allocate").astype(float),
        "segment_review": (seg == "Review").astype(float),
        "classifier_probability": cp,
        "rank_priority": rank,
        "pred_flms_raw": pred_flms,
        "shared_demand_score": shared_demand,
        "target_final_supply_prediction": target_supply,
        "aux_cut_rec": signals.loc[idx, "aux_cut_rec"].to_numpy(float),
        "aux_follow_rec": signals.loc[idx, "aux_follow_rec"].to_numpy(float),
        "aux_one_flm": signals.loc[idx, "aux_one_flm"].to_numpy(float),
        "aux_zero_out": signals.loc[idx, "aux_zero_out"].to_numpy(float),
        "aux_below_flm_remainder": signals.loc[idx, "aux_below_flm_remainder"].to_numpy(float),
        "step_units": step,
        "step_flms": step / flm,
        "allocated_units_so_far": allocated_now,
        "allocated_flms_so_far": allocated_now / flm,
        "dc_remaining_units": np.repeat(float(remaining_dc), len(idx)),
        "dc_remaining_flms": np.repeat(float(remaining_dc), len(idx)) / flm,
        "row_cap_remaining_units": np.maximum(0.0, row_cap - allocated_now),
        "row_cap_remaining_flms": np.maximum(0.0, row_cap - allocated_now) / flm,
        "flm": flm,
        "mil": mil,
        "supply": supply,
        "current_supply": current_supply,
        "final_supply_after_step": after_supply,
        "alloc_rec": rec,
        "proj_demand": proj,
        "d60_month": d60m,
        "weighted_velocity": vel,
        "demand_target": demand_target,
        "demand_gap_flm_before": np.maximum(0.0, demand_target - current_supply) / flm,
        "demand_gap_flm_after": np.maximum(0.0, demand_target - after_supply) / flm,
        "shared_demand_gap_flm_before": np.maximum(0.0, shared_demand - current_supply) / flm,
        "shared_demand_gap_flm_after": np.maximum(0.0, shared_demand - after_supply) / flm,
        "target_supply_gap_flm_before": np.maximum(0.0, target_supply - current_supply) / flm,
        "target_supply_gap_flm_after": np.maximum(0.0, target_supply - after_supply) / flm,
        "proj_gap_flm_before": np.maximum(0.0, proj - current_supply) / flm,
        "proj_gap_flm_after": np.maximum(0.0, proj - after_supply) / flm,
        "velocity_gap_flm_before": np.maximum(0.0, vel - current_supply) / flm,
        "velocity_gap_flm_after": np.maximum(0.0, vel - after_supply) / flm,
        "rec_remaining_flm_before": rec_remaining_before,
        "rec_remaining_flm_after": rec_remaining_after,
        "pred_remaining_flm_before": pred_remaining_before,
        "pred_remaining_flm_after": pred_remaining_after,
        "over_proj_after": ((proj > 0) & (after_supply > proj + flm)).astype(float),
        "over_d60_after": ((d60m > 0) & (after_supply > d60m + flm)).astype(float),
        "zero_demand_positive_step": ((np.maximum.reduce([mil, proj, d60m, vel]) <= 0) & (step > 0)).astype(float),
        "candidate_score_formula": formula_scores,
        "classifier_above_threshold": (cp >= threshold).astype(float),
        "review_soft_excess_after": np.where(seg == "Review", np.maximum(0.0, (allocated_now + step) / flm - cfg.review_soft_max_flm), 0.0),
    }
    mat = np.column_stack([np.nan_to_num(data[name], nan=0.0, posinf=0.0, neginf=0.0) for name in STEP_FEATURE_NAMES]).astype(np.float32)
    return mat


def _formula_score_array(
    seg: np.ndarray,
    cp: np.ndarray,
    rank: np.ndarray,
    pred_flms: np.ndarray,
    shared_demand: np.ndarray,
    target_supply: np.ndarray,
    aux_cut: np.ndarray,
    aux_follow: np.ndarray,
    aux_one: np.ndarray,
    aux_zero: np.ndarray,
    aux_rem: np.ndarray,
    flm: np.ndarray,
    supply: np.ndarray,
    allocated_now: np.ndarray,
    step: np.ndarray,
    rec: np.ndarray,
    proj: np.ndarray,
    d60m: np.ndarray,
    mil: np.ndarray,
    vel: np.ndarray,
    cfg: IterativeFLMConfig,
) -> np.ndarray:
    """Vectorized equivalent of the fallback marginal-FLM scoring formula."""
    current_supply = supply + allocated_now
    after_supply = current_supply + step
    demand_target = np.maximum.reduce([mil, proj, d60m, vel, shared_demand, target_supply, supply + pred_flms * flm])
    demand_gap_flm = np.maximum(0.0, demand_target - current_supply) / flm
    shared_demand_gap_flm = np.maximum(0.0, shared_demand - current_supply) / flm
    final_supply_gap_flm = np.maximum(0.0, target_supply - current_supply) / flm
    proj_gap_flm = np.maximum(0.0, proj - current_supply) / flm
    vel_gap_flm = np.maximum(0.0, vel - current_supply) / flm
    rec_limit = np.maximum(rec, 0.0) + cfg.max_above_rec_flm * flm
    rec_remaining_flm = np.maximum(0.0, rec_limit - allocated_now) / flm
    pred_remaining_flm = np.maximum(0.0, pred_flms - allocated_now / flm)
    allocated_flm = allocated_now / flm
    over_proj = ((proj > 0) & (after_supply > proj + flm)).astype(float)
    over_d60 = ((d60m > 0) & (after_supply > d60m + flm)).astype(float)
    zero_demand = ((np.maximum.reduce([mil, proj, d60m, vel]) <= 0) & (step > 0)).astype(float)
    review_penalty = np.where(seg == "Review", cfg.review_score_penalty, 0.0)
    review_soft_excess = np.where(seg == "Review", np.maximum(0.0, allocated_flm + step / flm - cfg.review_soft_max_flm), 0.0)
    return (
        cfg.classifier_weight * cp + cfg.rank_weight * rank + cfg.direct_sizer_weight * pred_remaining_flm
        + cfg.demand_gap_weight * demand_gap_flm + cfg.shared_demand_weight * shared_demand_gap_flm
        + cfg.final_supply_gap_weight * final_supply_gap_flm + 0.42 * proj_gap_flm + 0.36 * vel_gap_flm
        + cfg.rec_gap_weight * rec_remaining_flm + 0.22 * aux_follow + 0.14 * aux_one + 0.18 * aux_rem
        - cfg.zero_out_penalty * aux_zero - cfg.cut_rec_penalty * aux_cut * np.maximum(0.0, allocated_flm)
        - cfg.diminishing_return_per_flm * allocated_flm - cfg.oversupply_proj_penalty * over_proj
        - cfg.oversupply_d60_penalty * over_d60 - cfg.zero_demand_penalty * zero_demand
        - review_penalty - 0.45 * review_soft_excess
    ).astype(float)


def build_step_feature_matrix_examples(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    indices: np.ndarray,
    allocated_units_so_far: np.ndarray,
    remaining_dc_units: np.ndarray,
    step_units: np.ndarray,
    config: Dict[str, Any] | None = None,
    static: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build marginal-step features when each example has its own allocation state.

    `build_step_feature_matrix` is used by the live optimizer where a single
    allocated vector represents the current item cycle.  Training cache creation
    needs a compact source-row representation, so this helper accepts one
    allocated amount and one remaining-DC amount per example.  It prevents the
    training object from exploding into millions of repeated cycle rows.
    """
    cfg = IterativeFLMConfig(**(config or {}))
    if static is None:
        static = _static_arrays(df)
        static["row_cap"] = _row_caps(df, signals, cfg, static)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    allocated_now = np.asarray(allocated_units_so_far, dtype=float).reshape(-1)
    remaining_dc = np.asarray(remaining_dc_units, dtype=float).reshape(-1)
    step = np.asarray(step_units, dtype=float).reshape(-1)
    if len(allocated_now) == 1 and len(idx) > 1:
        allocated_now = np.repeat(allocated_now, len(idx))
    if len(remaining_dc) == 1 and len(idx) > 1:
        remaining_dc = np.repeat(remaining_dc, len(idx))
    if len(step) == 1 and len(idx) > 1:
        step = np.repeat(step, len(idx))

    flm = np.maximum(static["flm"][idx], 1.0)
    supply = static["supply"][idx]
    current_supply = supply + allocated_now
    after_supply = current_supply + step
    rec = static["rec"][idx]
    proj = static["proj"][idx]
    d60m = static["d60m"][idx]
    mil = static["mil"][idx]
    vel = static["vel"][idx]
    row_cap = static["row_cap"][idx]
    seg = signals.loc[idx, "model_segment"].astype(str).to_numpy()
    cp = signals.loc[idx, "classifier_probability"].to_numpy(float)
    threshold = np.where(seg == "Review", cfg.review_threshold, cfg.allocate_threshold)
    rank = signals.loc[idx, "rank_priority"].to_numpy(float)
    pred_flms = np.maximum(signals.loc[idx, "pred_flms_raw"].to_numpy(float), 0.0)
    shared_demand = np.maximum(signals.loc[idx, "shared_demand_score"].to_numpy(float), 0.0)
    target_supply = np.maximum(signals.loc[idx, "target_final_supply_prediction"].to_numpy(float), 0.0)
    aux_cut = signals.loc[idx, "aux_cut_rec"].to_numpy(float)
    aux_follow = signals.loc[idx, "aux_follow_rec"].to_numpy(float)
    aux_one = signals.loc[idx, "aux_one_flm"].to_numpy(float)
    aux_zero = signals.loc[idx, "aux_zero_out"].to_numpy(float)
    aux_rem = signals.loc[idx, "aux_below_flm_remainder"].to_numpy(float)

    demand_target = np.maximum.reduce([mil, proj, d60m, vel, shared_demand, target_supply, supply + pred_flms * flm])
    rec_limit = np.maximum(rec, 0.0) + cfg.max_above_rec_flm * flm
    rec_remaining_before = np.maximum(0.0, (rec_limit - allocated_now) / flm)
    rec_remaining_after = np.maximum(0.0, (rec_limit - allocated_now - step) / flm)
    pred_remaining_before = np.maximum(0.0, pred_flms - allocated_now / flm)
    pred_remaining_after = np.maximum(0.0, pred_flms - (allocated_now + step) / flm)
    formula_scores = _formula_score_array(
        seg, cp, rank, pred_flms, shared_demand, target_supply, aux_cut, aux_follow, aux_one, aux_zero, aux_rem,
        flm, supply, allocated_now, step, rec, proj, d60m, mil, vel, cfg,
    )
    data = {
        "segment_allocate": (seg == "Allocate").astype(float),
        "segment_review": (seg == "Review").astype(float),
        "classifier_probability": cp,
        "rank_priority": rank,
        "pred_flms_raw": pred_flms,
        "shared_demand_score": shared_demand,
        "target_final_supply_prediction": target_supply,
        "aux_cut_rec": aux_cut,
        "aux_follow_rec": aux_follow,
        "aux_one_flm": aux_one,
        "aux_zero_out": aux_zero,
        "aux_below_flm_remainder": aux_rem,
        "step_units": step,
        "step_flms": step / flm,
        "allocated_units_so_far": allocated_now,
        "allocated_flms_so_far": allocated_now / flm,
        "dc_remaining_units": remaining_dc,
        "dc_remaining_flms": remaining_dc / flm,
        "row_cap_remaining_units": np.maximum(0.0, row_cap - allocated_now),
        "row_cap_remaining_flms": np.maximum(0.0, row_cap - allocated_now) / flm,
        "flm": flm,
        "mil": mil,
        "supply": supply,
        "current_supply": current_supply,
        "final_supply_after_step": after_supply,
        "alloc_rec": rec,
        "proj_demand": proj,
        "d60_month": d60m,
        "weighted_velocity": vel,
        "demand_target": demand_target,
        "demand_gap_flm_before": np.maximum(0.0, demand_target - current_supply) / flm,
        "demand_gap_flm_after": np.maximum(0.0, demand_target - after_supply) / flm,
        "shared_demand_gap_flm_before": np.maximum(0.0, shared_demand - current_supply) / flm,
        "shared_demand_gap_flm_after": np.maximum(0.0, shared_demand - after_supply) / flm,
        "target_supply_gap_flm_before": np.maximum(0.0, target_supply - current_supply) / flm,
        "target_supply_gap_flm_after": np.maximum(0.0, target_supply - after_supply) / flm,
        "proj_gap_flm_before": np.maximum(0.0, proj - current_supply) / flm,
        "proj_gap_flm_after": np.maximum(0.0, proj - after_supply) / flm,
        "velocity_gap_flm_before": np.maximum(0.0, vel - current_supply) / flm,
        "velocity_gap_flm_after": np.maximum(0.0, vel - after_supply) / flm,
        "rec_remaining_flm_before": rec_remaining_before,
        "rec_remaining_flm_after": rec_remaining_after,
        "pred_remaining_flm_before": pred_remaining_before,
        "pred_remaining_flm_after": pred_remaining_after,
        "over_proj_after": ((proj > 0) & (after_supply > proj + flm)).astype(float),
        "over_d60_after": ((d60m > 0) & (after_supply > d60m + flm)).astype(float),
        "zero_demand_positive_step": ((np.maximum.reduce([mil, proj, d60m, vel]) <= 0) & (step > 0)).astype(float),
        "candidate_score_formula": formula_scores,
        "classifier_above_threshold": (cp >= threshold).astype(float),
        "review_soft_excess_after": np.where(seg == "Review", np.maximum(0.0, (allocated_now + step) / flm - cfg.review_soft_max_flm), 0.0),
    }
    return np.column_stack([np.nan_to_num(data[name], nan=0.0, posinf=0.0, neginf=0.0) for name in STEP_FEATURE_NAMES]).astype(np.float32)


def _build_compact_step_scorer_training_data(
    canon: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: IterativeFLMConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build bounded marginal-FLM examples from source Allocate/Review rows.

    This is intentionally source-row based.  It does not create one example per
    item-row per historical cycle.  Each source row contributes a first-FLM
    example, a small number of continuation examples when the historical target
    received multiple FLMs, and one stop example after the historical target.
    """
    static = _static_arrays(canon)
    static["row_cap"] = _row_caps(canon, signals, cfg, static)
    target = np.minimum(np.maximum(numeric_series(canon, TARGET_COLUMN).to_numpy(float), 0.0), static["row_cap"])
    group_key = item_group_key(canon).reset_index(drop=True)
    group_dc = np.zeros(len(canon), dtype=float)
    for _, positions in group_key.groupby(group_key, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        elig = _eligible_model_indices(signals, idxs)
        if len(elig):
            group_dc[elig] = _group_dc_pool(static, elig)

    example_idx: List[int] = []
    example_allocated: List[float] = []
    example_remaining_dc: List[float] = []
    example_step: List[float] = []
    labels: List[float] = []
    rows: List[Dict[str, Any]] = []

    for i in range(len(canon)):
        if str(signals.at[i, "model_segment"]) not in {"Allocate", "Review"}:
            continue
        flm = max(float(static["flm"][i]), 1.0)
        row_cap = max(float(static["row_cap"][i]), 0.0)
        dc_avail = max(float(group_dc[i]), 0.0)
        if row_cap <= 1e-9 or dc_avail <= 1e-9:
            continue
        first_step = flm if dc_avail >= flm - 1e-9 else (dc_avail if cfg.allow_final_partial else 0.0)
        if first_step <= 1e-9:
            continue
        capped_target = max(float(target[i]), 0.0)

        states = [0.0]
        # Add a few positive continuation states for rows that historically got
        # multiple FLMs.  This teaches the step scorer how need changes after a
        # row has already received inventory without exploding into every cycle.
        positive_steps = int(min(cfg.compact_positive_states_per_row, max(0, np.floor(capped_target / flm) - 1)))
        for k in range(1, positive_steps + 1):
            states.append(float(k * flm))
        if cfg.compact_add_stop_examples and capped_target > 0:
            stop_state = min(row_cap, capped_target)
            if all(abs(stop_state - s) > 1e-9 for s in states):
                states.append(float(stop_state))

        for state in states:
            if state >= row_cap - 1e-9:
                continue
            remaining_dc = max(dc_avail - state, 0.0)
            if remaining_dc <= 1e-9:
                continue
            step = flm if remaining_dc >= flm - 1e-9 else (remaining_dc if cfg.allow_final_partial else 0.0)
            if step <= 1e-9 or state + step > row_cap + 1e-9:
                continue
            label = 1.0 if capped_target >= state + min(step, flm) - 1e-9 else 0.0
            example_idx.append(i)
            example_allocated.append(state)
            example_remaining_dc.append(remaining_dc)
            example_step.append(step)
            labels.append(label)
            rows.append({
                "row_index": int(i),
                "allocation_group": str(group_key.iloc[i]),
                "training_mode": "compact_source_rows",
                "source_row_example_number": int(len(states)),
                "allocated_units_so_far": float(state),
                "label_should_get_next_flm": float(label),
                "step_units": float(step),
                "remaining_dc": float(remaining_dc),
                "target_units_capped": float(capped_target),
            })

    if not example_idx:
        return np.zeros((0, len(STEP_FEATURE_NAMES)), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), pd.DataFrame(rows)
    X = build_step_feature_matrix_examples(
        canon, signals,
        np.asarray(example_idx, dtype=int),
        np.asarray(example_allocated, dtype=float),
        np.asarray(example_remaining_dc, dtype=float),
        np.asarray(example_step, dtype=float),
        asdict(cfg),
        static,
    )
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    detail = pd.DataFrame(rows)
    return X, y, detail


def _build_full_cycle_step_scorer_training_data(df: pd.DataFrame, signals: pd.DataFrame, cfg: IterativeFLMConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Original full-cycle simulator retained for optional comparison."""
    canon = ensure_columns(df.copy(), include_target=True).reset_index(drop=True)
    signals = _prepare_signals(canon, signals)
    static = _static_arrays(canon)
    static["row_cap"] = _row_caps(canon, signals, cfg, static)
    group_key = item_group_key(canon).reset_index(drop=True)
    target = np.maximum(numeric_series(canon, TARGET_COLUMN).to_numpy(float), 0.0)
    rng = np.random.default_rng(cfg.step_training_seed)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    for group, positions in group_key.groupby(group_key, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        elig = _eligible_model_indices(signals, idxs)
        if len(elig) == 0:
            continue
        group_dc = _group_dc_pool(static, elig)
        remaining_dc = group_dc
        allocated = np.zeros(len(canon), dtype=float)
        cycle = 0
        while remaining_dc > 1e-9 and cycle < cfg.max_training_cycles_per_group:
            cycle += 1
            cand, steps = [], []
            for i in elig:
                flm = static["flm"][i]
                if allocated[i] >= static["row_cap"][i] - 1e-9:
                    continue
                if remaining_dc >= flm - 1e-9:
                    step = flm
                elif cfg.allow_final_partial:
                    step = remaining_dc
                else:
                    continue
                cand.append(i); steps.append(step)
            if not cand:
                break
            cand = np.asarray(cand, dtype=int)
            steps = np.asarray(steps, dtype=float)
            capped_target = np.minimum(target[cand], static["row_cap"][cand])
            labels = ((capped_target - allocated[cand]) >= np.minimum(steps, static["flm"][cand]) - 1e-9).astype(np.float32)
            positives = cand[labels > 0.5]
            include = np.arange(len(cand))
            if len(cand) > cfg.include_all_candidates_when_group_leq:
                pos_local = np.where(labels > 0.5)[0]
                neg_local = np.where(labels <= 0.5)[0]
                formula = []
                for local_j, i in enumerate(cand):
                    fs, _ = _formula_score(int(i), float(steps[local_j]), allocated, signals, cfg, static)
                    formula.append(fs)
                formula = np.asarray(formula)
                hard_neg = neg_local[np.argsort(-formula[neg_local])[:cfg.max_negative_candidates_per_cycle]] if len(neg_local) else np.array([], dtype=int)
                remaining_neg = np.setdiff1d(neg_local, hard_neg, assume_unique=False)
                rand_count = min(cfg.random_negative_candidates_per_cycle, len(remaining_neg))
                rand_neg = rng.choice(remaining_neg, size=rand_count, replace=False) if rand_count > 0 else np.array([], dtype=int)
                include = np.unique(np.concatenate([pos_local, hard_neg, rand_neg]))
            X_step = build_step_feature_matrix(canon, signals, allocated, remaining_dc, steps[include], cand[include], asdict(cfg), static)
            x_parts.append(X_step)
            y_parts.append(labels[include].reshape(-1, 1))
            for loc in include:
                rows.append({"allocation_group": group, "cycle_number": int(cycle), "row_index": int(cand[loc]), "label_should_get_next_flm": float(labels[loc]), "step_units": float(steps[loc]), "remaining_dc": float(remaining_dc), "training_mode": "full_cycle"})
            if len(positives) == 0:
                break
            capped_positive_target = np.minimum(target[positives], static["row_cap"][positives])
            best_i = positives[np.argmax((capped_positive_target - allocated[positives]) / np.maximum(static["flm"][positives], 1.0))]
            best_step = static["flm"][best_i] if remaining_dc >= static["flm"][best_i] - 1e-9 else remaining_dc
            best_step = min(best_step, max(0.0, min(target[best_i], static["row_cap"][best_i]) - allocated[best_i]), remaining_dc)
            if best_step <= 1e-9:
                break
            allocated[best_i] += best_step
            remaining_dc = max(0.0, remaining_dc - best_step)
    if not x_parts:
        return np.zeros((0, len(STEP_FEATURE_NAMES)), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), pd.DataFrame(rows)
    return np.vstack(x_parts).astype(np.float32), np.vstack(y_parts).astype(np.float32), pd.DataFrame(rows)


def build_step_scorer_training_data(df: pd.DataFrame, signals: pd.DataFrame, config: Dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create supervised examples for the neural marginal FLM step scorer.

    Only Allocate and Review source rows are used.  The default compact mode
    builds a bounded training object from those rows directly, so a pickle with
    33,567 eligible source rows no longer expands to more than a million cycle
    rows.  Set `step_training_mode="full_cycle"` in the optimizer config only
    when you intentionally want the older exhaustive simulator.
    """
    canon = ensure_columns(df.copy(), include_target=True).reset_index(drop=True)
    cfg = IterativeFLMConfig(**(config or {}))
    signals = _prepare_signals(canon, signals)
    owned_mask = signals["model_segment"].astype(str).isin(["Allocate", "Review"]).to_numpy()
    if not np.any(owned_mask):
        return np.zeros((0, len(STEP_FEATURE_NAMES)), dtype=np.float32), np.zeros((0, 1), dtype=np.float32), pd.DataFrame([])
    if not np.all(owned_mask):
        canon = canon.loc[owned_mask].reset_index(drop=True)
        signals = signals.loc[owned_mask].reset_index(drop=True)
    if str(cfg.step_training_mode).lower() == "full_cycle":
        return _build_full_cycle_step_scorer_training_data(canon, signals, cfg)
    return _build_compact_step_scorer_training_data(canon, signals, cfg)

def _score_row(
    idx: int,
    step_units: float,
    remaining_dc: float,
    allocated: np.ndarray,
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: IterativeFLMConfig,
    static: Dict[str, np.ndarray],
    step_scorer_model: Any | None = None,
) -> Tuple[float, str, Dict[str, float]]:
    seg = str(signals.at[idx, "model_segment"])
    if seg not in {"Allocate", "Review"}:
        return -1e12, "skip: non-eligible", {}
    cp = float(signals.at[idx, "classifier_probability"])
    threshold = cfg.allocate_threshold if seg == "Allocate" else cfg.review_threshold
    if cp < threshold:
        return -1e12, f"skip: classifier below threshold ({cp:.3f} < {threshold:.3f})", {}
    if allocated[idx] + step_units > static["row_cap"][idx] + 1e-9:
        return -1e12, "skip: row cap reached", {}
    formula, details = _formula_score(idx, step_units, allocated, signals, cfg, static)
    if step_scorer_model is not None and cfg.use_neural_step_scorer:
        X_step = build_step_feature_matrix(df, signals, allocated, remaining_dc, np.asarray([step_units]), np.asarray([idx]), asdict(cfg), static)
        neural_prob = float(np.asarray(step_scorer_model.predict(X_step)).reshape(-1)[0])
        details["neural_step_score"] = neural_prob
        details["formula_score"] = formula
        return neural_prob, "score: neural marginal FLM step scorer", details
    return formula, "score: deterministic marginal FLM formula", details


def apply_iterative_flm_allocator(df: pd.DataFrame, signals: pd.DataFrame, config: Dict[str, Any] | None = None, step_scorer_model: Any | None = None):
    """Cycle through each item group and allocate one FLM at a time.

    The scorer is evaluated in batches within each item cycle.  This is critical
    for testing speed with the neural step scorer: the old path called
    `step_scorer_model.predict()` once per candidate row per cycle, which could
    make testing look like it was accidentally using the million-row training
    cache.  This path builds one candidate matrix per cycle and performs one
    neural prediction call for all competing Allocate/Review rows.

    Returns: final_units, row_explain, group_audit, cycle_trace.
    """
    canon = ensure_columns(df.copy(), include_target=True).reset_index(drop=True)
    cfg = IterativeFLMConfig(**(config or {}))
    signals = _prepare_signals(canon, signals)
    n = len(canon)
    group_key = item_group_key(canon).reset_index(drop=True)
    quality = item_group_quality(canon).reset_index(drop=True)
    static = _static_arrays(canon)
    static["row_cap"] = _row_caps(canon, signals, cfg, static)
    allocated = np.zeros(n, dtype=float)
    row_reasons = ["not evaluated" for _ in range(n)]
    cycle_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    trace_limit = max(0, int(getattr(cfg, "max_cycle_trace_rows", 250000)))
    record_trace = bool(getattr(cfg, "record_cycle_trace", True)) and trace_limit != 0

    for group, positions in group_key.groupby(group_key, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        eligible_idxs = _eligible_model_indices(signals, idxs)
        if len(eligible_idxs) == 0:
            for idx in idxs:
                if row_reasons[idx] == "not evaluated":
                    row_reasons[idx] = "blank: not Allocate/Review"
            group_rows.append({
                "allocation_group": group,
                "allocation_group_quality": str(quality.iloc[idxs[0]]) if len(idxs) else "unknown",
                "dc_start": 0.0,
                "allocated_units": 0.0,
                "dc_remaining": 0.0,
                "rows_in_group": int(len(idxs)),
                "eligible_rows_in_group": 0,
                "cycles_run": 0,
                "partial_used": False,
                "over_allocated": False,
                "stop_reason": "no Allocate/Review rows in group",
            })
            continue

        group_dc = _group_dc_pool(static, eligible_idxs)
        remaining = group_dc
        partial_used = False
        cycle_no = 0
        stop_reason = "not started"
        neural_enabled = step_scorer_model is not None and cfg.use_neural_step_scorer

        while remaining > 1e-9 and cycle_no < cfg.max_cycles_per_group:
            cycle_no += 1
            cand_idx: List[int] = []
            cand_step: List[float] = []
            cand_partial: List[bool] = []
            cand_min_score: List[float] = []
            stop_hint = "no eligible row above threshold"

            # Build the candidate list first, applying cheap row-level vetoes.
            for idx in eligible_idxs:
                seg = str(signals.at[idx, "model_segment"])
                flm = static["flm"][idx]
                if remaining >= flm - 1e-9:
                    step = float(flm); partial = False
                elif cfg.allow_final_partial and (not partial_used or not cfg.partial_only_once_per_group):
                    step = float(remaining); partial = True
                else:
                    continue

                cp = float(signals.at[idx, "classifier_probability"])
                threshold = cfg.allocate_threshold if seg == "Allocate" else cfg.review_threshold
                if cp < threshold:
                    if row_reasons[idx] == "not evaluated":
                        row_reasons[idx] = f"blank/stop: classifier below threshold ({cp:.3f} < {threshold:.3f})"
                    continue
                if allocated[idx] + step > static["row_cap"][idx] + 1e-9:
                    if row_reasons[idx] == "not evaluated":
                        row_reasons[idx] = "blank/stop: row cap reached"
                    continue

                if neural_enabled:
                    min_score = cfg.min_review_neural_score if seg == "Review" else cfg.min_allocate_neural_score
                    if partial:
                        min_score = max(min_score, cfg.min_partial_neural_score)
                else:
                    min_score = cfg.min_review_cycle_score if seg == "Review" else cfg.min_allocate_cycle_score
                    if partial:
                        min_score = max(min_score, cfg.min_partial_cycle_score)
                cand_idx.append(int(idx))
                cand_step.append(float(step))
                cand_partial.append(bool(partial))
                cand_min_score.append(float(min_score))

            if not cand_idx:
                stop_reason = stop_hint
                break

            cand_idx_arr = np.asarray(cand_idx, dtype=int)
            cand_step_arr = np.asarray(cand_step, dtype=float)
            if neural_enabled:
                X_step = build_step_feature_matrix(canon, signals, allocated, remaining, cand_step_arr, cand_idx_arr, asdict(cfg), static)
                scores = np.asarray(step_scorer_model.predict(X_step), dtype=float).reshape(-1)
                scorer_type = "neural_step_scorer"
                score_reason = "score: neural marginal FLM step scorer"
            else:
                scores = np.asarray([_formula_score(int(i), float(st), allocated, signals, cfg, static)[0] for i, st in zip(cand_idx_arr, cand_step_arr)], dtype=float)
                scorer_type = "deterministic_formula"
                score_reason = "score: deterministic marginal FLM formula"

            min_scores = np.asarray(cand_min_score, dtype=float)
            valid = scores >= min_scores
            if not np.any(valid):
                best_pos = int(np.argmax(scores))
                best_score = float(scores[best_pos])
                best_min = float(min_scores[best_pos])
                best_idx = int(cand_idx_arr[best_pos])
                if row_reasons[best_idx] == "not evaluated":
                    row_reasons[best_idx] = f"blank/stop: marginal score below threshold ({best_score:.3f} < {best_min:.3f})"
                stop_reason = f"best score below threshold ({best_score:.3f} < {best_min:.3f})"
                break

            # Choose the highest valid marginal FLM step.
            valid_positions = np.where(valid)[0]
            best_pos = int(valid_positions[np.argmax(scores[valid_positions])])
            score = float(scores[best_pos])
            idx = int(cand_idx_arr[best_pos])
            step = float(cand_step_arr[best_pos])
            partial = bool(cand_partial[best_pos])
            min_score = float(min_scores[best_pos])
            formula, best_details = _formula_score(idx, step, allocated, signals, cfg, static)
            if neural_enabled:
                best_details["neural_step_score"] = score
                best_details["formula_score"] = formula
            before_row = allocated[idx]
            before_dc = remaining
            allocated[idx] += step
            remaining = max(0.0, remaining - step)
            if partial:
                partial_used = True
            row_reasons[idx] = "allocated by neural iterative item FLM cycle" if neural_enabled else "allocated by iterative item FLM cycle"
            if record_trace and len(cycle_rows) < trace_limit:
                cycle_rows.append({
                    "allocation_group": group,
                    "cycle_number": int(cycle_no),
                    "row_index": int(idx),
                    "model_segment": str(signals.at[idx, "model_segment"]),
                    "step_units": float(step),
                    "row_alloc_before": float(before_row),
                    "row_alloc_after": float(allocated[idx]),
                    "dc_before_cycle": float(before_dc),
                    "dc_after_cycle": float(remaining),
                    "cycle_score": float(score),
                    "minimum_required_score": float(min_score),
                    "partial_final_remainder": bool(partial),
                    "scorer_type": scorer_type,
                    "reason": score_reason,
                    **{k: float(v) for k, v in best_details.items()},
                })
            if remaining <= 1e-9:
                stop_reason = "DC exhausted"
                break

        if cycle_no >= cfg.max_cycles_per_group:
            stop_reason = "max_cycles_per_group reached"
        allocated = _apply_row_caps(allocated, static["row_cap"], row_reasons)
        total = float(allocated[eligible_idxs].sum())
        group_rows.append({
            "allocation_group": group,
            "allocation_group_quality": str(quality.iloc[idxs[0]]),
            "dc_start": float(group_dc),
            "allocated_units": total,
            "dc_remaining": float(group_dc - total),
            "rows_in_group": int(len(idxs)),
            "eligible_rows_in_group": int(len(eligible_idxs)),
            "cycles_run": int(cycle_no),
            "partial_used": bool(partial_used),
            "over_allocated": bool(total > group_dc + 1e-9),
            "stop_reason": stop_reason,
            "trace_truncated": bool(record_trace and len(cycle_rows) >= trace_limit),
        })

    # Hard repairs: no row may exceed Alloc. Rec. + 1 FLM and no group may
    # spend more than the eligible Allocate/Review DC pool.
    allocated = _apply_row_caps(allocated, static["row_cap"], row_reasons)
    for group, positions in group_key.groupby(group_key, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        eligible_idxs = _eligible_model_indices(signals, idxs)
        if len(eligible_idxs) == 0:
            continue
        group_dc = _group_dc_pool(static, eligible_idxs)
        excess = float(allocated[eligible_idxs].sum() - group_dc)
        if excess > 1e-9:
            order = sorted(eligible_idxs, key=lambda i: (allocated[i], signals.at[i, "rank_priority"]))
            for idx in order:
                take = min(excess, allocated[idx])
                allocated[idx] -= take
                excess -= take
                row_reasons[idx] = "repaired: reduced to prevent item-level DC overspend"
                if excess <= 1e-9:
                    break
    allocated = _apply_row_caps(allocated, static["row_cap"], row_reasons)
    final_units = np.rint(np.maximum(allocated, 0.0)).astype(int)
    final_units = np.minimum(final_units, np.rint(static["row_cap"]).astype(int))
    row_explain = pd.DataFrame({
        "row_index": np.arange(n, dtype=int),
        "allocation_group": group_key.to_numpy(),
        "allocation_group_quality": quality.to_numpy(),
        "model_segment": signals["model_segment"].to_numpy(),
        "classifier_probability": signals["classifier_probability"].to_numpy(float),
        "rank_priority": signals["rank_priority"].to_numpy(float),
        "pred_flms_raw": signals["pred_flms_raw"].to_numpy(float),
        "shared_demand_score": signals["shared_demand_score"].to_numpy(float),
        "target_final_supply_prediction": signals["target_final_supply_prediction"].to_numpy(float),
        "row_cap_units": static["row_cap"],
        "predicted_final_alloc": final_units,
        "predicted_final_supply": static["supply"] + final_units,
        "allocated_flms": final_units / np.maximum(static["flm"], 1.0),
        "decision_reason": row_reasons,
    })
    group_audit = pd.DataFrame(group_rows)
    if not group_audit.empty:
        group_audit["dc_remaining"] = group_audit["dc_start"] - group_audit["allocated_units"]
        group_audit["over_allocated"] = group_audit["allocated_units"] > group_audit["dc_start"] + 1e-9
    cycle_trace = pd.DataFrame(cycle_rows)
    return final_units, row_explain, group_audit, cycle_trace
