from __future__ import annotations

import io
import itertools
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from allocation_feature_engineering import (
    APPROVED_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    detect_header_and_table,
    ensure_columns,
    load_feature_config,
    numeric_series,
    text_series,
    eligible_mask,
    allocate_mask,
    review_mask,
    raw_dc_bucket,
    flm_dc_bucket,
)
from allocation_iterative_flm_optimizer import (
    STEP_FEATURE_NAMES,
    apply_iterative_flm_allocator,
    item_group_key,
    item_group_quality,
    load_optimizer_config,
)
try:
    from allocation_nn_core import NumpyMLP, make_meta_features
except ImportError:
    from allocation_nn_core import NumpyMLP

    def make_meta_features(x, classifier_probs=None, rank_scores=None, aux_outputs=None):
        """Fallback for older core files; keeps Streamlit deploys from crashing."""
        x = np.asarray(x, dtype=np.float32)
        parts = [x]
        if classifier_probs is not None:
            parts.append(np.asarray(classifier_probs, dtype=np.float32).reshape(-1, 1))
        if rank_scores is not None:
            parts.append(np.asarray(rank_scores, dtype=np.float32).reshape(-1, 1))
        if aux_outputs is not None:
            aux = np.asarray(aux_outputs, dtype=np.float32)
            if aux.ndim == 1:
                aux = aux.reshape(-1, 1)
            parts.append(aux)
        return np.concatenate(parts, axis=1).astype(np.float32)

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR
_PLOTLY_CHART_COUNTER = itertools.count(1)

st.set_page_config(
    page_title="Allocation Iterative FLM Model",
    page_icon="📦",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Visual style
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    .metric-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(128,128,128,.045);
        min-height: 115px;
    }
    .soft-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        background: rgba(128,128,128,.035);
        margin-bottom: .75rem;
    }
    .small-muted {opacity:.72; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 Allocation Iterative FLM Model")
st.markdown(
    """
    <div class="soft-card">
      <b>Upload an allocation workbook, predict Final Alloc., audit against existing allocations, and inspect exactly how the model uses features.</b>
      <div class="small-muted" style="margin-top:.35rem;">
        This app is built for the current Model 3 single-review neural system: shared demand, shared final supply, direct Allocate neural sizing, Review-only neural iterative 1-FLM scoring, and no AK/Site 802 specialist routing.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load helpers
# -----------------------------------------------------------------------------

def read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {} if default is None else default


def read_csv_if_exists(name: str) -> pd.DataFrame:
    p = ART / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"


def fmt_pct(x) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "—"


def fmt_num(x, nd=3) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def clean_num(s) -> np.ndarray:
    return pd.to_numeric(pd.Series(s), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)


def int_or_blank(values) -> pd.Series:
    arr = np.rint(clean_num(values)).astype(np.int64)
    return pd.Series(np.where(arr > 0, arr.astype(object), ""))


def recommendation_display_frame(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a display/download copy where zero allocation recommendations are blank.

    The model still keeps numeric recommendation units internally for charts, filters,
    metrics, and business-rule checks. This helper is only for user-facing tables
    and exported recommendation CSVs.
    """
    out = df.copy()
    if columns is None:
        columns = [
            "Predicted Final Alloc",
            "Predicted Final Alloc Audit",
            "Recommended Final Alloc",
            "Model Recommended Final Alloc",
        ]
    for col in columns:
        if col in out.columns:
            out[col] = int_or_blank(out[col]).values
    return out


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    aliases = {"final alloc", "final alloc.", "final allocate", "final allocation"}
    for c in df.columns:
        key = str(c).strip().lower().replace("_", " ")
        key = " ".join(key.split())
        if key in aliases:
            return c
    return TARGET_COLUMN if TARGET_COLUMN in df.columns else None


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

REQUIRED_MODEL_KEYS = [
    "shared_demand",
    "shared_final_supply",
    "allocate_classifier",
    "allocate_ranker",
    "allocate_auxiliary",
    "allocate_regressor",
    "review_classifier",
    "review_ranker",
    "review_auxiliary",
    "review_regressor",
    "iterative_flm_step_scorer",
]


def _find_model_file(model_dir: Path, file_name: str, manifest: Optional[dict] = None) -> Path:
    direct = model_dir / file_name
    if direct.exists():
        return direct
    manifest = manifest or {}
    entry = manifest.get("models", {}).get(file_name, {}) if isinstance(manifest, dict) else {}
    parts = entry.get("parts") or []
    if parts:
        tmp = Path(tempfile.mkdtemp(prefix="allocation_model_parts_")) / file_name
        with open(tmp, "wb") as out:
            for part in parts:
                part_path = model_dir / part
                if not part_path.exists():
                    raise FileNotFoundError(f"Missing split model part: {part_path.name}")
                out.write(part_path.read_bytes())
        return tmp
    raise FileNotFoundError(f"Missing model file: {file_name}")


@st.cache_resource(show_spinner="Loading neural allocation models...")
def load_bundle():
    registry_path = ART / "registry.json"
    feature_config_path = ART / "feature_config.json"
    if not registry_path.exists():
        raise FileNotFoundError("registry.json was not found in the app folder. Put the model artifacts in the same GitHub repo folder as app.py.")
    if not feature_config_path.exists():
        raise FileNotFoundError("feature_config.json was not found in the app folder. Put the model artifacts in the same GitHub repo folder as app.py.")

    registry = read_json(registry_path)
    feature_config = load_feature_config(feature_config_path)
    manifest = read_json(ART / "model_part_manifest.json", {})
    model_map = registry.get("models", {})

    missing_keys = [k for k in REQUIRED_MODEL_KEYS if k not in model_map]
    if missing_keys:
        raise KeyError(f"registry.json is missing model keys: {missing_keys}")

    models = {}
    for key in REQUIRED_MODEL_KEYS:
        fn = model_map[key]
        path = _find_model_file(ART, fn, manifest)
        models[key] = NumpyMLP.load(path)

    opt_name = registry.get("allocator", "iterative_flm_optimizer.json")
    optimizer_config = load_optimizer_config(ART / opt_name)
    for k in ["allocate_threshold", "review_threshold", "min_allocate_neural_score", "min_review_neural_score", "min_partial_neural_score"]:
        if k in registry:
            optimizer_config[k] = registry[k]

    meta = {
        "registry": registry,
        "feature_config": feature_config,
        "optimizer_config": optimizer_config,
        "models": models,
        "model_summary": read_json(ART / "model_summary.json", {}),
        "training_history": read_json(ART / "training_history.json", {}),
        "tuning_output": read_json(ART / "tuning_output.json", {}),
        "early_stopping": read_json(ART / "early_stopping_summary.json", {}),
        "step_cache_meta": read_json(ART / "iterative_step_scorer_training_cache_metadata.json", {}),
        "test_input_audit": read_json(ART / "test_input_audit.json", {}),
        "part_manifest": manifest,
    }
    return meta


# -----------------------------------------------------------------------------
# Workbook ingestion
# -----------------------------------------------------------------------------

def _sheet_names_from_upload(uploaded) -> List[str]:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return ["csv"]
    data = uploaded.getvalue()
    engine = "pyxlsb" if name.endswith(".xlsb") else None
    try:
        xl = pd.ExcelFile(io.BytesIO(data), engine=engine)
        return xl.sheet_names
    except Exception:
        return []


def read_upload(uploaded, sheet_name: Optional[str] = None) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object, low_memory=False)
        return detect_header_and_table(raw)

    engine = "pyxlsb" if name.endswith(".xlsb") else None
    xl = pd.ExcelFile(io.BytesIO(data), engine=engine)
    if sheet_name and sheet_name != "Auto detect":
        sheets = [sheet_name]
    else:
        working = [s for s in xl.sheet_names if "working" in str(s).lower() and "table" in str(s).lower()]
        sheets = working or xl.sheet_names[:8]

    best_df = None
    best_score = -1
    last_error = None
    for sheet in sheets:
        try:
            raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, dtype=object, engine=engine)
            table = detect_header_and_table(raw)
            score = sum(1 for c in ["Flag", "Dc Avail", "Final Alloc.", "Alloc. Rec.", "Site", "FLM", "Class Name", "Line Name"] if c in table.columns)
            if score > best_score:
                best_score = score
                best_df = table
        except Exception as e:
            last_error = e
    if best_df is None:
        raise ValueError(f"Could not detect an allocation working table. Last error: {last_error}")
    return best_df


def remove_repeated_headers_and_filter(df: pd.DataFrame, only_eligible: bool = True) -> pd.DataFrame:
    work = df.dropna(how="all").copy()
    if len(work) == 0:
        return work
    row_text = work.apply(lambda r: "|".join("" if pd.isna(v) else str(v) for v in r.to_numpy()), axis=1).str.upper()
    repeated = row_text.str.contains("FINAL ALLOC", na=False) & row_text.str.contains("ALLOC", na=False) & row_text.str.contains("FLAG", na=False)
    work = work.loc[~repeated].reset_index(drop=True)
    if only_eligible:
        canon = ensure_columns(work, include_target=True)
        work = work.loc[eligible_mask(canon)].reset_index(drop=True)
    return work


# -----------------------------------------------------------------------------
# Prediction path
# -----------------------------------------------------------------------------

def _predict_segment(segment: str, X: np.ndarray, models: Dict[str, NumpyMLP]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clf = models[f"{segment}_classifier"]
    ranker = models[f"{segment}_ranker"]
    aux = models[f"{segment}_auxiliary"]
    reg = models[f"{segment}_regressor"]
    prob = clf.predict(X).reshape(-1)
    rank = ranker.predict(X).reshape(-1)
    aux_pred = aux.predict(X)
    meta_x = make_meta_features(X, prob, rank, aux_pred)
    pred_flms = np.maximum(reg.predict(meta_x).reshape(-1), 0.0)
    return prob, rank, aux_pred, pred_flms


def _row_cap_units(canon: pd.DataFrame, optimizer_config: Dict) -> np.ndarray:
    """Hard row cap used by both final paths: Alloc. Rec. + 1 FLM by default."""
    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    rec = np.maximum(numeric_series(canon, "Alloc. Rec.").to_numpy(float), 0.0)
    max_above = float(optimizer_config.get("max_above_rec_flm", 1.0) or 1.0)
    return np.maximum(0.0, rec + max_above * flm)


def _direct_allocate_neural_units(canon: pd.DataFrame, signals: pd.DataFrame, optimizer_config: Dict) -> np.ndarray:
    """Use the trained Allocate neural regressor directly for Allocate rows.

    Review rows are intentionally left at zero here because they are handled by
    the neural iterative FLM step scorer. The direct Allocate path still uses
    the Allocate classifier gate, FLM rounding, and the row cap of
    Alloc. Rec. + 1 FLM.
    """
    n = len(canon)
    out = np.zeros(n, dtype=float)
    mask = allocate_mask(canon)
    if not np.any(mask):
        return out.astype(int)

    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    pred_flms = np.maximum(pd.to_numeric(signals["pred_flms_raw"], errors="coerce").fillna(0.0).to_numpy(float), 0.0)
    classifier = pd.to_numeric(signals["classifier_probability"], errors="coerce").fillna(0.0).to_numpy(float)
    threshold = float(optimizer_config.get("allocate_threshold", 0.0) or 0.0)
    row_cap = _row_cap_units(canon, optimizer_config)

    # The regressor is trained in FLM units. Convert back to units by rounding
    # FLMs first, so Allocate recommendations remain normal pack multiples.
    direct_units = np.rint(pred_flms) * flm
    direct_units[classifier < threshold] = 0.0
    direct_units = np.minimum(np.maximum(direct_units, 0.0), row_cap)
    out[mask] = direct_units[mask]
    return np.rint(out).astype(int)


def _group_dc_pool(canon: pd.DataFrame, positions: np.ndarray) -> float:
    if len(positions) == 0:
        return 0.0
    dc = np.maximum(numeric_series(canon, "Dc Avail").to_numpy(float), 0.0)
    vals = dc[positions]
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if len(vals) else 0.0


def _repair_units_to_group_dc(canon: pd.DataFrame, units: np.ndarray, priority: np.ndarray | None = None) -> np.ndarray:
    """Hard repair so the direct Allocate path cannot spend more than item DC."""
    fixed = np.maximum(np.asarray(units, dtype=float).copy(), 0.0)
    keys = item_group_key(canon).reset_index(drop=True)
    if priority is None:
        priority = np.zeros(len(canon), dtype=float)
    priority = np.asarray(priority, dtype=float)
    for _, positions in keys.groupby(keys, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        group_dc = _group_dc_pool(canon, idxs)
        excess = float(fixed[idxs].sum() - group_dc)
        if excess <= 1e-9:
            continue
        # Reduce lowest-priority/highest-allocation rows first. This protects
        # the strongest direct neural recommendations when item DC is tight.
        order = sorted(idxs, key=lambda i: (priority[i], -fixed[i]))
        for idx in order:
            if fixed[idx] <= 0:
                continue
            take = min(excess, fixed[idx])
            fixed[idx] -= take
            excess -= take
            if excess <= 1e-9:
                break
    return np.rint(np.maximum(fixed, 0.0)).astype(int)


def _remaining_dc_for_review(canon: pd.DataFrame, allocate_units: np.ndarray) -> np.ndarray:
    """Return a Dc Avail vector where Review rows see DC left after direct Allocate spend."""
    remaining_dc = np.maximum(numeric_series(canon, "Dc Avail").to_numpy(float), 0.0).copy()
    keys = item_group_key(canon).reset_index(drop=True)
    allocate_units = np.asarray(allocate_units, dtype=float)
    for _, positions in keys.groupby(keys, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        group_dc = _group_dc_pool(canon, idxs)
        spent = float(np.maximum(allocate_units[idxs], 0.0).sum())
        remaining_dc[idxs] = max(0.0, group_dc - spent)
    return remaining_dc


def _combine_group_audit(canon: pd.DataFrame, allocate_units: np.ndarray, review_units: np.ndarray, review_group_audit: pd.DataFrame) -> pd.DataFrame:
    keys = item_group_key(canon).reset_index(drop=True)
    quality = item_group_quality(canon).reset_index(drop=True)
    review_group_map = {}
    if isinstance(review_group_audit, pd.DataFrame) and not review_group_audit.empty and "allocation_group" in review_group_audit.columns:
        review_group_map = review_group_audit.set_index("allocation_group").to_dict(orient="index")
    rows = []
    for group, positions in keys.groupby(keys, sort=False).groups.items():
        idxs = np.asarray(list(positions), dtype=int)
        dc_start = _group_dc_pool(canon, idxs)
        allocate_spend = float(np.maximum(allocate_units[idxs], 0.0).sum())
        review_spend = float(np.maximum(review_units[idxs], 0.0).sum())
        total = allocate_spend + review_spend
        rg = review_group_map.get(group, {})
        rows.append({
            "allocation_group": group,
            "allocation_group_quality": str(quality.iloc[idxs[0]]) if len(idxs) else "unknown",
            "dc_start": float(dc_start),
            "allocate_neural_units": float(allocate_spend),
            "review_iterative_units": float(review_spend),
            "allocated_units": float(total),
            "dc_remaining": float(dc_start - total),
            "rows_in_group": int(len(idxs)),
            "eligible_rows_in_group": int(len(idxs)),
            "cycles_run": int(rg.get("cycles_run", 0) or 0),
            "partial_used": bool(rg.get("partial_used", False)),
            "over_allocated": bool(total > dc_start + 1e-9),
            "stop_reason": str(rg.get("stop_reason", "allocate direct neural path; no review cycle for this group")),
            "trace_truncated": bool(rg.get("trace_truncated", False)),
        })
    return pd.DataFrame(rows)


def predict_dataframe(df: pd.DataFrame, bundle: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canon = ensure_columns(df.copy(), include_target=True).reset_index(drop=True)
    X_base, _, canon = build_feature_matrix(canon, config=bundle["feature_config"], fit=False)
    models = bundle["models"]

    shared_demand = np.maximum(models["shared_demand"].predict(X_base).reshape(-1), 0.0)
    shared_final_supply = np.maximum(models["shared_final_supply"].predict(X_base).reshape(-1), 0.0)
    X_with_shared = np.concatenate([X_base, shared_demand.reshape(-1, 1), shared_final_supply.reshape(-1, 1)], axis=1).astype(np.float32)

    signals = pd.DataFrame({
        "model_segment": "none",
        "classifier_probability": np.zeros(len(canon)),
        "rank_priority": np.zeros(len(canon)),
        "pred_flms_raw": np.zeros(len(canon)),
        "shared_demand_score": shared_demand,
        "target_final_supply_prediction": shared_final_supply,
        "aux_cut_rec": np.zeros(len(canon)),
        "aux_follow_rec": np.zeros(len(canon)),
        "aux_one_flm": np.zeros(len(canon)),
        "aux_zero_out": np.zeros(len(canon)),
        "aux_below_flm_remainder": np.zeros(len(canon)),
    })

    for segment, mask, display in [
        ("allocate", allocate_mask(canon), "Allocate"),
        ("review", review_mask(canon), "Review"),
    ]:
        if mask.any():
            p, r, aux, flms = _predict_segment(segment, X_with_shared[mask], models)
            idx = np.where(mask)[0]
            signals.loc[idx, "model_segment"] = display
            signals.loc[idx, "classifier_probability"] = p
            signals.loc[idx, "rank_priority"] = r
            signals.loc[idx, "pred_flms_raw"] = flms
            aux = np.asarray(aux)
            for j, name in enumerate(["aux_cut_rec", "aux_follow_rec", "aux_one_flm", "aux_zero_out", "aux_below_flm_remainder"]):
                if aux.ndim == 2 and aux.shape[1] > j:
                    signals.loc[idx, name] = aux[:, j]

    # Final routing:
    #   • Allocate rows use the trained Allocate neural regressor directly.
    #   • Review rows use the neural iterative FLM step scorer.
    # The Review cycle sees only the DC remaining after the direct Allocate path.
    optimizer_config = dict(bundle["optimizer_config"] or {})
    priority = pd.to_numeric(signals["classifier_probability"], errors="coerce").fillna(0.0).to_numpy(float)
    allocate_units = _direct_allocate_neural_units(canon, signals, optimizer_config)
    allocate_units = _repair_units_to_group_dc(canon, allocate_units, priority=priority)

    review_units = np.zeros(len(canon), dtype=int)
    row_explain = pd.DataFrame({
        "row_index": np.arange(len(canon), dtype=int),
        "allocation_group": item_group_key(canon).to_numpy(),
        "allocation_group_quality": item_group_quality(canon).to_numpy(),
        "model_segment": signals["model_segment"].to_numpy(),
        "classifier_probability": signals["classifier_probability"].to_numpy(float),
        "rank_priority": signals["rank_priority"].to_numpy(float),
        "pred_flms_raw": signals["pred_flms_raw"].to_numpy(float),
        "shared_demand_score": signals["shared_demand_score"].to_numpy(float),
        "target_final_supply_prediction": signals["target_final_supply_prediction"].to_numpy(float),
        "row_cap_units": _row_cap_units(canon, optimizer_config),
        "predicted_final_alloc": np.zeros(len(canon), dtype=int),
        "predicted_final_supply": numeric_series(canon, "Supply").to_numpy(float),
        "allocated_flms": np.zeros(len(canon), dtype=float),
        "decision_reason": np.where(
            allocate_mask(canon),
            "allocated by direct Allocate neural regressor; Review rows use iterative FLM scorer",
            "pending Review iterative scorer",
        ),
        "final_decision_path": np.where(allocate_mask(canon), "Allocate direct neural regressor", "Review iterative FLM scorer"),
    })

    review_group_audit = pd.DataFrame()
    cycle_trace = pd.DataFrame()
    rev_mask = review_mask(canon)
    if np.any(rev_mask):
        review_canon = canon.loc[rev_mask].copy().reset_index(drop=True)
        review_signals = signals.loc[rev_mask].copy().reset_index(drop=True)
        review_original_idx = np.where(rev_mask)[0]
        remaining_dc_all = _remaining_dc_for_review(canon, allocate_units)
        review_canon["Dc Avail"] = remaining_dc_all[rev_mask]
        r_units, r_explain, review_group_audit, cycle_trace = apply_iterative_flm_allocator(
            review_canon,
            review_signals,
            config=optimizer_config,
            step_scorer_model=models.get("iterative_flm_step_scorer"),
        )
        review_units[review_original_idx] = np.asarray(r_units, dtype=int)
        if not r_explain.empty:
            mapped = r_explain.copy()
            mapped["row_index"] = review_original_idx[mapped["row_index"].astype(int).to_numpy()]
            for _, row in mapped.iterrows():
                idx = int(row["row_index"])
                for col in mapped.columns:
                    if col == "row_index":
                        continue
                    row_explain.at[idx, col] = row[col]
                row_explain.at[idx, "final_decision_path"] = "Review iterative FLM scorer"

    final_units = np.asarray(allocate_units, dtype=int) + np.asarray(review_units, dtype=int)
    group_audit = _combine_group_audit(canon, allocate_units, review_units, review_group_audit)

    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    supply = numeric_series(canon, "Supply").to_numpy(float)
    row_explain["predicted_final_alloc"] = final_units.astype(int)
    row_explain["predicted_final_supply"] = supply + final_units
    row_explain["allocated_flms"] = final_units / flm
    row_explain.loc[allocate_mask(canon) & (final_units <= 0), "decision_reason"] = "blank: direct Allocate neural regressor recommended zero"
    row_explain.loc[allocate_mask(canon) & (final_units > 0), "decision_reason"] = "allocated by direct Allocate neural regressor"
    row_explain.loc[review_mask(canon) & (final_units <= 0) & row_explain["decision_reason"].eq("pending Review iterative scorer"), "decision_reason"] = "blank: Review iterative scorer recommended zero"

    audit = canon.copy()
    audit["Predicted Final Alloc"] = final_units.astype(int)
    audit["Predicted Final Alloc Display"] = int_or_blank(final_units).values
    audit["Predicted Final Supply"] = supply + final_units
    audit = pd.concat([audit, signals.add_prefix("signal__")], axis=1)
    explain_cols = [c for c in row_explain.columns if c not in {"row_index"}]
    audit = pd.concat([audit, row_explain[explain_cols].add_prefix("allocator__")], axis=1)
    return canon, audit, row_explain, group_audit, cycle_trace


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def segment_metrics(canon: pd.DataFrame, audit: pd.DataFrame, actual_values) -> pd.DataFrame:
    pred = clean_num(audit["Predicted Final Alloc"])
    actual = clean_num(actual_values)
    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    dc = np.maximum(numeric_series(canon, "Dc Avail").to_numpy(float), 0.0)
    masks = {
        "all": np.ones(len(canon), dtype=bool),
        "allocate": allocate_mask(canon),
        "review": review_mask(canon),
        "nonzero_predictions": pred > 0,
        "blank_predictions": pred <= 0,
    }
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        p, a, f, d = pred[mask], actual[mask], flm[mask], dc[mask]
        err = p - a
        abs_err = np.abs(err)
        rows.append({
            "segment": name,
            "rows": int(mask.sum()),
            "mae_units": float(np.mean(abs_err)),
            "rmse_units": float(np.sqrt(np.mean(err ** 2))),
            "exact_rate": float(np.mean(p == a)),
            "within_1_flm_rate": float(np.mean(abs_err <= f)),
            "false_positives": int(((p > 0) & (a <= 0)).sum()),
            "false_negatives": int(((p <= 0) & (a > 0)).sum()),
            "pred_units": float(p.sum()),
            "actual_units": float(a.sum()),
            "unit_delta": float(p.sum() - a.sum()),
            "negative_violations": int((p < 0).sum()),
            "row_over_dc_violations": int((p > d + 1e-9).sum()),
        })
    return pd.DataFrame(rows)


def business_rule_metrics(canon: pd.DataFrame, audit: pd.DataFrame, group_audit: pd.DataFrame) -> pd.DataFrame:
    pred = clean_num(audit["Predicted Final Alloc"])
    supply = numeric_series(canon, "Supply").to_numpy(float)
    flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
    rec = numeric_series(canon, "Alloc. Rec.").to_numpy(float)
    d60m = numeric_series(canon, "D60").to_numpy(float) / 2.0
    proj = numeric_series(canon, "Proj. Demand").to_numpy(float)
    final_supply = supply + pred
    rows = [
        {"metric": "group_over_allocated_count", "value": int(group_audit.get("over_allocated", pd.Series(dtype=bool)).astype(bool).sum()) if not group_audit.empty else 0},
        {"metric": "partial_allocation_count", "value": int(((pred > 0) & (pred < flm)).sum())},
        {"metric": "over_rec_plus_1flm_count", "value": int((pred > rec + flm + 1e-9).sum())},
        {"metric": "final_supply_over_d60_plus_1flm_count", "value": int(((d60m > 0) & (final_supply > d60m + flm)).sum())},
        {"metric": "final_supply_over_proj_plus_1flm_count", "value": int(((proj > 0) & (final_supply > proj + flm)).sum())},
        {"metric": "zero_demand_positive_alloc_count", "value": int(((proj <= 0) & (d60m <= 0) & (pred > 0)).sum())},
        {"metric": "review_positive_alloc_count", "value": int((review_mask(canon) & (pred > 0)).sum())},
        {"metric": "allocate_positive_alloc_count", "value": int((allocate_mask(canon) & (pred > 0)).sum())},
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------------

def plot_bar(df, x, y, title, orientation="v", color=None, text=None, height=420):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return None
    fig = px.bar(df, x=x, y=y, color=color, text=text, orientation=orientation, title=title)
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=58, b=20), legend_title_text="")
    return fig


def plot_line(df, x, y, title, color=None, height=420):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return None
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=58, b=20), legend_title_text="")
    return fig


def show_plot(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")


# -----------------------------------------------------------------------------
# Feature explanation helpers
# -----------------------------------------------------------------------------

FEATURE_FAMILY_RULES = [
    ("hash__", "Categorical hash identity"),
    ("demand", "Demand signal"),
    ("velocity", "Velocity / trend"),
    ("shortage", "Shortage / need"),
    ("supply", "Supply / final supply"),
    ("rec", "Allocation recommendation"),
    ("proj", "Projected demand"),
    ("rank", "Rank / priority"),
    ("dc", "DC availability"),
    ("flm", "FLM / pack-size"),
    ("cost", "Cost / unit economics"),
    ("site", "Site/store context"),
    ("class_line", "Class-line peer context"),
    ("peer", "Peer comparison"),
    ("zero", "Sparse / zero demand"),
    ("ak", "AK marker only"),
    ("site802", "Site 802 marker only"),
]


def feature_family(name: str) -> str:
    low = str(name).lower()
    for token, fam in FEATURE_FAMILY_RULES:
        if token in low:
            return fam
    return "Base worksheet field"


def feature_description(name: str) -> str:
    low = str(name).lower()
    if name in APPROVED_COLUMNS:
        return "Original worksheet input used directly by the neural network."
    if low.startswith("hash__"):
        return "Hashed categorical representation that lets the NumPy model learn class, line, site, state, flag, rank text, and DC bucket identities without storing one column per category."
    if "weighted_velocity" in low:
        return "Blended demand velocity using L30, D30, D60/2, LW×4.29, TTM/12, and projected demand."
    if "sheet_need" in low:
        return "Demand-supported target need based on MIL, projected demand, recommendation, velocity, and historical demand signals."
    if "shortage" in low or "gap" in low:
        return "Measures how far current supply is below a demand, projection, recommendation, or target-supply signal."
    if "oversupply" in low or "overstock" in low:
        return "Risk feature estimating whether another FLM would leave the store over-supplied."
    if "rec" in low:
        return "Compares model need against the workbook Allocation Recommendation and the 1-FLM-above-recommendation cap."
    if "rank" in low:
        return "Priority/ranking signal used to choose which store should receive limited DC inventory first."
    if "dc" in low:
        return "Distribution-center inventory availability or a bucket describing available inventory pressure."
    if "peer" in low or "class_line" in low:
        return "Context feature comparing this row to similar rows within the same class-line or site group."
    if "zero" in low or "sparse" in low:
        return "Guardrail feature for rows with weak or sparse historical demand."
    return "Engineered feature derived only from approved worksheet columns."


def build_feature_catalog_table(bundle: Dict) -> pd.DataFrame:
    cfg = bundle["feature_config"]
    names = list(cfg.final_feature_names or [])
    df = pd.DataFrame({"feature": names})
    df["family"] = df["feature"].map(feature_family)
    df["description"] = df["feature"].map(feature_description)
    df["source"] = np.where(df["feature"].isin(APPROVED_COLUMNS), "Original worksheet column", np.where(df["feature"].str.startswith("hash__"), "Categorical hash", "Engineered numeric feature"))
    return df


# -----------------------------------------------------------------------------
# Model dependency / model-usage helpers
# -----------------------------------------------------------------------------

META_OUTPUT_MAP = {
    "meta_output_0": "classifier_probability",
    "meta_output_1": "rank_priority",
    "meta_output_2": "aux_cut_rec",
    "meta_output_3": "aux_follow_rec",
    "meta_output_4": "aux_one_flm",
    "meta_output_5": "aux_zero_out",
    "meta_output_6": "aux_below_flm_remainder",
}

UPSTREAM_SIGNAL_TO_SOURCE = {
    "shared_demand_score": "shared_demand",
    "shared_demand_gap_flm_before": "shared_demand",
    "shared_demand_gap_flm_after": "shared_demand",
    "target_final_supply_prediction": "shared_final_supply",
    "final_supply_gap_flm_before": "shared_final_supply",
    "final_supply_gap_flm_after": "shared_final_supply",
    "target_supply_gap_flm_before": "shared_final_supply",
    "target_supply_gap_flm_after": "shared_final_supply",
    "classifier_probability": "segment_classifier",
    "classifier_above_threshold": "segment_classifier",
    "rank_priority": "segment_ranker",
    "pred_flms_raw": "segment_regressor",
    "pred_remaining_flm_before": "segment_regressor",
    "pred_remaining_flm_after": "segment_regressor",
    "aux_cut_rec": "segment_auxiliary",
    "aux_follow_rec": "segment_auxiliary",
    "aux_one_flm": "segment_auxiliary",
    "aux_zero_out": "segment_auxiliary",
    "aux_below_flm_remainder": "segment_auxiliary",
}

DYNAMIC_STEP_WORDS = [
    "allocated_", "dc_remaining", "row_cap_remaining", "current_supply", "after_step",
    "gap_flm", "remaining_flm", "over_", "candidate_score_formula", "step_units",
    "step_flms", "cycle", "before", "after", "review_soft_excess",
]

BASE_STEP_FEATURES = {
    "segment_allocate", "segment_review", "flm", "mil", "supply", "alloc_rec",
    "proj_demand", "d60_month", "weighted_velocity", "demand_target",
}


def normalize_feature_importance(fi: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return a consistent view of the model_feature_dashboard_top100.csv export."""
    if fi is None or fi.empty:
        return pd.DataFrame(), {}
    out = fi.copy()
    cols = {
        "model": "model_name" if "model_name" in out.columns else ("model" if "model" in out.columns else out.columns[0]),
        "feature": "feature_name" if "feature_name" in out.columns else ("feature" if "feature" in out.columns else out.columns[0]),
        "family": "feature_family" if "feature_family" in out.columns else ("family" if "family" in out.columns else None),
        "rank": "feature_rank" if "feature_rank" in out.columns else None,
        "importance": "importance_percent" if "importance_percent" in out.columns else ("importance_normalized" if "importance_normalized" in out.columns else None),
    }
    if cols["importance"] is None:
        numeric_cols = [c for c in out.columns if pd.to_numeric(out[c], errors="coerce").notna().any()]
        cols["importance"] = numeric_cols[-1] if numeric_cols else out.columns[-1]
    out[cols["importance"]] = pd.to_numeric(out[cols["importance"]], errors="coerce").fillna(0.0)
    out[cols["model"]] = out[cols["model"]].astype(str)
    out[cols["feature"]] = out[cols["feature"]].astype(str)
    if cols["family"]:
        out[cols["family"]] = out[cols["family"]].astype(str)
    return out, cols


def model_signal_category(feature_name: str, model_name: str = "") -> str:
    """Classify a top feature by whether it comes from another model, a rule/state value, or base features."""
    f = str(feature_name)
    low = f.lower()
    if f in META_OUTPUT_MAP:
        return "Stacked neural output"
    if f in {"shared_demand_score", "shared_demand_gap_flm_before", "shared_demand_gap_flm_after"}:
        return "Shared demand output"
    if f in {"target_final_supply_prediction", "final_supply_gap_flm_before", "final_supply_gap_flm_after", "target_supply_gap_flm_before", "target_supply_gap_flm_after"}:
        return "Shared final-supply output"
    if f in {"classifier_probability", "rank_priority", "pred_flms_raw"} or low.startswith("aux_"):
        return "Segment neural output"
    if str(model_name) == "iterative_flm_step_scorer" and (f in BASE_STEP_FEATURES):
        return "Base worksheet / demand state"
    if str(model_name) == "iterative_flm_step_scorer" and any(word in low for word in DYNAMIC_STEP_WORDS):
        return "Iterative cycle state / rule context"
    if low.startswith("hash__"):
        return "Base categorical identity"
    return "Base worksheet / engineered feature"


def dependency_share_for_features(fi: pd.DataFrame, cols: Dict[str, str], model_name: str, feature_names: List[str]) -> Tuple[float, str]:
    if fi.empty or not feature_names:
        return 0.0, ""
    model_col, feature_col, imp_col = cols["model"], cols["feature"], cols["importance"]
    g = fi.loc[fi[model_col].astype(str).eq(model_name)].copy()
    if g.empty:
        return 0.0, ""
    denom = float(g[imp_col].sum()) or 1.0
    hit = g.loc[g[feature_col].isin(feature_names)].copy()
    share = float(hit[imp_col].sum()) / denom
    evidence = ", ".join(hit.sort_values(imp_col, ascending=False)[feature_col].head(5).astype(str).tolist())
    return share, evidence


def dependency_share_by_prefix(fi: pd.DataFrame, cols: Dict[str, str], model_name: str, prefixes: List[str]) -> Tuple[float, str]:
    if fi.empty or not prefixes:
        return 0.0, ""
    model_col, feature_col, imp_col = cols["model"], cols["feature"], cols["importance"]
    g = fi.loc[fi[model_col].astype(str).eq(model_name)].copy()
    if g.empty:
        return 0.0, ""
    denom = float(g[imp_col].sum()) or 1.0
    mask = pd.Series(False, index=g.index)
    for p in prefixes:
        mask = mask | g[feature_col].astype(str).str.startswith(p)
    hit = g.loc[mask].copy()
    share = float(hit[imp_col].sum()) / denom
    evidence = ", ".join(hit.sort_values(imp_col, ascending=False)[feature_col].head(5).astype(str).tolist())
    return share, evidence


def build_dependency_edges(fi: pd.DataFrame) -> pd.DataFrame:
    fi, cols = normalize_feature_importance(fi)
    rows = []

    def add(source, target, relationship, feature_names=None, prefixes=None, always_used=True, note=""):
        if prefixes:
            share, evidence = dependency_share_by_prefix(fi, cols, target, prefixes)
        else:
            share, evidence = dependency_share_for_features(fi, cols, target, feature_names or [])
        rows.append({
            "source_model": source,
            "downstream_model": target,
            "relationship": relationship,
            "top100_importance_share": share,
            "top100_importance_pct": share * 100.0,
            "evidence_features_in_downstream_top100": evidence or "Not in top-100; still wired as an input" if always_used else evidence,
            "design_dependency": bool(always_used),
            "note": note,
        })

    shared_targets = [
        "allocate_classifier", "allocate_ranker", "allocate_auxiliary", "allocate_regressor",
        "review_classifier", "review_ranker", "review_auxiliary", "review_regressor",
    ]
    for target in shared_targets:
        add("shared_demand", target, "Appended learned demand context", ["shared_demand_score"])
        add("shared_final_supply", target, "Appended learned final-supply context", ["target_final_supply_prediction"])

    # Regressors are stacked on top of their classifier, ranker, and auxiliary outputs.
    for segment in ["allocate", "review"]:
        reg = f"{segment}_regressor"
        add(f"{segment}_classifier", reg, "Meta feature into FLM sizing regressor", ["meta_output_0"], note="meta_output_0 = classifier_probability")
        add(f"{segment}_ranker", reg, "Meta feature into FLM sizing regressor", ["meta_output_1"], note="meta_output_1 = rank_priority")
        add(f"{segment}_auxiliary", reg, "Auxiliary behavior outputs into FLM sizing regressor", ["meta_output_2", "meta_output_3", "meta_output_4", "meta_output_5", "meta_output_6"], note="meta_output_2..6 = cut/follow/one-FLM/zero/remainder signals")

    # Final runtime routing. Allocate rows use the direct Allocate neural regressor.
    add("allocate_classifier", "final_allocate_direct_path", "Runtime classifier gate for Allocate rows", ["classifier_probability", "classifier_above_threshold"], note="Allocate rows do not use the iterative FLM scorer at runtime.")
    add("allocate_ranker", "final_allocate_direct_path", "Priority/repair signal for Allocate rows", ["rank_priority"], note="Used for visibility and DC repair priority around direct Allocate predictions.")
    add("allocate_regressor", "final_allocate_direct_path", "Direct FLM sizing output for Allocate rows", ["pred_flms_raw"], note="The app converts the Allocate regressor's predicted FLMs into Final Alloc units.")
    add("allocate_auxiliary", "final_allocate_direct_path", "Behavior signals available beside direct Allocate sizing", ["aux_cut_rec", "aux_follow_rec", "aux_one_flm", "aux_zero_out", "aux_below_flm_remainder"])

    # Review iterative step scorer consumes shared and Review-model outputs directly.
    add("shared_demand", "iterative_flm_step_scorer", "Direct marginal-step feature for Review rows", ["shared_demand_score", "shared_demand_gap_flm_before", "shared_demand_gap_flm_after"])
    add("shared_final_supply", "iterative_flm_step_scorer", "Direct marginal-step feature for Review rows", ["target_final_supply_prediction", "target_supply_gap_flm_before", "target_supply_gap_flm_after", "final_supply_gap_flm_before", "final_supply_gap_flm_after"])
    add("review_classifier", "iterative_flm_step_scorer", "Direct marginal-step neural signal", ["classifier_probability", "classifier_above_threshold"])
    add("review_ranker", "iterative_flm_step_scorer", "Direct marginal-step neural signal", ["rank_priority"])
    add("review_regressor", "iterative_flm_step_scorer", "Direct marginal-step neural signal", ["pred_flms_raw", "pred_remaining_flm_before", "pred_remaining_flm_after"])
    add("review_auxiliary", "iterative_flm_step_scorer", "Direct marginal-step neural signal", ["aux_cut_rec", "aux_follow_rec", "aux_one_flm", "aux_zero_out", "aux_below_flm_remainder"])

    out = pd.DataFrame(rows)
    if not out.empty:
        out["top100_importance_pct"] = out["top100_importance_pct"].round(2)
    return out


def build_model_signal_mix(fi: pd.DataFrame) -> pd.DataFrame:
    fi, cols = normalize_feature_importance(fi)
    if fi.empty:
        return pd.DataFrame()
    model_col, feature_col, imp_col = cols["model"], cols["feature"], cols["importance"]
    work = fi.copy()
    work["signal_source"] = [model_signal_category(f, m) for f, m in zip(work[feature_col], work[model_col])]
    grouped = work.groupby([model_col, "signal_source"], as_index=False)[imp_col].sum()
    totals = grouped.groupby(model_col)[imp_col].transform("sum").replace(0, np.nan)
    grouped["share_of_top100_importance"] = (grouped[imp_col] / totals).fillna(0.0)
    grouped["share_pct"] = grouped["share_of_top100_importance"] * 100.0
    grouped = grouped.rename(columns={model_col: "model", imp_col: "importance_sum"})
    return grouped.sort_values(["model", "share_pct"], ascending=[True, False])


def build_iterative_insight_table(fi: pd.DataFrame) -> pd.DataFrame:
    fi, cols = normalize_feature_importance(fi)
    if fi.empty:
        return pd.DataFrame()
    model_col, feature_col, imp_col = cols["model"], cols["feature"], cols["importance"]
    rank_col = cols.get("rank")
    step = fi.loc[fi[model_col].astype(str).eq("iterative_flm_step_scorer")].copy()
    if step.empty:
        return pd.DataFrame()
    step["signal_source"] = [model_signal_category(f, "iterative_flm_step_scorer") for f in step[feature_col]]
    step["source_detail"] = step[feature_col].map(lambda f: UPSTREAM_SIGNAL_TO_SOURCE.get(str(f), "optimizer_state_or_base_feature"))
    cols_out = [c for c in [rank_col, feature_col, "signal_source", "source_detail", cols.get("family"), imp_col, "importance_normalized", "cumulative_importance_percent"] if c and c in step.columns]
    return step[cols_out].sort_values(rank_col if rank_col else imp_col, ascending=True if rank_col else False)


def sankey_from_dependency_edges(edges: pd.DataFrame):
    if edges is None or edges.empty:
        return None
    show = edges.copy()
    show["value"] = pd.to_numeric(show["top100_importance_share"], errors="coerce").fillna(0.0)
    # Keep zero-evidence wired dependencies visible but much thinner.
    show["value"] = np.where(show["value"] > 0, show["value"] * 100.0, 1.0)
    labels = pd.Index(pd.concat([show["source_model"], show["downstream_model"]]).astype(str).unique())
    index = {name: i for i, name in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=list(labels), pad=18, thickness=15),
        link=dict(
            source=show["source_model"].astype(str).map(index),
            target=show["downstream_model"].astype(str).map(index),
            value=show["value"],
            customdata=show["relationship"].astype(str),
            hovertemplate="%{source.label} → %{target.label}<br>Weight: %{value:.2f}<br>%{customdata}<extra></extra>",
        ),
    )])
    fig.update_layout(title="Model-to-model usage map", height=620, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# -----------------------------------------------------------------------------
# Load app bundle
# -----------------------------------------------------------------------------
try:
    bundle = load_bundle()
except Exception as exc:
    st.error("The app could not load the model bundle from the repo root.")
    st.exception(exc)
    st.markdown("### Required files expected beside `app.py`")
    st.code("""registry.json
feature_config.json
iterative_flm_optimizer.json
shared_demand_model.npz
shared_final_supply_model.npz
allocate_classifier_model.npz
allocate_ranker_model.npz
allocate_auxiliary_model.npz
allocate_regressor_model.npz
review_classifier_model.npz
review_ranker_model.npz
review_auxiliary_model.npz
review_regressor_model.npz
iterative_flm_step_scorer_model.npz""")
    st.stop()

registry = bundle["registry"]
summary = bundle["model_summary"]
early = bundle["early_stopping"]
step_meta = bundle["step_cache_meta"]
tuning_output = bundle.get("tuning_output", {})
test_input_audit = bundle.get("test_input_audit", {})

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model controls")
    only_eligible = st.checkbox("Only process Allocate and Review rows", value=True)
    show_trace = st.checkbox("Show cycle trace after prediction", value=False)
    st.divider()
    st.caption("Loaded model")
    st.write("Version:", registry.get("version", "—"))
    st.write("Rows trained:", fmt_int(registry.get("rows", summary.get("rows", 0))))
    st.write("Base features:", fmt_int(registry.get("feature_count_base", summary.get("features_base", 0))))
    st.write("With shared features:", fmt_int(registry.get("feature_count_with_shared", summary.get("features_with_shared", 0))))
    st.write("Step features:", fmt_int(registry.get("step_scorer_feature_count", summary.get("step_scorer_features", 0))))

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
predict_tab, audit_tab, model_tab, dependency_tab, features_tab, optimizer_tab, test_tab, diagnostics_tab, files_tab = st.tabs([
    "Predict",
    "Audit",
    "Model overview",
    "Model usage map",
    "Feature intelligence",
    "Iterative FLM optimizer",
    "Test results",
    "Diagnostics",
    "Files",
])

# -----------------------------------------------------------------------------
# Predict tab
# -----------------------------------------------------------------------------
with predict_tab:
    st.subheader("Predict Final Alloc.")
    st.markdown("Upload a daily allocation workbook or CSV. The app will fill `Final Alloc.` for eligible Allocate/Review rows and leave zero predictions blank in the output file.")
    uploaded = st.file_uploader("Upload workbook or CSV", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="predict_upload")
    sheet_choice = None
    if uploaded and not uploaded.name.lower().endswith(".csv"):
        sheets = _sheet_names_from_upload(uploaded)
        if sheets:
            sheet_choice = st.selectbox("Sheet", ["Auto detect"] + sheets, index=0)
    if uploaded:
        try:
            raw = read_upload(uploaded, sheet_choice)
            cleaned = remove_repeated_headers_and_filter(raw, only_eligible)
            canon, audit, row_explain, group_audit, cycle_trace = predict_dataframe(cleaned, bundle)
            output = cleaned.copy().reset_index(drop=True)
            target_col = find_target_column(output) or TARGET_COLUMN
            output[target_col] = int_or_blank(audit["Predicted Final Alloc"]).values
            pred = clean_num(audit["Predicted Final Alloc"])
            st.success(f"Predicted {len(output):,} rows. Item-level DC groups processed: {len(group_audit):,}.")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Rows", fmt_int(len(output)))
            c2.metric("Nonzero rows", fmt_int((pred > 0).sum()))
            c3.metric("Predicted units", fmt_int(pred.sum()))
            c4.metric("Allocate direct units", fmt_int(group_audit.get("allocate_neural_units", pd.Series(dtype=float)).sum() if not group_audit.empty else 0))
            c5.metric("Review iterative units", fmt_int(group_audit.get("review_iterative_units", pd.Series(dtype=float)).sum() if not group_audit.empty else 0))
            c6.metric("Group overspends", fmt_int(group_audit.get("over_allocated", pd.Series(dtype=bool)).astype(bool).sum() if not group_audit.empty else 0))

            seg_units = pd.DataFrame({
                "segment": ["Allocate", "Review"],
                "predicted_units": [float(pred[allocate_mask(canon)].sum()), float(pred[review_mask(canon)].sum())],
                "nonzero_rows": [int(((pred > 0) & allocate_mask(canon)).sum()), int(((pred > 0) & review_mask(canon)).sum())],
            })
            col_a, col_b = st.columns(2)
            with col_a:
                show_plot(plot_bar(seg_units, "segment", "predicted_units", "Predicted units by segment", text="predicted_units"))
            with col_b:
                if not group_audit.empty:
                    top_groups = group_audit.sort_values("allocated_units", ascending=False).head(20)
                    show_plot(plot_bar(top_groups, "allocated_units", "allocation_group", "Top item groups by allocated units", orientation="h"))

            st.markdown("### Spot-check predictions")
            filter_choice = st.selectbox("Filter", ["All", "Nonzero allocations", "Allocate", "Review", "Potential errors / repairs", "Highest neural priority"])
            view = audit.copy()
            if filter_choice == "Nonzero allocations":
                view = view.loc[view["Predicted Final Alloc"] > 0]
            elif filter_choice == "Allocate":
                view = view.loc[allocate_mask(canon)]
            elif filter_choice == "Review":
                view = view.loc[review_mask(canon)]
            elif filter_choice == "Potential errors / repairs":
                view = view.loc[view.get("allocator__decision_reason", pd.Series(index=view.index, dtype=str)).astype(str).str.contains("repair|cap|below threshold", case=False, na=False)]
            elif filter_choice == "Highest neural priority":
                view = view.sort_values(["signal__rank_priority", "signal__classifier_probability"], ascending=False)

            display_cols = [c for c in [
                "Class Name", "Line Name", "Item", "Product ID", "Site", "State", "Flag", "MIL", "FLM", "L30", "D30", "D60", "LW", "TTM", "Supply", "Dc Avail", "Proj. Demand", "Alloc. Rec.",
                "Predicted Final Alloc", "Predicted Final Supply", "signal__classifier_probability", "signal__rank_priority", "signal__pred_flms_raw", "signal__shared_demand_score", "signal__target_final_supply_prediction", "allocator__final_decision_path", "allocator__decision_reason",
            ] if c in view.columns]
            display_view = recommendation_display_frame(view[display_cols])
            st.dataframe(display_view.head(1000), use_container_width=True)

            audit_download = recommendation_display_frame(audit)
            st.download_button("Download filled CSV", output.to_csv(index=False).encode("utf-8"), file_name="allocation_filled_output.csv", mime="text/csv", key="pred_download_output")
            st.download_button("Download prediction audit CSV", audit_download.to_csv(index=False).encode("utf-8"), file_name="allocation_prediction_audit.csv", mime="text/csv", key="pred_download_audit")
            st.download_button("Download item group audit CSV", group_audit.to_csv(index=False).encode("utf-8"), file_name="allocation_item_group_audit.csv", mime="text/csv", key="pred_download_group")
            if show_trace:
                st.markdown("### Iterative cycle trace")
                st.dataframe(cycle_trace.head(2500), use_container_width=True)
                st.download_button("Download cycle trace CSV", cycle_trace.to_csv(index=False).encode("utf-8"), file_name="allocation_iterative_cycle_trace.csv", mime="text/csv", key="pred_download_trace")
        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

# -----------------------------------------------------------------------------
# Audit tab
# -----------------------------------------------------------------------------
with audit_tab:
    st.subheader("Audit against existing Final Alloc.")
    st.markdown("Use this when a file already contains `Final Alloc.` values. The app predicts the same file and compares model output against the existing column.")
    uploaded_audit = st.file_uploader("Upload workbook or CSV for audit", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="audit_upload")
    sheet_audit = None
    if uploaded_audit and not uploaded_audit.name.lower().endswith(".csv"):
        sheets = _sheet_names_from_upload(uploaded_audit)
        if sheets:
            sheet_audit = st.selectbox("Audit sheet", ["Auto detect"] + sheets, index=0, key="audit_sheet")
    if uploaded_audit:
        try:
            raw = read_upload(uploaded_audit, sheet_audit)
            cleaned = remove_repeated_headers_and_filter(raw, only_eligible)
            target_col = find_target_column(cleaned)
            if target_col is None:
                st.warning("No Final Alloc. column was detected. Run Predict instead.")
            else:
                actual_values = cleaned[target_col]
                canon, audit, row_explain, group_audit, cycle_trace = predict_dataframe(cleaned, bundle)
                metrics = segment_metrics(canon, audit, actual_values)
                rules = business_rule_metrics(canon, audit, group_audit)
                pred = clean_num(audit["Predicted Final Alloc"])
                actual = clean_num(actual_values)
                flm = np.maximum(numeric_series(canon, "FLM").to_numpy(float), 1.0)
                all_row = metrics.loc[metrics["segment"].eq("all")].iloc[0]

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Rows", fmt_int(all_row["rows"]))
                c2.metric("Exact rate", fmt_pct(all_row["exact_rate"]))
                c3.metric("Within 1 FLM", fmt_pct(all_row["within_1_flm_rate"]))
                c4.metric("MAE units", fmt_num(all_row["mae_units"]))
                c5.metric("Unit delta", fmt_int(all_row["unit_delta"]))

                col1, col2 = st.columns(2)
                with col1:
                    plot_df = metrics[["segment", "exact_rate", "within_1_flm_rate"]].melt("segment", var_name="metric", value_name="rate")
                    fig = px.bar(plot_df, x="segment", y="rate", color="metric", barmode="group", title="Exact and within-1-FLM rates by segment")
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")
                with col2:
                    scatter_df = pd.DataFrame({"Actual Final Alloc": actual, "Predicted Final Alloc": pred, "FLM": flm})
                    fig = px.scatter(scatter_df.sample(min(len(scatter_df), 5000), random_state=42), x="Actual Final Alloc", y="Predicted Final Alloc", size="FLM", opacity=0.55, title="Predicted vs actual Final Alloc. sample")
                    maxv = max(float(scatter_df["Actual Final Alloc"].max()), float(scatter_df["Predicted Final Alloc"].max()), 1.0)
                    fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", name="Perfect match"))
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")

                st.markdown("### Segment metrics")
                st.dataframe(metrics, use_container_width=True)
                st.markdown("### Business-rule metrics")
                show_plot(plot_bar(rules.sort_values("value", ascending=True), "value", "metric", "Business rule checks", orientation="h"))
                st.dataframe(rules, use_container_width=True)

                row_audit = cleaned.copy().reset_index(drop=True)
                row_audit["Actual Final Alloc"] = np.rint(actual).astype(int)
                row_audit["Predicted Final Alloc"] = np.rint(pred).astype(int)
                row_audit["Absolute Error Units"] = np.abs(row_audit["Predicted Final Alloc"] - row_audit["Actual Final Alloc"])
                row_audit["Signed Error Units"] = row_audit["Predicted Final Alloc"] - row_audit["Actual Final Alloc"]
                row_audit["Within 1 FLM"] = row_audit["Absolute Error Units"].to_numpy(float) <= flm
                row_audit["False Positive"] = (row_audit["Predicted Final Alloc"] > 0) & (row_audit["Actual Final Alloc"] <= 0)
                row_audit["False Negative"] = (row_audit["Predicted Final Alloc"] <= 0) & (row_audit["Actual Final Alloc"] > 0)
                row_audit = pd.concat([row_audit, audit[[c for c in audit.columns if c.startswith("signal__") or c.startswith("allocator__")]].reset_index(drop=True)], axis=1)

                error_filter = st.selectbox("Error filter", ["Largest errors", "All errors", "False positives", "False negatives", "Outside 1 FLM", "All rows"])
                view = row_audit.copy()
                if error_filter == "Largest errors":
                    view = view.sort_values("Absolute Error Units", ascending=False)
                elif error_filter == "All errors":
                    view = view.loc[view["Absolute Error Units"] > 0]
                elif error_filter == "False positives":
                    view = view.loc[view["False Positive"]]
                elif error_filter == "False negatives":
                    view = view.loc[view["False Negative"]]
                elif error_filter == "Outside 1 FLM":
                    view = view.loc[~view["Within 1 FLM"]]
                st.dataframe(recommendation_display_frame(view).head(1000), use_container_width=True)
                row_audit_download = recommendation_display_frame(row_audit)
                st.download_button("Download audit metrics CSV", metrics.to_csv(index=False).encode("utf-8"), file_name="audit_metrics.csv", mime="text/csv", key="audit_metrics_download")
                st.download_button("Download row-level audit CSV", row_audit_download.to_csv(index=False).encode("utf-8"), file_name="row_level_audit.csv", mime="text/csv", key="audit_row_download")
        except Exception as exc:
            st.error("Audit failed.")
            st.exception(exc)

# -----------------------------------------------------------------------------
# Model overview tab
# -----------------------------------------------------------------------------
with model_tab:
    st.subheader("Model overview")
    st.markdown(
        """
        The current deployment is a **single-review, no-specialist neural allocation stack**. It uses shared context models first, routes rows into Allocate or Review neural stacks, then uses two final decision paths: Allocate rows are sized directly by the trained Allocate neural regressor, while Review rows are handled by the neural iterative FLM step scorer one FLM at a time. The final repair layer still protects item-level DC and the Alloc. Rec. + 1 FLM cap.
        """
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Training rows", fmt_int(registry.get("rows", summary.get("rows", 0))))
    c2.metric("Allocate rows", fmt_int(summary.get("allocate_rows", step_meta.get("allocate_rows", 0))))
    c3.metric("Review rows", fmt_int(summary.get("review_rows", step_meta.get("review_rows", 0))))
    c4.metric("Base features", fmt_int(registry.get("feature_count_base", summary.get("features_base", 0))))
    c5.metric("Step features", fmt_int(registry.get("step_scorer_feature_count", summary.get("step_scorer_features", 0))))

    st.markdown("### Decision flow")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#EEF2FF"];
          A [label="Uploaded workbook\nAllocate + Review rows"];
          B [label="Feature engineering\n914 base features"];
          C [label="Shared demand model"];
          D [label="Shared final-supply model"];
          E [label="Allocate neural stack\nclassifier + ranker + aux + sizer"];
          F [label="Review neural stack\nclassifier + ranker + aux + sizer"];
          G [label="Allocate direct neural regressor\nAllocate rows only"];
          H [label="Review neural FLM step scorer\nReview rows only"];
          I [label="Final Alloc.\nDC-safe mixed output"];
          A -> B; B -> C; B -> D; C -> E; C -> F; D -> E; D -> F; E -> G; F -> H; G -> I; H -> I;
        }
        """
    )

    manifest = bundle.get("part_manifest", {}).get("models", {})
    size_rows = []
    for k, fn in registry.get("models", {}).items():
        entry = manifest.get(fn, {})
        size_rows.append({"model": k, "file": fn, "size_mb": float(entry.get("size_mb", 0) or 0)})
    size_df = pd.DataFrame(size_rows)
    if not size_df.empty:
        show_plot(plot_bar(size_df.sort_values("size_mb", ascending=True), "size_mb", "model", "Model artifact sizes", orientation="h"))
        st.dataframe(size_df, use_container_width=True)

    if early:
        early_df = pd.DataFrame([{"model": k, **v} for k, v in early.items()])
        col1, col2 = st.columns(2)
        with col1:
            show_plot(plot_bar(early_df.sort_values("epochs_run", ascending=True), "epochs_run", "model", "Epochs run by model", orientation="h"))
        with col2:
            if "best_val_loss" in early_df.columns:
                show_plot(plot_bar(early_df.sort_values("best_val_loss", ascending=True), "best_val_loss", "model", "Best validation loss by model", orientation="h"))
        with st.expander("Early stopping details", expanded=False):
            st.dataframe(early_df, use_container_width=True)

    st.markdown("### Thresholds and optimizer controls")
    st.json({
        "allocate_threshold": registry.get("allocate_threshold"),
        "review_threshold": registry.get("review_threshold"),
        "min_allocate_neural_score": registry.get("min_allocate_neural_score"),
        "min_review_neural_score": registry.get("min_review_neural_score"),
        "min_partial_neural_score": registry.get("min_partial_neural_score"),
        "review_logic": registry.get("review_logic"),
        "specialists": registry.get("specialists"),
        "training_mode": registry.get("training_mode"),
    })

    st.markdown("### Full packaged model parameters")
    st.markdown("These files are included in the flat package so the Streamlit app can explain the model even when the GitHub repo only contains the `.npz` model weights.")
    with st.expander("registry.json — model map, thresholds, approved columns", expanded=False):
        st.json(registry)
    with st.expander("feature_config.json — approved columns, feature names, means/stds, hash dimensions", expanded=False):
        fc = bundle["feature_config"]
        st.json({
            "version": getattr(fc, "version", None),
            "approved_columns": getattr(fc, "approved_columns", []),
            "target_column": getattr(fc, "target_column", TARGET_COLUMN),
            "numeric_feature_names": getattr(fc, "numeric_feature_names", []),
            "extra_feature_count": len(getattr(fc, "extra_feature_names", []) or []),
            "final_feature_count": len(getattr(fc, "final_feature_names", []) or []),
            "hash_dims": getattr(fc, "hash_dims", {}),
        })
    with st.expander("iterative_flm_optimizer.json — hard rules and neural step thresholds", expanded=False):
        st.json(bundle["optimizer_config"])
    with st.expander("training_history.json — training metrics and model configuration", expanded=False):
        st.json(bundle.get("training_history", {}))
    with st.expander("tuning_output.json — tuned thresholds", expanded=False):
        st.json(tuning_output)


# -----------------------------------------------------------------------------
# Model usage / dependency tab
# -----------------------------------------------------------------------------
with dependency_tab:
    st.subheader("Model usage map: how the neural stack feeds itself")
    st.markdown(
        """
        This page answers two related questions: **which model outputs are reused downstream**, and **how the final mixed decision is split**. Allocate rows now use the direct Allocate neural regressor, while Review rows use the neural iterative FLM step scorer. The percentages below are based on the packaged `model_feature_dashboard_top100.csv` weight-path importance file, so they describe visible top-feature reliance rather than pretending to be exact causal attribution.
        """
    )

    feature_importance = read_csv_if_exists("model_feature_dashboard_top100.csv")
    dependency_edges = build_dependency_edges(feature_importance)
    signal_mix = build_model_signal_mix(feature_importance)
    iterative_table = build_iterative_insight_table(feature_importance)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models in registry", fmt_int(len(registry.get("models", {}))))
    c2.metric("Dependency edges shown", fmt_int(len(dependency_edges)))
    c3.metric("Shared outputs", "2")
    c4.metric("Iterative step inputs", fmt_int(len(STEP_FEATURE_NAMES)))

    if not dependency_edges.empty:
        fig = sankey_from_dependency_edges(dependency_edges)
        show_plot(fig)
        st.markdown("### Model-to-model dependency table")
        display_edges = dependency_edges.copy()
        display_edges["top100_importance_pct"] = display_edges["top100_importance_pct"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_edges, use_container_width=True)

        st.markdown("### Shared-model usage by downstream model")
        shared = dependency_edges.loc[dependency_edges["source_model"].isin(["shared_demand", "shared_final_supply"])].copy()
        if not shared.empty:
            shared_chart = shared.copy()
            shared_chart["top100_importance_pct"] = pd.to_numeric(shared_chart["top100_importance_pct"], errors="coerce").fillna(0.0)
            fig = px.bar(
                shared_chart,
                x="downstream_model",
                y="top100_importance_pct",
                color="source_model",
                barmode="group",
                title="Direct shared-output presence in each downstream model's top features",
                labels={"top100_importance_pct": "Share of downstream top-feature importance (%)", "downstream_model": "Downstream model"},
            )
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=120), xaxis_tickangle=-35)
            show_plot(fig)
            st.caption("A zero bar does not mean the shared model is absent. It means that specific shared output was not strong enough to appear in that downstream model's top-100 feature list. The shared outputs are still appended into the downstream feature matrix by design.")

    if not signal_mix.empty:
        st.markdown("### Source mix inside each model's top features")
        signal_chart = signal_mix.copy()
        fig = px.bar(
            signal_chart,
            x="model",
            y="share_pct",
            color="signal_source",
            title="Where each model's visible top-feature strength comes from",
            labels={"share_pct": "Share of top-feature importance (%)", "signal_source": "Signal source"},
        )
        fig.update_layout(height=560, margin=dict(l=20, r=20, t=60, b=130), xaxis_tickangle=-35)
        show_plot(fig)
        st.dataframe(signal_mix, use_container_width=True)

    st.markdown("### How the Review iterative model gets its insight")
    st.markdown(
        """
        The Review-side `iterative_flm_step_scorer` is intentionally a **meta-decision model**. It does not only look at raw worksheet demand. At runtime it is applied to Review rows and sees the shared demand estimate, shared final-supply estimate, Review classifier probability, Review rank priority, Review raw FLM sizing estimate, Review auxiliary behavior outputs, and the live state of the item cycle after each FLM is placed.

        In practical terms:

        - **Shared models** contribute broad demand and target-final-supply context.
        - **Review classifiers** indicate whether a Review row deserves to compete for units.
        - **Review rankers** provide store priority within an item group.
        - **Review regressors** estimate how many FLMs the Review row wants before DC constraints.
        - **Auxiliary heads** describe behavior such as cut recommendation, follow recommendation, one-FLM, zero-out, or below-FLM remainder.
        - **Cycle-state features** tell the scorer what has already been allocated, how much DC remains, whether another FLM would exceed recommendation room, and whether supply would become risky after the next step.
        """
    )

    if not iterative_table.empty:
        source_mix = iterative_table.copy()
        imp_col = "importance_percent" if "importance_percent" in source_mix.columns else None
        if imp_col:
            mix = source_mix.groupby("signal_source", as_index=False)[imp_col].sum()
            total = float(mix[imp_col].sum()) or 1.0
            mix["share_pct"] = mix[imp_col] / total * 100.0
            show_plot(plot_bar(mix.sort_values("share_pct", ascending=True), "share_pct", "signal_source", "Iterative step-scorer insight sources", orientation="h"))
        st.dataframe(iterative_table, use_container_width=True)

    st.markdown("### Architecture facts")
    arch_rows = [
        {"layer": "Base feature matrix", "output_used_by": "Shared models + Allocate/Review stacks", "details": f"{registry.get('feature_count_base', summary.get('features_base', '—'))} base features"},
        {"layer": "shared_demand", "output_used_by": "Allocate stack, Review stack, iterative step scorer", "details": "Adds learned demand score to downstream models."},
        {"layer": "shared_final_supply", "output_used_by": "Allocate stack, Review stack, iterative step scorer", "details": "Adds learned target final-supply estimate."},
        {"layer": "Allocate classifier/ranker/aux/regressor", "output_used_by": "Final Allocate direct path", "details": "Produces the direct Allocate row decision and raw FLM size. The app converts this neural FLM prediction into Allocate Final Alloc units."},
        {"layer": "Review classifier/ranker/aux/regressor", "output_used_by": "Review iterative step scorer", "details": "Produces Review probability, priority, behavior signals, and raw FLM size for the marginal step scorer."},
        {"layer": "iterative_flm_step_scorer", "output_used_by": "Final Review path", "details": f"{registry.get('step_scorer_feature_count', len(STEP_FEATURE_NAMES))} marginal-step features; applied to Review rows one FLM at a time."},
        {"layer": "Hard optimizer repair", "output_used_by": "Final output", "details": "Protects DC inventory, excludes non-eligible rows, and caps Final Alloc. at Alloc. Rec. + 1 FLM."},
    ]
    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True)

    if not dependency_edges.empty:
        st.download_button(
            "Download model dependency table",
            dependency_edges.to_csv(index=False).encode("utf-8"),
            file_name="model_dependency_usage_map.csv",
            mime="text/csv",
            key="download_model_dependency_usage_map",
        )

# -----------------------------------------------------------------------------
# Feature intelligence tab
# -----------------------------------------------------------------------------
with features_tab:
    st.subheader("Feature intelligence")
    feature_table = build_feature_catalog_table(bundle)
    feature_importance = read_csv_if_exists("model_feature_dashboard_top100.csv")
    feature_catalog_file = read_csv_if_exists("feature_catalog.csv")

    st.markdown(
        """
        The model only uses fields derived from the uploaded allocation worksheet. The base feature matrix contains original worksheet columns, engineered demand/supply/ranking signals, group-context features, and categorical hash features. The shared demand and final-supply models add two learned context signals before the Allocate and Review stacks run.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final feature count", fmt_int(len(feature_table)))
    c2.metric("Original worksheet inputs", fmt_int(len(APPROVED_COLUMNS)))
    c3.metric("Feature families", fmt_int(feature_table["family"].nunique()))
    c4.metric("Step-scorer features", fmt_int(len(STEP_FEATURE_NAMES)))

    fam_counts = feature_table["family"].value_counts().reset_index()
    fam_counts.columns = ["family", "count"]
    col1, col2 = st.columns(2)
    with col1:
        show_plot(plot_bar(fam_counts.sort_values("count", ascending=True), "count", "family", "Feature count by family", orientation="h"))
    with col2:
        source_counts = feature_table["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig = px.pie(source_counts, names="source", values="count", title="Feature source mix")
        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")

    st.markdown("### Feature families and what they mean")
    st.markdown(
        """
        | Feature family | How it helps allocation decisions |
        |---|---|
        | **Demand signal** | Blends recent, medium-term, weekly, trailing-twelve-month, and projected demand to avoid overreacting to one noisy column. |
        | **Supply / final supply** | Measures current inventory and projected final supply after each possible allocation. |
        | **Shortage / need** | Converts demand gaps into units and FLMs so the model learns if one more pack is meaningful. |
        | **Allocation recommendation** | Tells the model when to follow, reduce, blank, or cautiously exceed `Alloc. Rec.`; the optimizer caps output at `Alloc. Rec. + 1 FLM`. |
        | **Rank / priority** | Helps the model choose which stores win inventory when an item has limited DC availability. |
        | **Class-line peer context** | Compares each row against similar rows competing in the same class/line and item pool. |
        | **Categorical hash identity** | Encodes `Class Name`, `Line Name`, `Site`, `State`, `Flag`, rank text, and DC buckets in a NumPy-only deployable format. |
        | **Sparse / zero demand guardrails** | Helps suppress allocations where demand support is weak or isolated. |
        """
    )

    if not feature_importance.empty:
        st.markdown("### Top 100 features by model")
        st.markdown(
            "This table comes from the packaged neural weight-path importance report. "
            "Each model has its own top-100 feature list, so the shared demand model, final-supply model, Allocate stack, Review stack, and iterative step scorer can be inspected separately."
        )
        # Normalize expected column names from the feature dashboard export.
        fi = feature_importance.copy()
        model_col = "model_name" if "model_name" in fi.columns else ("model" if "model" in fi.columns else fi.columns[0])
        feature_col = "feature_name" if "feature_name" in fi.columns else ("feature" if "feature" in fi.columns else fi.columns[0])
        family_col = "feature_family" if "feature_family" in fi.columns else ("family" if "family" in fi.columns else None)
        rank_col = "feature_rank" if "feature_rank" in fi.columns else None
        importance_col = "importance_percent" if "importance_percent" in fi.columns else ("importance_normalized" if "importance_normalized" in fi.columns else None)
        if importance_col is None:
            numeric_cols = [c for c in fi.columns if pd.to_numeric(fi[c], errors="coerce").notna().any()]
            importance_col = numeric_cols[-1] if numeric_cols else fi.columns[-1]
        fi[importance_col] = pd.to_numeric(fi[importance_col], errors="coerce").fillna(0.0)

        model_options = sorted(fi[model_col].dropna().astype(str).unique().tolist())
        selected_model = st.selectbox("Select model to inspect", model_options, key="feature_top100_model_select") if model_options else None
        if selected_model:
            model_view = fi.loc[fi[model_col].astype(str).eq(selected_model)].copy()
            if rank_col and rank_col in model_view.columns:
                model_view = model_view.sort_values(rank_col)
            else:
                model_view = model_view.sort_values(importance_col, ascending=False)
            chart_view = model_view.head(30).copy()
            title = f"Top 30 features for {selected_model}"
            show_plot(plot_bar(chart_view.sort_values(importance_col, ascending=True), importance_col, feature_col, title, orientation="h"))
            if family_col and family_col in model_view.columns:
                fam = model_view[family_col].fillna("Unknown").value_counts().reset_index()
                fam.columns = ["family", "count"]
                show_plot(plot_bar(fam.sort_values("count", ascending=True), "count", "family", f"Top-100 feature families: {selected_model}", orientation="h"))
            show_cols = [c for c in [rank_col, model_col, "model_role", "model_task", "model_input_dim", "model_output_dim", "hidden_layers", feature_col, family_col, "importance_raw", "importance_normalized", "importance_percent", "cumulative_importance_percent", "importance_method"] if c and c in model_view.columns]
            st.dataframe(model_view[show_cols], use_container_width=True)

        st.markdown("### Cross-model feature coverage")
        summary_rows = []
        for model_name, g in fi.groupby(model_col):
            row = {"model": model_name, "top_features_reported": int(len(g))}
            if family_col and family_col in g.columns:
                row["feature_families"] = int(g[family_col].nunique())
                row["top_family"] = str(g[family_col].fillna("Unknown").value_counts().index[0]) if len(g) else "—"
            if "model_input_dim" in g.columns:
                row["input_dim"] = int(pd.to_numeric(g["model_input_dim"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(g["model_input_dim"], errors="coerce").notna().any() else None
            summary_rows.append(row)
        fi_summary = pd.DataFrame(summary_rows)
        if not fi_summary.empty:
            st.dataframe(fi_summary, use_container_width=True)
            show_plot(plot_bar(fi_summary.sort_values("top_features_reported", ascending=True), "top_features_reported", "model", "Top-feature rows packaged by model", orientation="h"))
        st.download_button("Download top-100 feature dashboard", fi.to_csv(index=False).encode("utf-8"), file_name="model_feature_dashboard_top100.csv", mime="text/csv", key="feature_top100_download")
    elif not feature_catalog_file.empty:
        st.markdown("### Packaged feature catalog")
        st.dataframe(feature_catalog_file.head(500), use_container_width=True)

    st.markdown("### Search every deployed feature")
    q = st.text_input("Search feature names or descriptions", "")
    view = feature_table.copy()
    if q.strip():
        m = view.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)
        view = view.loc[m]
    st.dataframe(view.head(1000), use_container_width=True)
    st.download_button("Download deployed feature catalog", feature_table.to_csv(index=False).encode("utf-8"), file_name="deployed_feature_catalog.csv", mime="text/csv", key="feature_catalog_download")

# -----------------------------------------------------------------------------
# Optimizer tab
# -----------------------------------------------------------------------------
with optimizer_tab:
    st.subheader("Review iterative FLM optimizer + Allocate direct neural path")
    st.markdown(
        """
        The final allocation layer now uses a mixed path. **Allocate rows use the direct Allocate neural regressor** trained on Allocate rows. **Review rows use the neural iterative FLM step scorer**, which cycles through Review candidates one FLM at a time using the DC left after direct Allocate decisions. After each Review FLM, current supply, remaining demand gap, recommendation room, and remaining DC are updated before the next cycle.
        """
    )
    opt = bundle["optimizer_config"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Neural scorer", "On" if opt.get("use_neural_step_scorer", True) else "Off")
    c2.metric("Allocate classifier gate", fmt_num(opt.get("allocate_threshold", 0), 3))
    c3.metric("Review step threshold", fmt_num(opt.get("min_review_neural_score", 0), 3))
    c4.metric("Max above rec", f"{fmt_num(opt.get('max_above_rec_flm', 1), 1)} FLM")

    step_features = pd.DataFrame({"feature": STEP_FEATURE_NAMES})
    step_features["family"] = step_features["feature"].map(feature_family)
    step_counts = step_features["family"].value_counts().reset_index()
    step_counts.columns = ["family", "count"]
    col1, col2 = st.columns(2)
    with col1:
        show_plot(plot_bar(step_counts.sort_values("count", ascending=True), "count", "family", "Step-scorer feature families", orientation="h"))
    with col2:
        if step_meta:
            meta_rows = pd.DataFrame([
                {"metric": "source rows", "value": step_meta.get("row_count", 0)},
                {"metric": "allocate rows", "value": step_meta.get("allocate_rows", 0)},
                {"metric": "review rows", "value": step_meta.get("review_rows", 0)},
                {"metric": "step examples", "value": step_meta.get("step_examples", 0)},
                {"metric": "positive step examples", "value": step_meta.get("positive_step_examples", 0)},
            ])
            show_plot(plot_bar(meta_rows, "metric", "value", "Step-scorer cache composition", text="value"))

    st.markdown("### Step-scorer feature explanations")
    step_features["description"] = step_features["feature"].map(feature_description)
    st.dataframe(step_features, use_container_width=True)

    detail = read_csv_if_exists("iterative_step_scorer_training_detail.csv")
    if not detail.empty:
        st.markdown("### Step-scorer training examples")
        st.dataframe(detail.head(1000), use_container_width=True)
        if "label_should_get_next_flm" in detail.columns:
            label_counts = detail["label_should_get_next_flm"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]
            show_plot(plot_bar(label_counts, "label", "count", "Positive vs negative marginal-FLM examples", text="count"))

    st.markdown("### Optimizer configuration")
    st.json(opt)


# -----------------------------------------------------------------------------
# Test results tab
# -----------------------------------------------------------------------------
with test_tab:
    st.subheader("Packaged test results")
    st.markdown(
        "These are the held-out/test outputs generated from the trained iterative model. "
        "The test path confirms the app uses only Allocate and Review rows, loads the iterative step scorer, and does not rebuild the old million-row step-scorer training set during testing."
    )
    overall = read_csv_if_exists("overall_segment_metrics.csv")
    business = read_csv_if_exists("business_rule_metrics.csv")
    component = read_csv_if_exists("component_model_metrics.csv")
    grouped = read_csv_if_exists("grouped_metrics_all.csv")
    group_audit_static = read_csv_if_exists("iterative_group_audit.csv")
    cycle_static = read_csv_if_exists("iterative_cycle_trace.csv")
    largest = read_csv_if_exists("largest_errors_top500.csv")
    prediction_detail = read_csv_if_exists("prediction_detail.csv")
    report_path = ART / "TEST_REPORT.md"

    if test_input_audit:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows scored", fmt_int(test_input_audit.get("rows_to_score", 0)))
        c2.metric("Allocate rows", fmt_int(test_input_audit.get("allocate_rows", 0)))
        c3.metric("Review rows", fmt_int(test_input_audit.get("review_rows", 0)))
        c4.metric("Cycle rows", fmt_int(test_input_audit.get("cycle_trace_rows_recorded", 0)))
        c5.metric("Step scorer loaded", "Yes" if test_input_audit.get("iterative_step_scorer_loaded") else "No")
        with st.expander("Full test input audit", expanded=False):
            st.json(test_input_audit)

    if not overall.empty:
        all_row = overall.loc[overall["segment"].astype(str).str.lower().eq("all")].head(1)
        if not all_row.empty:
            r = all_row.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Overall exact", fmt_pct(r.get("exact_rate")))
            c2.metric("Within 1 FLM", fmt_pct(r.get("within_1_flm_rate")))
            c3.metric("MAE units", fmt_num(r.get("mae_units")))
            c4.metric("Pred units", fmt_int(r.get("pred_units")))
            c5.metric("Unit delta", fmt_int(r.get("unit_delta")))
        plot_df = overall[[c for c in ["segment", "exact_rate", "within_1_flm_rate"] if c in overall.columns]].copy()
        if {"segment", "exact_rate", "within_1_flm_rate"}.issubset(plot_df.columns):
            melted = plot_df.melt("segment", var_name="metric", value_name="rate")
            fig = px.bar(melted, x="segment", y="rate", color="metric", barmode="group", title="Test accuracy by segment")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")
        st.dataframe(overall, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if not business.empty:
            st.markdown("### Business rule results")
            if {"metric", "value"}.issubset(business.columns):
                show_plot(plot_bar(business.sort_values("value", ascending=True), "value", "metric", "Business rule metrics", orientation="h"))
            st.dataframe(business, use_container_width=True)
    with col2:
        if not component.empty:
            st.markdown("### Component model test metrics")
            metric_options = [c for c in ["accuracy", "precision", "recall", "f1", "mae", "rmse", "rank_corr", "brier"] if c in component.columns and pd.to_numeric(component[c], errors="coerce").notna().any()]
            metric = st.selectbox("Component metric to chart", metric_options, key="test_component_metric") if metric_options else None
            if metric and "component" in component.columns:
                tmp = component.copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0)
                show_plot(plot_bar(tmp.sort_values(metric, ascending=True), metric, "component", f"Component model metric: {metric}", orientation="h"))
            st.dataframe(component, use_container_width=True)

    if not grouped.empty:
        st.markdown("### Grouped test metrics")
        if {"group_col", "group_value", "mae_units"}.issubset(grouped.columns):
            group_col = st.selectbox("Group metric split", sorted(grouped["group_col"].dropna().astype(str).unique().tolist()), key="test_group_col")
            tmp = grouped.loc[grouped["group_col"].astype(str).eq(group_col)].copy()
            tmp["mae_units"] = pd.to_numeric(tmp["mae_units"], errors="coerce").fillna(0)
            show_plot(plot_bar(tmp.sort_values("mae_units", ascending=True).tail(25), "mae_units", "group_value", f"Worst 25 {group_col} groups by MAE", orientation="h"))
        st.dataframe(grouped.head(1500), use_container_width=True)

    if not group_audit_static.empty:
        st.markdown("### Iterative item group audit")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Groups", fmt_int(len(group_audit_static)))
        if "over_allocated" in group_audit_static.columns:
            c2.metric("Over-allocated groups", fmt_int(group_audit_static["over_allocated"].astype(bool).sum()))
        if "allocated_units" in group_audit_static.columns:
            c3.metric("Allocated units", fmt_int(pd.to_numeric(group_audit_static["allocated_units"], errors="coerce").sum()))
        if "cycles_run" in group_audit_static.columns:
            c4.metric("Cycles", fmt_int(pd.to_numeric(group_audit_static["cycles_run"], errors="coerce").sum()))
        st.dataframe(group_audit_static.head(1000), use_container_width=True)

    if not cycle_static.empty:
        st.markdown("### Iterative cycle trace")
        if {"model_segment", "cycle_score"}.issubset(cycle_static.columns):
            tmp = cycle_static.copy()
            tmp["cycle_score"] = pd.to_numeric(tmp["cycle_score"], errors="coerce")
            fig = px.histogram(tmp.dropna(subset=["cycle_score"]).sample(min(len(tmp), 20000), random_state=42), x="cycle_score", color="model_segment", nbins=40, title="Recorded neural step-score distribution")
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}")
        st.dataframe(cycle_static.head(1500), use_container_width=True)

    if not largest.empty:
        st.markdown("### Largest errors")
        st.dataframe(largest.head(500), use_container_width=True)

    if not prediction_detail.empty:
        st.markdown("### Prediction detail sample")
        cols = [c for c in ["Class Name", "Line Name", "Item", "Site", "State", "Flag", "Supply", "Dc Avail", "Proj. Demand", "Alloc. Rec.", "Final Alloc.", "Predicted Final Alloc", "predicted_final_alloc", "classifier_probability", "rank_priority", "pred_flms_raw", "shared_demand_score", "target_final_supply_prediction", "decision_reason"] if c in prediction_detail.columns]
        st.dataframe(prediction_detail[cols].head(1000) if cols else prediction_detail.head(1000), use_container_width=True)

    if report_path.exists():
        with st.expander("Full markdown test report", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8", errors="ignore"))

    downloads = []
    for name in ["overall_segment_metrics.csv", "business_rule_metrics.csv", "component_model_metrics.csv", "grouped_metrics_all.csv", "iterative_group_audit.csv", "iterative_cycle_trace.csv", "largest_errors_top500.csv", "prediction_detail.csv", "TEST_REPORT.md"]:
        fp = ART / name
        if fp.exists():
            downloads.append(name)
    if downloads:
        with zipfile.ZipFile(io.BytesIO(), "w") as _:
            pass
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in downloads:
                z.write(ART / name, arcname=name)
        st.download_button("Download packaged test results", buf.getvalue(), file_name="allocation_test_results_packaged.zip", mime="application/zip", key="download_packaged_test_results")

# -----------------------------------------------------------------------------
# Diagnostics tab
# -----------------------------------------------------------------------------
with diagnostics_tab:
    st.subheader("Packaged diagnostics")
    st.markdown("These reports are loaded when their CSV files are present in the repo root. They are not required for prediction, but they make the Streamlit site much more useful for model review.")
    business = read_csv_if_exists("business_rule_metrics.csv")
    component = read_csv_if_exists("component_model_metrics.csv")
    grouped = read_csv_if_exists("grouped_metrics_all.csv")
    largest = read_csv_if_exists("largest_errors_top500.csv")
    group_audit_static = read_csv_if_exists("iterative_group_audit.csv")
    cycle_static = read_csv_if_exists("iterative_cycle_trace.csv")

    if not business.empty:
        st.markdown("### Business rule metrics")
        if "metric" in business.columns and "value" in business.columns:
            show_plot(plot_bar(business.sort_values("value", ascending=True), "value", "metric", "Business rule metrics", orientation="h"))
        st.dataframe(business, use_container_width=True)

    if not component.empty:
        st.markdown("### Component model metrics")
        st.dataframe(component, use_container_width=True)
        numeric_candidates = [c for c in component.columns if c not in {"component"}]
        numeric_candidates = [c for c in numeric_candidates if pd.to_numeric(component[c], errors="coerce").notna().any()]
        if numeric_candidates and "component" in component.columns:
            metric = st.selectbox("Component metric", numeric_candidates)
            tmp = component.copy()
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce").fillna(0)
            show_plot(plot_bar(tmp.sort_values(metric, ascending=True), metric, "component", f"Component comparison: {metric}", orientation="h"))

    if not grouped.empty:
        st.markdown("### Grouped metrics")
        if {"group_col", "group_value", "mae_units"}.issubset(grouped.columns):
            group_col = st.selectbox("Group column", sorted(grouped["group_col"].dropna().unique().tolist()))
            tmp = grouped.loc[grouped["group_col"].eq(group_col)].copy()
            tmp["mae_units"] = pd.to_numeric(tmp["mae_units"], errors="coerce").fillna(0)
            show_plot(plot_bar(tmp.sort_values("mae_units", ascending=True).tail(25), "mae_units", "group_value", f"Worst groups by MAE: {group_col}", orientation="h"))
        st.dataframe(grouped.head(1000), use_container_width=True)

    if not group_audit_static.empty:
        st.markdown("### Iterative group audit")
        c1, c2, c3 = st.columns(3)
        c1.metric("Groups audited", fmt_int(len(group_audit_static)))
        if "over_allocated" in group_audit_static.columns:
            c2.metric("Over-allocated groups", fmt_int(group_audit_static["over_allocated"].astype(bool).sum()))
        if "allocated_units" in group_audit_static.columns:
            c3.metric("Allocated units", fmt_int(pd.to_numeric(group_audit_static["allocated_units"], errors="coerce").sum()))
        if {"allocated_units", "allocation_group"}.issubset(group_audit_static.columns):
            tmp = group_audit_static.copy()
            tmp["allocated_units"] = pd.to_numeric(tmp["allocated_units"], errors="coerce").fillna(0)
            show_plot(plot_bar(tmp.sort_values("allocated_units").tail(25), "allocated_units", "allocation_group", "Top groups by iterative allocated units", orientation="h"))
        st.dataframe(group_audit_static.head(1000), use_container_width=True)

    if not largest.empty:
        st.markdown("### Largest errors")
        st.dataframe(largest.head(500), use_container_width=True)

    if not cycle_static.empty:
        st.markdown("### Packaged cycle trace sample")
        st.dataframe(cycle_static.head(1000), use_container_width=True)

# -----------------------------------------------------------------------------
# Files tab
# -----------------------------------------------------------------------------
with files_tab:
    st.subheader("Files in app folder")
    rows = []
    for p in sorted(APP_DIR.iterdir()):
        if p.is_file():
            rows.append({"file": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 3)})
    files_df = pd.DataFrame(rows)
    st.dataframe(files_df, use_container_width=True)

    st.markdown("### Expected minimum deployment files")
    st.code("""app.py
requirements.txt
allocation_feature_engineering.py
allocation_nn_core.py
allocation_iterative_flm_optimizer.py
registry.json
feature_config.json
iterative_flm_optimizer.json
shared_demand_model.npz
shared_final_supply_model.npz
allocate_classifier_model.npz
allocate_ranker_model.npz
allocate_auxiliary_model.npz
allocate_regressor_model.npz
review_classifier_model.npz
review_ranker_model.npz
review_auxiliary_model.npz
review_regressor_model.npz
iterative_flm_step_scorer_model.npz""")

    st.markdown("### Packaged metadata and diagnostics included in this zip")
    st.markdown(
        "The flat package also includes `registry.json`, `feature_config.json`, `iterative_flm_optimizer.json`, `model_summary.json`, "
        "`training_history.json`, `tuning_output.json`, `early_stopping_summary.json`, `model_feature_dashboard_top100.csv`, and the packaged test-result CSV/Markdown files. "
        "Those files let the app explain model parameters, top features, and test results even if your GitHub repo only had the `.npz` weights before adding this package."
    )
