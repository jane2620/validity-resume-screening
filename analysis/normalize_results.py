#!/usr/bin/env python3
"""
Normalizes and outputs full DataFrames/CSVs.

Example usage:
    python analysis/normalize_results.py --inputs evaluations/baseline --output_dir analysis/normalized_data/synthetic/baseline
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Normalize evaluation CSVs into single DataFrame/CSV")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=["evaluations/baseline"],
        help="One or more CSV files or directories containing eval CSVs",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="analysis/normalized_data",
        help="Directory to write normalized CSV",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default="all_results.csv",
        help="Output CSV filename",
    )
    return p.parse_args()


def _gather_csv_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
    return files


def _infer_framework_from_path(path: Path) -> str:
    s = str(path).lower()
    if "/baseline_noabstain/" in s:
        return "baseline_noabstain"
    if "/baseline/" in s:
        return "baseline"
    return "unknown"


def _infer_model_from_filename(path: Path) -> str:
    stem = path.stem
    if "_paired_resume_decisions_" in stem:
        return stem.split("_paired_resume_decisions_", 1)[0]
    return stem


def _infer_constructed_by(path: Path) -> str:
    s = path.name.lower()
    if "_claude_" in s:
        return "claude"
    if "_gemini_" in s:
        return "gemini"
    if "_rel6_shard" in s or "_v_k_shard" in s:
        return "gemini"
    return "unknown"


def _coerce_demographics(row: Dict[str, Any]) -> tuple[str, str]:
    db = str(row.get("demographic_base") or "").strip()
    dv = str(row.get("demographic_variant") or "").strip()

    if "B_" in db or "B_" in dv:
        print(db, end=" ")
        print(dv)
        print("--------------------------------")

    if db and dv:
        return db, dv
    dem = row.get("demographics")
    if isinstance(dem, (list, tuple)) and len(dem) >= 2:
        return str(dem[0]), str(dem[1])
    if isinstance(dem, str):
        try:
            arr = json.loads(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
        try:
            arr = ast.literal_eval(dem)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return str(arr[0]), str(arr[1])
        except Exception:
            pass
    return "", ""


def _coerce_diff_quals(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
            try:
                j = ast.literal_eval(s)
                if isinstance(j, list):
                    return [str(x).strip() for x in j if str(x).strip()]
            except Exception:
                pass
    return []

def _compute_diff_length(base: str, variant: str) -> int:
    """Compute the total length of the unified diff between base and variant resumes."""
    base_lines = str(base).splitlines(keepends=True)
    variant_lines = str(variant).splitlines(keepends=True)
    diff = difflib.unified_diff(
        base_lines,
        variant_lines,
        fromfile="Base Resume",
        tofile="Variant Resume",
        lineterm="",
        n=0  # No context lines needed for length calculation
    )
    total_length = sum(len(line) for line in diff)
    return total_length


def filter_diff_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where diff_length >= 200. Computes diff_length if column doesn't exist."""
    if "diff_length" not in df.columns:
        if "base_resume" in df.columns and "variant_resume" in df.columns:
            df = df.copy()
            df["diff_length"] = df.apply(
                lambda row: _compute_diff_length(
                    row.get("base_resume", ""),
                    row.get("variant_resume", "")
                ),
                axis=1
            )
        else:
            # If required columns don't exist, return original dataframe
            return df
    
    return df[df["diff_length"] >= 200].copy()


def filter_summary_only_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where summary_only == True. Computes summary_only if column doesn't exist."""
    if "summary_only" not in df.columns:
        if "base_resume" in df.columns and "variant_resume" in df.columns:
            # Import here to avoid circular imports
            try:
                from skills_in_summary import analyze_df_for_summary_only_changes
                df = df.copy()
                summary_analysis = analyze_df_for_summary_only_changes(df, added_skills_col='differed_qualifications')
                df["summary_only"] = summary_analysis["all_summary_only"]
            except ImportError:
                # If import fails, return original dataframe
                return df
        else:
            # If required columns don't exist, return original dataframe
            return df
    
    return df[df["summary_only"] == False].copy()


def _normalize_df(df: pd.DataFrame, framework: str, model_id: str, constructed_by: str) -> pd.DataFrame:
    """Normalize a single CSV DataFrame - same logic as compute_metrics_from_results.py"""
    d = df.copy()
    d["framework"] = framework
    d["model_id"] = model_id
    d["constructed_by"] = constructed_by

    # experiment type
    if "experiment_type_norm" in d.columns:
        d["experiment_type"] = d["experiment_type_norm"].fillna(d.get("experiment_type", ""))
    else:
        d["experiment_type"] = d.get("experiment_type", "")
    d["experiment_type"] = d["experiment_type"].astype(str).str.lower()

    # demographics
    if "demographic_base" not in d.columns:
        d["demographic_base"] = ""
    if "demographic_variant" not in d.columns:
        d["demographic_variant"] = ""
    if (d["demographic_base"] == "").any() or (d["demographic_variant"] == "").any():
        # print("here")
        bases: List[str] = []
        vars_: List[str] = []
        for _, r in d.iterrows():
            b, v = _coerce_demographics(dict(r))
            bases.append(b)
            vars_.append(v)
        d["demographic_base"] = d["demographic_base"].mask(d["demographic_base"] == "", bases)
        d["demographic_variant"] = d["demographic_variant"].mask(d["demographic_variant"] == "", vars_)

    # pair_type
    if "pair_type" in d.columns:
        d["pair_type"] = d["pair_type"].astype(str).str.lower()
    else:
        d["pair_type"] = ""

    # num_differed (k)
    if "num_differed" in d.columns:
        d["num_differed"] = pd.to_numeric(d["num_differed"], errors="coerce").fillna(0).astype(int)
    else:
        d["num_differed"] = 0
    d["k"] = d["num_differed"]  # Alias for convenience

    # is_valid
    if "is_valid" in d.columns:
        d["is_valid"] = d["is_valid"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        d["is_valid"] = False

    # abstained
    if "abstained" in d.columns:
        d["abstained"] = d["abstained"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        d["abstained"] = False

    # decision
    if "decision" in d.columns:
        d["decision"] = d["decision"].astype(str).str.lower().fillna("")
    else:
        d["decision"] = ""

    # better (ground truth)
    if "better" in d.columns:
        d["better"] = d["better"].astype(str).str.lower().fillna("")
    else:
        d["better"] = ""

    # differed_qualifications list
    if "differed_qualifications" in d.columns:
        d["diff_quals_list"] = d["differed_qualifications"].apply(_coerce_diff_quals)
    else:
        d["diff_quals_list"] = [[] for _ in range(len(d))]

    # Add derived columns for easier analysis
    # Parse race and gender from demographics
    d["race_base"] = d["demographic_base"].str.split("_").str[0]
    d["gender_base"] = d["demographic_base"].str.split("_").str[1]
    d["race_variant"] = d["demographic_variant"].str.split("_").str[0]
    d["gender_variant"] = d["demographic_variant"].str.split("_").str[1]

    # Compute derived metrics that are commonly used
    d["is_correct"] = (
        ((d["decision"] == "first") & (d["better"] == "first")) |
        ((d["decision"] == "second") & (d["better"] == "second")) |
        ((d["abstained"] == "True") & (d["better"] == "equal"))
    )

    # Selection indicators
    d["selected_first"] = d["decision"] == "first"
    d["selected_second"] = d["decision"] == "second"
    
    return d


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = _gather_csv_files(args.inputs)
    if not files:
        print("No CSV inputs found.")
        return

    print(f"Found {len(files)} CSV files to process...")
    
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[ERROR] Failed to read {fp}: {e}")
            traceback.print_exc()
            continue
        if df.empty:
            continue
        framework = _infer_framework_from_path(fp)
        model_id = _infer_model_from_filename(fp)
        constructed_by = _infer_constructed_by(fp)
        try:
            frames.append(_normalize_df(df, framework=framework, model_id=model_id, constructed_by=constructed_by))
        except Exception as e:
            print(f"[ERROR] Normalization failed for {fp}: {e}")
            traceback.print_exc()
            continue

    if not frames:
        print("No rows after normalization.")
        return

    df_all = pd.concat(frames, ignore_index=True)


    df_valid = df_all.copy()

    S = df_valid[df_valid["num_differed"] > 0].copy()
    
    df_equal = df_all.copy()
    E = df_equal[df_equal["num_differed"] == 0].copy()
    
    if not S.empty:
        S_path = out_dir / "strict_preference_pairs.csv"
        S.to_csv(S_path, index=False)
        print(f"Saved {len(S):,} strict preference pairs to {S_path}")
    
    if not E.empty:
        E_path = out_dir / "equal_pairs.csv"
        E.to_csv(E_path, index=False)
        print(f"Saved {len(E):,} equal pairs to {E_path}")


if __name__ == "__main__":
    main()