#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Example usage:

python analysis/correlated_errors_analysis.py --datasets baseline=./evaluations/baseline baseline_noabstain=./evaluations/baseline_noabstain --output_dir ./analysis/correlated_errors
"""


VALID_TRUE = {"true", "1", "yes", "y", "t"}

@dataclass
class DatasetResults:
    name: str
    records: pd.DataFrame
    per_prompt: pd.DataFrame
    pairwise: pd.DataFrame
    accuracies: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlated error analysis across model outputs."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Pairs of dataset_name=directory (e.g., baseline=/path/to/baseline)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_outputs/correlated_errors",
        help="Directory to store output tables and plots.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of hardest prompts to include in the summary table.",
    )
    return parser.parse_args()


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(VALID_TRUE)


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


def _make_question_ids(df: pd.DataFrame) -> pd.Series:
    components = [
        df["job_title"].fillna("").astype(str),
        df["num_differed"].fillna("").astype(str),
        df["differed_qualifications"].fillna("").astype(str),
        df["demographic_base"].fillna("").astype(str),
        df["demographic_variant"].fillna("").astype(str),
    ]
    combined = components[0]
    for part in components[1:]:
        combined = combined + "|::|" + part
    return combined.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())


def _load_directory(dir_path: Path) -> pd.DataFrame:
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")

    frames: List[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"Skipping {fp} (failed to read): {exc}")
            continue

        model_id = fp.stem.split("_paired_resume_decisions_", 1)[0]

        df = df.copy()
        df["model_id"] = model_id

        # Normalize experiment_type columns but don't filter - include all experiment types
        if "experiment_type_norm" in df.columns:
            df["experiment_type_norm"] = df["experiment_type_norm"].astype(str).str.lower()
        if "experiment_type" in df.columns:
            df["experiment_type"] = df["experiment_type"].astype(str).str.lower()

        df["prompt_id"] = df["prompt_id"].astype(str)
        df["is_valid_bool"] = _to_bool(df["is_valid"])
        df["abstained_bool"] = _to_bool(df["abstained"]) if "abstained" in df.columns else False
        df["legacy_prompt_id"] = df["prompt_id"]
        df["question_id"] = _make_question_ids(df)

        # Build column list, including experiment_type columns if they exist
        columns_to_keep = [
            "job_title",
            "job_description",
            "job_source",
            "pair_type",
            "num_differed",
            "demographic_base",
            "demographic_variant",
            "decision",
            "is_valid_bool",
            "abstained_bool",
            "model_id",
            "base_resume",
            "variant_resume",
            "differed_qualifications",
            "question_id",
            "legacy_prompt_id",
        ]
        # Add experiment_type columns if they exist
        if "experiment_type_norm" in df.columns:
            columns_to_keep.append("experiment_type_norm")
        if "experiment_type" in df.columns:
            columns_to_keep.append("experiment_type")
        
        # Only keep columns that actually exist in the dataframe
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        frames.append(df[columns_to_keep])

    if not frames:
        raise ValueError(f"No rows found in {dir_path}")

    full_df = pd.concat(frames, ignore_index=True)
    full_df.drop_duplicates(subset=["question_id", "model_id"], keep="last", inplace=True)
    
    # Filter rows where diff length < 120
    if "base_resume" in full_df.columns and "variant_resume" in full_df.columns:
        print(f"Filtering rows by diff length (before: {len(full_df)} rows)")
        full_df["diff_length"] = full_df.apply(
            lambda row: _compute_diff_length(
                row.get("base_resume", ""),
                row.get("variant_resume", "")
            ),
            axis=1
        )
        full_df = full_df[full_df["diff_length"] >= 200].copy()
        full_df = full_df.drop(columns=["diff_length"])
        print(f"Filtered to {len(full_df)} rows (diff length >= 200)")
    else:
        print("Warning: base_resume or variant_resume columns not found, skipping diff length filter")
    
    return full_df


def _compute_per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("question_id", as_index=False).agg(
        num_models=("model_id", "nunique"),
        num_errors=("is_valid_bool", lambda x: (~x).sum()),
        num_abstained=("abstained_bool", "sum"),
        job_title=("job_title", "first"),
        job_description=("job_description", "first"),
        job_source=("job_source", "first"),
        pair_type=("pair_type", "first"),
        num_differed=("num_differed", "first"),
        demographic_base=("demographic_base", "first"),
        demographic_variant=("demographic_variant", "first"),
        legacy_prompt_id=("legacy_prompt_id", "first"),
        base_resume=("base_resume", "first"),
        variant_resume=("variant_resume", "first"),
        differed_qualifications=("differed_qualifications", "first"),
        experiment_type_norm=("experiment_type_norm", "first"),
        decision=("decision", "first"),
    )
    grouped["error_rate"] = grouped["num_errors"] / grouped["num_models"].clip(lower=1)
    grouped.sort_values(["num_errors", "error_rate"], ascending=[False, False], inplace=True)
    return grouped


def _compute_pairwise(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    pivot = df.pivot_table(
        index="question_id", columns="model_id", values="is_valid_bool", aggfunc="first"
    )
    # Sort models alphabetically
    pivot = pivot.sort_index(axis=1)
    accuracies = pivot.mean()
    # Sort accuracies alphabetically by index (model_id)
    accuracies = accuracies.sort_index()

    pivot_bool = pivot.fillna(False)
    pivot_bool = pivot_bool.astype(bool)
    errors = ~pivot_bool
    models = list(errors.columns)
    data = np.full((len(models), len(models)), np.nan)

    for i, mi in enumerate(models):
        err_i = errors[mi]
        for j, mj in enumerate(models):
            err_j = errors[mj]
            mask = (err_i | err_j).fillna(False)
            mask = mask.astype(bool)
            denom = mask.sum()
            if denom == 0:
                continue
            numerator = (err_i & err_j).fillna(False)
            numerator = numerator.astype(bool)
            numerator = numerator[mask].sum()
            data[i, j] = numerator / denom

    pairwise_df = pd.DataFrame(data, index=models, columns=models)
    # Sort by alphabetical order
    pairwise_df = pairwise_df.sort_index(axis=0).sort_index(axis=1)
    return pairwise_df, accuracies


def _compute_pairwise_by_job(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series, int]]:
    """Compute pairwise agreement for each job title separately. Returns dict mapping job_title to (pairwise_df, accuracies, n)."""
    results = {}
    
    if "job_title" not in df.columns:
        return results
    
    for job_title, job_df in df.groupby("job_title", dropna=False):
        if job_df.empty:
            continue
        
        n = len(job_df["question_id"].unique())
        
        pivot = job_df.pivot_table(
            index="question_id", columns="model_id", values="is_valid_bool", aggfunc="first"
        )
        # Sort models alphabetically
        pivot = pivot.sort_index(axis=1)
        accuracies = pivot.mean()
        # Sort accuracies alphabetically by index (model_id)
        accuracies = accuracies.sort_index()
        
        pivot_bool = pivot.fillna(False)
        pivot_bool = pivot_bool.astype(bool)
        errors = ~pivot_bool
        models = list(errors.columns)
        data = np.full((len(models), len(models)), np.nan)
        
        for i, mi in enumerate(models):
            err_i = errors[mi]
            for j, mj in enumerate(models):
                err_j = errors[mj]
                mask = (err_i | err_j).fillna(False)
                mask = mask.astype(bool)
                denom = mask.sum()
                if denom == 0:
                    continue
                numerator = (err_i & err_j).fillna(False)
                numerator = numerator.astype(bool)
                numerator = numerator[mask].sum()
                data[i, j] = numerator / denom
        
        pairwise_df = pd.DataFrame(data, index=models, columns=models)
        # Sort by alphabetical order
        pairwise_df = pairwise_df.sort_index(axis=0).sort_index(axis=1)
        
        results[str(job_title)] = (pairwise_df, accuracies, n)
    
    return results


def _compute_pairwise_by_k(df: pd.DataFrame) -> Dict[int, Tuple[pd.DataFrame, pd.Series, int]]:
    """Compute pairwise agreement for each k (num_differed) value separately. Returns dict mapping k to (pairwise_df, accuracies, n)."""
    results = {}
    
    if "num_differed" not in df.columns:
        return results
    
    for k, k_df in df.groupby("num_differed", dropna=False):
        if k_df.empty:
            continue
        
        n = len(k_df["question_id"].unique())
        
        pivot = k_df.pivot_table(
            index="question_id", columns="model_id", values="is_valid_bool", aggfunc="first"
        )
        # Sort models alphabetically
        pivot = pivot.sort_index(axis=1)
        accuracies = pivot.mean()
        # Sort accuracies alphabetically by index (model_id)
        accuracies = accuracies.sort_index()
        
        pivot_bool = pivot.fillna(False)
        pivot_bool = pivot_bool.astype(bool)
        errors = ~pivot_bool
        models = list(errors.columns)
        data = np.full((len(models), len(models)), np.nan)
        
        for i, mi in enumerate(models):
            err_i = errors[mi]
            for j, mj in enumerate(models):
                err_j = errors[mj]
                mask = (err_i | err_j).fillna(False)
                mask = mask.astype(bool)
                denom = mask.sum()
                if denom == 0:
                    continue
                numerator = (err_i & err_j).fillna(False)
                numerator = numerator.astype(bool)
                numerator = numerator[mask].sum()
                data[i, j] = numerator / denom
        
        pairwise_df = pd.DataFrame(data, index=models, columns=models)
        # Sort by alphabetical order
        pairwise_df = pairwise_df.sort_index(axis=0).sort_index(axis=1)
        
        results[int(k)] = (pairwise_df, accuracies, n)
    
    return results

def _compute_pairwise_agreement(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute pairwise agreement rate P(same answer) across models."""
    pivot = df.pivot_table(
        index="question_id", columns="model_id", values="is_valid_bool", aggfunc="first"
    )
    # Sort models alphabetically
    pivot = pivot.sort_index(axis=1)
    accuracies = pivot.mean()
    # Sort accuracies alphabetically by index (model_id)
    accuracies = accuracies.sort_index()

    models = list(pivot.columns)
    data = np.full((len(models), len(models)), np.nan)

    for i, mi in enumerate(models):
        ai = pivot[mi]
        for j, mj in enumerate(models):
            aj = pivot[mj]
            mask = ai.notna() & aj.notna()
            denom = mask.sum()
            if denom == 0:
                continue
            numerator = (ai[mask] == aj[mask]).sum()
            data[i, j] = numerator / denom

    pairwise_df = pd.DataFrame(data, index=models, columns=models)
    # Sort by alphabetical order
    pairwise_df = pairwise_df.sort_index(axis=0).sort_index(axis=1)
    return pairwise_df, accuracies


def _plot_heatmap(pairwise_df: pd.DataFrame, accuracies: pd.Series, out_path: Path, title: str = None) -> None:
    plt.figure(figsize=(max(6, len(pairwise_df) * 0.6), max(4, len(pairwise_df) * 0.6)))
    sns.heatmap(
        pairwise_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Error Agreement Rate"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plot_title = title if title else "Error Agreement Rate"
    plt.title(plot_title)
    # subtitle = "\n".join(f"{model}: {acc:.3f}" for model, acc in accuracies.items())
    # plt.suptitle(subtitle, y=0.94, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze_dataset(name: str, dir_path: Path, out_dir: Path, topk: int) -> DatasetResults:
    df = _load_directory(dir_path)

    print(df.columns)

    per_prompt = _compute_per_prompt(df)
    pairwise, accuracies = _compute_pairwise(df)

    agreement, _ = _compute_pairwise_agreement(df)

    agreement.to_csv(out_dir / "pairwise_agreement_rate.csv")
    _plot_heatmap(agreement, accuracies, out_dir / "pairwise_agreement_rate.png", 
                title="Error Agreement Rate")

    dataset_dir = out_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    per_prompt.to_csv(dataset_dir / "per_prompt_error_table.csv", index=False)
    pairwise.to_csv(dataset_dir / "pairwise_error_agreement.csv")
    accuracies.to_csv(dataset_dir / "model_accuracy.csv", header=["accuracy"])

    hardest = per_prompt
    hardest.to_csv(dataset_dir / f"all_hardest_prompts.csv", index=False)

    _plot_heatmap(pairwise, accuracies, dataset_dir / "pairwise_error_agreement.png")

    # Compute and save pairwise agreement by k
    pairwise_by_k = _compute_pairwise_by_k(df)
    k_dir = dataset_dir / "by_k"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    for k, (pairwise_df, k_accuracies, n) in sorted(pairwise_by_k.items()):
        pairwise_df.to_csv(k_dir / f"pairwise_error_agreement_k{k}.csv")
        k_accuracies.to_csv(k_dir / f"model_accuracy_k{k}.csv", header=["accuracy"])
        
        title = f"Error Agreement Rate: k={k} (n={n})"
        _plot_heatmap(
            pairwise_df, 
            k_accuracies, 
            k_dir / f"pairwise_error_agreement_k{k}.png",
            title=title
        )

    # Compute and save pairwise agreement by job
    pairwise_by_job = _compute_pairwise_by_job(df)
    jobs_dir = dataset_dir / "by_job"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    
    for job_title, (pairwise_df, job_accuracies, n) in pairwise_by_job.items():
        # Sanitize job title for filename
        safe_job_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in job_title)
        safe_job_title = safe_job_title.replace(' ', '_')
        
        pairwise_df.to_csv(jobs_dir / f"pairwise_error_agreement_{safe_job_title}.csv")
        job_accuracies.to_csv(jobs_dir / f"model_accuracy_{safe_job_title}.csv", header=["accuracy"])
        
        title = f"Error Agreement Rate: {job_title} (n={n})"
        _plot_heatmap(
            pairwise_df, 
            job_accuracies, 
            jobs_dir / f"pairwise_error_agreement_{safe_job_title}.png",
            title=title
        )

    return DatasetResults(
        name=name,
        records=df,
        per_prompt=per_prompt,
        pairwise=pairwise,
        accuracies=accuracies,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()

    dataset_specs: Dict[str, Path] = {}
    for spec in args.datasets:
        if "=" not in spec:
            raise ValueError(f"Dataset spec '{spec}' must look like name=/path/to/dir")
        name, path = spec.split("=", 1)
        dataset_specs[name.strip()] = Path(path).expanduser().resolve()

    results: List[DatasetResults] = []
    for name, path in dataset_specs.items():
        print(f"Analyzing {name} from {path}")
        res = analyze_dataset(name, path, out_dir, args.topk)
        results.append(res)
        print(
            f"  -> saved tables and heatmap under {out_dir / name} | "
            f"{len(res.per_prompt)} prompts"
        )


if __name__ == "__main__":
    main()
