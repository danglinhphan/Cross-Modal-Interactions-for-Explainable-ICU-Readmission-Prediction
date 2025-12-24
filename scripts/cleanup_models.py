#!/usr/bin/env python3
"""
Script to cleanup model artifacts that are not referenced in a metrics CSV.
Behavior:
 - Reads a CSV file (like `metrics_filtered_ge0_5.csv`) of metrics with columns including 'file' and metric values ('precision','recall','f1')
 - Keeps directories containing a metrics file referenced in the CSV that meet the threshold for precision/recall/f1 simultaneously
 - Deletes model files (.joblib, .pkl, .bin, .model) outside the keep set when `--force` is set
 - Optionally, delete the complete run "outputs/<run>" directories with `--remove-run-dirs`
 - Default is dry-run that lists what would be deleted

Usage:
    python scripts/cleanup_models.py --metrics-csv metrics_filtered_ge0_5.csv [--base-dir .] [--dry-run] [--force] [--threshold 0.5] [--strict] [--remove-run-dirs]

"""

import argparse
import csv
import os
import sys
from pathlib import Path

MODEL_EXTS = [".joblib", ".pkl", ".model", ".bin"]

def parse_metrics_csv(csv_path, threshold=0.5, strict=False):
    keep_dirs = set()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # columns: model,file,desc,precision,recall,f1
        if 'file' not in reader.fieldnames:
            raise ValueError("CSV must contain a 'file' column pointing to model output/metric files")
        for row in reader:
            file_path = row['file'].strip()
            if not file_path:
                continue
            # try to parse metrics: precision, recall, f1
            try:
                precision = float(row.get('precision', '')) if 'precision' in row else None
                recall = float(row.get('recall', '')) if 'recall' in row else None
                f1 = float(row.get('f1', '')) if 'f1' in row else None
            except Exception:
                precision = recall = f1 = None

            # determine keep if all three metrics pass threshold
            def _passes(p):
                if p is None:
                    return False
                if strict:
                    return p > threshold
                else:
                    return p >= threshold

            if not (_passes(precision) and _passes(recall) and _passes(f1)):
                # not meeting threshold simultaneously; skip
                continue
            # normalize to absolute path within repo
            p = (Path.cwd() / file_path).resolve()
            keep_dirs.add(str(p.parent))
    return keep_dirs


def parse_metrics_csvs(csv_paths, threshold=0.5, strict=False):
    """Parse multiple CSV paths (or directories) and return a unified set of keep directories."""
    keep_dirs = set()
    for p in csv_paths:
        pp = Path(p)
        if not pp.exists():
            # if p looks like a directory, scan for CSVs inside
            if pp.suffix == '':
                continue
        if pp.is_dir():
            for child in pp.glob('*.csv'):
                keep_dirs.update(parse_metrics_csv(child, threshold=threshold, strict=strict))
        else:
            keep_dirs.update(parse_metrics_csv(pp, threshold=threshold, strict=strict))
    return keep_dirs


def find_model_files(base_dir):
    base_dir = Path(base_dir)
    files = []
    # Default search folders under the project that likely contain model artifacts
    if str(base_dir) == '.':
        candidate_dirs = [
            Path('Train_model_XGB/outputs'),
            Path('Train_model_STACK/outputs'),
            Path('Train_EBM/outputs'),
            Path('outputs')
        ]
        search_dirs = [d for d in candidate_dirs if d.exists()]
        if not search_dirs:
            # fallback: use repo root as before
            search_dirs = [base_dir]
    else:
        search_dirs = [base_dir]

    for sd in search_dirs:
        for ext in MODEL_EXTS:
            files.extend(sd.rglob(f"*{ext}"))
    return [str(p) for p in files]


def is_in_keep(path, keep_dirs):
    p = Path(path).resolve()
    for kd in keep_dirs:
        if str(p).startswith(kd):
            return True
    return False


def list_deletable_models(metrics_csvs, base_dir='.', threshold=0.5, strict=False, remove_run_dirs=False):
    # metrics_csvs: single string or list of paths, or dir
    if isinstance(metrics_csvs, (list, tuple)):
        keep_dirs = parse_metrics_csvs(metrics_csvs, threshold=threshold, strict=strict)
    else:
        keep_dirs = parse_metrics_csvs([metrics_csvs], threshold=threshold, strict=strict)
    model_files = find_model_files(base_dir)
    deletable = [mf for mf in model_files if not is_in_keep(mf, keep_dirs) and "/outputs/keep/" not in mf]

    # Optionally, include entire run directories for deletion if none of their model files are kept
    deletable_dirs = set()
    if remove_run_dirs:
        # determine run root as the subdirectory under an outputs folder. e.g., Train_model_XGB/outputs/<run_dir>
        for mf in model_files:
            p = Path(mf)
            parts = p.parts
            # find last occurrence of 'outputs' in parts
            if 'outputs' in parts:
                idx = len(parts) - 1 - list(reversed(parts)).index('outputs')
                if idx + 1 < len(parts):
                    run_root = Path(*parts[:idx+2])
                else:
                    run_root = Path(*parts[:idx+1])
            else:
                # fallback: parent dir containing the model
                run_root = p.parent
            # if run_root not in any keep_dirs then mark for deletion
            if not is_in_keep(str(run_root), keep_dirs) and "/outputs/keep/" not in str(run_root):
                deletable_dirs.add(str(run_root))
        # exclude parent outputs root from deletable dirs
        deletable_dirs = {d for d in deletable_dirs if not d.endswith('outputs')}
    return keep_dirs, deletable, sorted(deletable_dirs)
    return keep_dirs, deletable


def delete_models(model_paths):
    deleted = []
    for p in model_paths:
        try:
            os.remove(p)
            deleted.append(p)
            parent = Path(p).parent
            # remove empty directory if not keep
            if parent.exists() and not any(parent.iterdir()) and "/outputs/keep/" not in str(parent):
                parent.rmdir()
        except Exception:
            pass
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Cleanup model files not present in the metrics CSV")
    parser.add_argument("--metrics-csv", nargs='+', default=["metrics_filtered_ge0_5.csv"], help="One or more CSVs or directories that list kept model output locations (space-separated)")
    parser.add_argument("--metrics-dir", default=None, help="Directory of CSV metrics files; used if --metrics-csv omitted or to add an additional source")
    parser.add_argument("--base-dir", default=".", help="Base dir to search for outputs (default: current working directory)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to require for metrics (default 0.5)")
    parser.add_argument("--strict", action="store_true", help="Use strict '>' comparison instead of '>=' for thresholds")
    parser.add_argument("--remove-run-dirs", action="store_true", help="Delete whole run directories under outputs that don't meet the threshold")
    parser.add_argument("--dry-run", action="store_true", help="Show files that would be deleted")
    parser.add_argument("--force", action="store_true", help="Actually delete the files")
    args = parser.parse_args()

    # metrics csv may be a list or a pattern
    csv_paths = args.metrics_csv
    if args.metrics_dir:
        csv_paths = csv_paths + [args.metrics_dir]

    try:
        keep_dirs = parse_metrics_csvs(csv_paths, threshold=args.threshold, strict=args.strict)
    except Exception as e:
        print(f"Failed to parse metrics CSV(s) {csv_paths}: {e}")
        sys.exit(1)
    if not keep_dirs:
        print("No keep entries detected in CSV — aborting.")
        sys.exit(0)

    print("Keep directories (derived from CSV):")
    for k in sorted(keep_dirs):
        print(" - ", k)

    base_dir = Path(args.base_dir)
    model_files = find_model_files(base_dir)
    keep_dirs, to_delete, deletable_dirs = list_deletable_models(csv_paths, base_dir=args.base_dir, threshold=args.threshold, strict=args.strict, remove_run_dirs=args.remove_run_dirs)

    if not to_delete and not deletable_dirs:
        print("No model files to delete — everything under outputs seems to be in the keep set.")
        sys.exit(0)

    if args.dry_run or not args.force:
        print("Dry-run (no files will be deleted). Files/dirs that would be deleted:")
        for p in to_delete:
            print(p)
        for d in deletable_dirs:
            print(f"(DIR) {d}")
        if not args.force:
            print("Run with --force to delete the files.")
        sys.exit(0)

    # Perform deletion
    for p in to_delete:
        try:
            print(f"Deleting {p}")
            os.remove(p)
            parent = Path(p).parent
            # remove empty directory if not keep
            if parent.exists() and not any(parent.iterdir()) and "/outputs/keep/" not in str(parent):
                print(f"Removing empty folder {parent}")
                parent.rmdir()
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
    if args.remove_run_dirs and deletable_dirs:
        import shutil
        base_resolved = base_dir.resolve()
        for d in deletable_dirs:
            try:
                dirp = Path(d).resolve()
                # safety - only delete if inside base_dir
                if base_resolved in dirp.parents or dirp == base_resolved:
                    print(f"Removing run directory {dirp}")
                    shutil.rmtree(dirp)
                else:
                    print(f"Skipping deletion of {dirp} — not inside base_dir {base_resolved}")
            except Exception as e:
                print(f"Failed to remove directory {d}: {e}")
    print("Deletion completed")


if __name__ == '__main__':
    main()
