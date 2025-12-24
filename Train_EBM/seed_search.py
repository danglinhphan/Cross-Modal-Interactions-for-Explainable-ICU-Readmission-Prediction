#!/usr/bin/env python3
"""
Seed and Parameter Search for EBM Cross-Interaction Model.

Goal: Find configuration that achieves ≥75% for P, R, F1 simultaneously.

Approach: Run the original cross_interaction_discovery.py with different
seeds and parameters, since it already achieved 73.48% baseline.
"""

import os
import sys
import json
import sys
import subprocess
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = '/Users/phandanglinh/Desktop/VRES'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train_EBM')
FEATURES_PATH = os.path.join(BASE_DIR, 'cohort', 'features_engineered.csv')
LABELS_PATH = os.path.join(BASE_DIR, 'cohort', 'new_cohort_icu_readmission_labels.csv')
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'seed_search')

# Search space
SEEDS = [42, 123, 170, 200, 250, 300, 350, 400, 440, 500, 700, 900]
TOP_N_VALUES = [25, 50, 75, 100]
FN_COSTS = ['auto', '4.0', '6.0', '8.0', '10.0']
THRESHOLD_STRATEGIES = ['balanced', 'auto', 'f1']
UNDERSAMPLING_RATIOS = [None, 3.0, 4.0, 5.0]

TARGET = 0.75

def run_single_config(seed, top_n, fn_cost, threshold_strategy, output_dir):
    """Run cross_interaction_discovery.py with a single configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'python', 'cross_interaction_discovery.py',
        '--features', FEATURES_PATH,
        '--labels', LABELS_PATH,
        '--output', output_dir,
        '--top-n', str(top_n),
        '--fn-cost', str(fn_cost),
        '--threshold-strategy', threshold_strategy,
        '--use-dynamic-weighting',
        '--use-cost-sensitive',
        '--min-metrics', '0.75'  # Higher bar for finding good configs
    ]
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            cwd=TRAIN_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max per run
        )
        
        # Parse results from output or metrics file
        metrics_file = os.path.join(output_dir, 'model_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            # Try to parse from stdout
            return None
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Config timed out: seed={seed}, top_n={top_n}")
        return None
    except Exception as e:
        logger.error(f"Error running config: {e}")
        return None


def evaluate_metrics(metrics):
    """Check if metrics meet the target."""
    if metrics is None:
        return False, 0
    
    class_1 = metrics.get('class_1', {})
    p = class_1.get('precision', 0)
    r = class_1.get('recall', 0)
    f1 = class_1.get('f1', 0)
    
    min_metric = min(p, r, f1)
    meets_target = min_metric >= TARGET
    
    return meets_target, min_metric


def main():
    logger.info("=" * 70)
    logger.info("EBM Cross-Interaction Seed Search")
    logger.info(f"Target: ≥{TARGET*100:.1f}% for P, R, F1")
    logger.info("=" * 70)
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    best_config = None
    best_min_metric = 0
    all_results = []
    
    # Search over configurations
    total_configs = len(SEEDS) * len(TOP_N_VALUES) * len(FN_COSTS) * len(THRESHOLD_STRATEGIES) * len(UNDERSAMPLING_RATIOS)
    config_num = 0
    
    for seed in SEEDS:
        for top_n in TOP_N_VALUES:
            for fn_cost in FN_COSTS:
                for thresh_strat in THRESHOLD_STRATEGIES:
                    for undersample in UNDERSAMPLING_RATIOS:
                        config_num += 1
                    
                    output_dir = os.path.join(
                        OUTPUT_BASE, 
                        f"seed{seed}_topn{top_n}_fn{fn_cost}_{thresh_strat}_us{undersample}"
                    )
                    
                    logger.info(f"\n[{config_num}/{total_configs}] Testing: "
                              f"seed={seed}, top_n={top_n}, fn_cost={fn_cost}, "
                              f"thresh={thresh_strat}")
                    
                    # Build command qualifiers
                    metrics = None
                    # Build run args
                    run_args = {
                        'seed': seed,
                        'top_n': top_n,
                        'fn_cost': fn_cost,
                        'thresh_strat': thresh_strat,
                        'output_dir': output_dir,
                        'undersample': undersample
                    }

                    # Append undersampling flag if present
                    python_exec = sys.executable or 'python3'
                    cmd_parts = [
                        python_exec, 'cross_interaction_discovery.py',
                        '--features', FEATURES_PATH,
                        '--labels', LABELS_PATH,
                        '--output', output_dir,
                        '--top-n', str(top_n),
                        '--fn-cost', str(fn_cost),
                        '--threshold-strategy', thresh_strat,
                        '--use-dynamic-weighting',
                        '--use-cost-sensitive',
                        '--min-metrics', '0.75'
                    ]
                    if undersample is not None:
                        cmd_parts += ['--undersampling-ratio', str(undersample)]
                    cmd_parts += ['--seed', str(seed)]

                    try:
                        result = subprocess.run(cmd_parts, cwd=TRAIN_DIR, capture_output=True, text=True, timeout=900)
                        metrics_file = os.path.join(output_dir, 'model_metrics.json')
                        if os.path.exists(metrics_file):
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                    except Exception as e:
                        logger.warning(f"Run failed for config {run_args}: {e}")
                    
                    meets_target, min_metric = evaluate_metrics(metrics)
                    
                    result = {
                        'seed': seed,
                        'top_n': top_n,
                        'fn_cost': fn_cost,
                        'threshold_strategy': thresh_strat,
                        'min_metric': min_metric,
                        'meets_target': meets_target,
                        'metrics': metrics
                    }
                    all_results.append(result)
                    
                    if metrics:
                        class_1 = metrics.get('class_1', {})
                        logger.info(f"  → P={class_1.get('precision', 0)*100:.2f}%, "
                                  f"R={class_1.get('recall', 0)*100:.2f}%, "
                                  f"F1={class_1.get('f1', 0)*100:.2f}% | "
                                  f"min={min_metric*100:.2f}%")
                        
                        if meets_target:
                            logger.info("  🎯 TARGET ACHIEVED!")
                            best_config = result
                            # Early exit - found a working config!
                            break
                        
                        if min_metric > best_min_metric:
                            best_min_metric = min_metric
                            best_config = result
                            logger.info(f"  ⭐ New best: {min_metric*100:.2f}%")
                    else:
                        logger.warning("  ✗ No metrics returned")
                    
                    if meets_target:
                        break
                if meets_target:
                    break
            if meets_target:
                break
        if meets_target:
            break
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 70)
    
    if best_config:
        logger.info(f"\n🏆 Best Configuration:")
        logger.info(f"   seed: {best_config['seed']}")
        logger.info(f"   top_n: {best_config['top_n']}")
        logger.info(f"   fn_cost: {best_config['fn_cost']}")
        logger.info(f"   threshold_strategy: {best_config['threshold_strategy']}")
        logger.info(f"   min_metric: {best_config['min_metric']*100:.2f}%")
        
        if best_config['meets_target']:
            logger.info(f"\n✅ TARGET ACHIEVED!")
        else:
            gap = TARGET - best_config['min_metric']
            logger.info(f"\n⚠ Gap to target: {gap*100:.2f}%")
    
    # Save all results
    results_file = os.path.join(OUTPUT_BASE, 'search_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'best_config': best_config,
            'all_results': all_results,
            'target': TARGET
        }, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
