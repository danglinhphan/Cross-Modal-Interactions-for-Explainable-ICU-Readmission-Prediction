from scripts.cleanup_models import list_deletable_models

if __name__ == '__main__':
    metrics_csvs = ['metrics_filtered_ge0_5.csv', 'metrics_ge_50pct_simult.csv']
    base_dir = '.'
    keep_dirs, deletable, deletable_dirs = list_deletable_models(metrics_csvs, base_dir=base_dir, threshold=0.5, strict=False, remove_run_dirs=True)
    print('Keep directories:')
    for k in keep_dirs:
        print(' -', k)
    print('\nDeletable models:')
    for d in deletable:
        print(' -', d)
    print('\nDeletable directories:')
    for d in deletable_dirs:
        print(' -', d)
