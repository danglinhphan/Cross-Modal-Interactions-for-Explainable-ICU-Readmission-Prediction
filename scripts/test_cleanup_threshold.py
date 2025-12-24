from pathlib import Path
import csv
import os
from scripts.cleanup_models import list_deletable_models

def setup_test_environment():
    base = Path('Train_model_XGB/outputs')
    pass_dir = base / 'tmp_run_pass'
    fail_dir = base / 'tmp_run_fail'
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    pass_model = pass_dir / 'model_pass.pkl'
    fail_model = fail_dir / 'model_fail.pkl'
    # create fake files
    pass_model.write_text('pass')
    fail_model.write_text('fail')
    # create CSV
    csv_file = Path('test_metrics_tmp.csv')
    with csv_file.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['model','file','precision','recall','f1'])
        writer.writeheader()
        writer.writerow({'model':'pass','file':str(pass_model), 'precision':'0.6', 'recall':'0.6', 'f1':'0.6'})
        writer.writerow({'model':'fail','file':str(fail_model), 'precision':'0.4', 'recall':'0.45', 'f1':'0.49'})
        # additional case: exactly-equal threshold
        # put an extra file in pass_dir (exactly 0.5) to test strict comparison later
        eq_model = pass_dir / 'model_eq.pkl'
        eq_model.write_text('eq')
        writer.writerow({'model':'eq','file':str(eq_model), 'precision':'0.5','recall':'0.5','f1':'0.5'})
    return csv_file, pass_dir, fail_dir, pass_model, fail_model

def cleanup_test_environment(csv_file, pass_dir, fail_dir, pass_model, fail_model):
    try:
        csv_file.unlink()
    except Exception:
        pass
    for p in [pass_model, fail_model, pass_dir, fail_dir]:
        try:
            if p.is_file():
                p.unlink()
            if p.is_dir():
                p.rmdir()
        except Exception:
            pass

if __name__ == '__main__':
    csv_file, pass_dir, fail_dir, pass_model, fail_model = setup_test_environment()
    try:
        keep, deletable, deletable_dirs = list_deletable_models([str(csv_file)], base_dir='.', threshold=0.5, strict=False, remove_run_dirs=True)
        print('Keep dirs:')
        for k in keep:
            print(' -', k)
        print('\nDeletable files:')
        for d in deletable:
            print(' -', d)
        print('\nDeletable dirs:')
        for d in deletable_dirs:
            print(' -', d)
        # test strict behavior: strict=True should drop the equal entry (eq)
        keep_s, deletable_s, deletable_dirs_s = list_deletable_models([str(csv_file)], base_dir='.', threshold=0.5, strict=True, remove_run_dirs=True)
        print('\nStrict mode results:')
        print('Keep (strict):')
        for k in keep_s:
            print(' -', k)
        print('\nDeletable (strict):')
        for d in deletable_s:
            print(' -', d)
        print('\nDeletable dirs (strict):')
        for d in deletable_dirs_s:
            print(' -', d)
    finally:
        cleanup_test_environment(csv_file, pass_dir, fail_dir, pass_model, fail_model)
