"""Apply cohort-level filters and write filtered datasets.

By default the script will:
 - remove patients with AGE < 18
 - remove admissions that are readmissions (Number_of_Prior_Readmits > 0)
 - remove rows missing > 1/3 of important measured flags (treated as not measured when the *_measured flag==0)

Outputs are written to cohort/ with `_filtered` suffix to avoid overwriting originals.
"""

import csv
import os
from shutil import copyfile

COHORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'cohort')
FILTERED_IN = os.path.join(COHORT_DIR, 'filtered_cohort.csv')
TRAIN_IN = os.path.join(COHORT_DIR, 'processed_train.csv')
TEST_IN = os.path.join(COHORT_DIR, 'processed_test.csv')
LABELS_IN = os.path.join(COHORT_DIR, 'labels.csv')
LABELS_48H_IN = os.path.join(COHORT_DIR, 'labels_48h.csv')

TRAIN_OUT = os.path.join(COHORT_DIR, 'processed_train_filtered.csv')
TEST_OUT = os.path.join(COHORT_DIR, 'processed_test_filtered.csv')
COHORT_OUT = os.path.join(COHORT_DIR, 'filtered_cohort_filtered.csv')
LABELS_OUT = os.path.join(COHORT_DIR, 'labels_filtered.csv')
LABELS_48H_OUT = os.path.join(COHORT_DIR, 'labels_48h_filtered.csv')

MEASURED_FLAGS = [
    'BUN_measured','Creatinine_measured','Glucose_measured','HMG_measured','WBC_measured','Platelet_measured',
    'Lactate_measured','Albumin_measured','SBP_measured','DBP_measured','HeartRate_measured','RespRate_measured',
    'SpO2_measured','PaO2_FiO2_measured','GCS_measured','UrineOutput_measured'
]


def read_csv_as_dict(path):
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
        header = r.fieldnames
    return header, rows


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def should_remove_row(row):
    # AGE < 18
    try:
        age = float(row.get('AGE',''))
        if age < 18.0:
            return True
    except Exception:
        # if AGE missing, do not remove here (could be handled by measured flags)
        pass

    # Number_of_Prior_Readmits > 0 => post discharge readmission
    try:
        if float(row.get('Number_of_Prior_Readmits', '0')) > 0:
            return True
    except Exception:
        pass

    # measured flags: treat 0 / false / empty as not measured
    existing = [c for c in MEASURED_FLAGS if c in row]
    if existing:
        cnt_not_measured = 0
        for c in existing:
            v = row.get(c, '').strip()
            if v == '' or v in ('0','False','false','FALSE'):
                cnt_not_measured += 1
        if cnt_not_measured > len(existing) / 3.0:
            return True

    return False


def main():
    # Determine which ICUSTAY_IDs should be removed using raw features where available
    # Use cohort/features.csv for accurate Number_of_Prior_Readmits and measured flags
    features_path = os.path.join(COHORT_DIR, 'features.csv')
    remove_set = set()

    if os.path.exists(features_path):
        h, feats = read_csv_as_dict(features_path)
        for row in feats:
            icu = row.get('ICUSTAY_ID')
            # AGE: consult filtered_cohort's AGE if available, else use features.csv AGE
            try:
                age_val = None
                # get age from filtered cohort if present
                # we'll handle this globally below; for now prefer features.csv AGE
                if row.get('AGE') is not None and row.get('AGE')!='':
                    age_val = float(row.get('AGE'))
                if age_val is not None and age_val < 18.0:
                    remove_set.add(icu)
                    continue
            except Exception:
                pass

            # raw Number_of_Prior_Readmits from features.csv should reflect true prior reads
            try:
                if row.get('Number_of_Prior_Readmits') is not None and row.get('Number_of_Prior_Readmits')!='':
                    if float(row.get('Number_of_Prior_Readmits')) > 0:
                        remove_set.add(icu)
                        continue
            except Exception:
                pass

            # measured flags check (treat 0/False/empty as not measured)
            existing = [c for c in MEASURED_FLAGS if c in row]
            if existing:
                cnt_not_measured = 0
                for c in existing:
                    v = row.get(c, '').strip()
                    if v == '' or v in ('0','False','false','FALSE'):
                        cnt_not_measured += 1
                if cnt_not_measured > len(existing) / 3.0:
                    remove_set.add(icu)
                    continue
    else:
        # fallback: if features.csv not present, iterate processed_train/test and use their values
        for p in (TRAIN_IN, TEST_IN):
            header, rows = read_csv_as_dict(p)
            for row in rows:
                if should_remove_row(row):
                    remove_set.add(row.get('ICUSTAY_ID'))

    # report
    total_rows = 0
    for p in (TRAIN_IN, TEST_IN):
        with open(p, 'r') as f:
            total_rows += sum(1 for _ in f) - 1

    print('Total rows (train+test):', total_rows)
    print('Rows to remove (ICUSTAY_IDs):', len(remove_set))

    # create filtered processed_train and processed_test
    for p_in, p_out in ((TRAIN_IN, TRAIN_OUT),(TEST_IN, TEST_OUT)):
        header, rows = read_csv_as_dict(p_in)
        keep = [r for r in rows if r.get('ICUSTAY_ID') not in remove_set]
        write_csv(p_out, header, keep)
        print('Wrote', len(keep), 'rows to', p_out)

    # filter cohort/filtered_cohort.csv by ICUSTAY_ID
    header, rows = read_csv_as_dict(FILTERED_IN)
    keep = [r for r in rows if str(r.get('ICUSTAY_ID')) not in remove_set]
    write_csv(COHORT_OUT, header, keep)
    print('Wrote filtered cohort', len(keep), 'rows to', COHORT_OUT)

    # filter labels.csv and labels_48h.csv (if they exist)
    if os.path.exists(LABELS_IN):
        header, rows = read_csv_as_dict(LABELS_IN)
        keep = [r for r in rows if str(r.get('ICUSTAY_ID')) not in remove_set]
        write_csv(LABELS_OUT, header, keep)
        print('Wrote filtered labels to', LABELS_OUT, 'rows:', len(keep))

    if os.path.exists(LABELS_48H_IN):
        header, rows = read_csv_as_dict(LABELS_48H_IN)
        keep = [r for r in rows if str(r.get('ICUSTAY_ID')) not in remove_set]
        write_csv(LABELS_48H_OUT, header, keep)
        print('Wrote filtered 48h labels to', LABELS_48H_OUT, 'rows:', len(keep))


if __name__ == '__main__':
    main()
