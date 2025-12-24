
import pandas as pd
import numpy as np

FEATURE_FILE = '/Users/phandanglinh/Desktop/VRES/cohort/features_phase5.csv'
LABEL_FILE = '/Users/phandanglinh/Desktop/VRES/cohort/new_cohort_icu_readmission_labels.csv'

def check_target():
    df = pd.read_csv(FEATURE_FILE, nrows=1)
    cols = list(df.columns)
    print(f"Total columns: {len(cols)}")
    
    potential_targets = [c for c in cols if 'label' in c.lower() or 'readmission' in c.lower() or 'y' == c.lower()]
    print(f"Potential target columns in features file: {potential_targets}")
    
    # Check label file if not in features
    if not potential_targets:
        print("Target not found in features file. Checking distinct label file...")
        if os.path.exists(LABEL_FILE):
             lbl = pd.read_csv(LABEL_FILE, nrows=1)
             print(f"Label file columns: {list(lbl.columns)}")
        else:
             print(f"Label file not found: {LABEL_FILE}")

import os
if __name__ == "__main__":
    check_target()
