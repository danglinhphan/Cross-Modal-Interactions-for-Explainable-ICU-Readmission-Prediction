import sqlite3
import csv

# Path to the database
db_path = '/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query to filter cohort: age >= 18, ICU stay >= 24 hours, and exclude patients who died during their first ICU stay
query = """
WITH first_icu AS (
    SELECT i.SUBJECT_ID, i.HADM_ID, i.ICUSTAY_ID, i.INTIME, i.OUTTIME,
           ROW_NUMBER() OVER (PARTITION BY i.SUBJECT_ID ORDER BY i.INTIME) AS rn
    FROM ICUSTAYS i
)
SELECT DISTINCT p.SUBJECT_ID, fi.HADM_ID, fi.ICUSTAY_ID,
       ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25) AS age,
       fi.INTIME, fi.OUTTIME,
       (julianday(fi.OUTTIME) - julianday(fi.INTIME)) * 24 AS icu_hours
FROM PATIENTS p
JOIN ADMISSIONS a ON p.SUBJECT_ID = a.SUBJECT_ID
JOIN first_icu fi ON a.HADM_ID = fi.HADM_ID AND fi.rn = 1
WHERE ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25) >= 18
  AND (julianday(fi.OUTTIME) - julianday(fi.INTIME)) * 24 >= 24
  AND (a.DEATHTIME IS NULL OR a.DEATHTIME NOT BETWEEN fi.INTIME AND fi.OUTTIME)
"""

# Execute the query
cursor.execute(query)
results = cursor.fetchall()

# Close the connection
conn.close()

# Write to CSV
csv_path = '/Users/phandanglinh/Desktop/VRES/cohort/filtered_cohort.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'AGE', 'INTIME', 'OUTTIME', 'ICU_HOURS'])
    # Write data
    for row in results:
        writer.writerow(row)

print(f"Filtered cohort saved to {csv_path}")