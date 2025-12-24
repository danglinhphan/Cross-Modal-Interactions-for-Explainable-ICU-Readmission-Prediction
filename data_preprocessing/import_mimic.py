import sqlite3
import csv
import os

db_path = '/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db'
csv_dir = '/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        table_name = file[:-4]  # remove .csv
        csv_path = os.path.join(csv_dir, file)
        print(f"Processing {file}...")
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Create table
            columns = ', '.join([f'"{col}" TEXT' for col in header])
            cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({columns})')
            
            # Insert data in batches
            batch = []
            batch_size = 5000000
            for row in reader:
                batch.append(row)
                if len(batch) == batch_size:
                    placeholders = ','.join(['?'] * len(header))
                    cursor.executemany(f'INSERT INTO {table_name} VALUES ({placeholders})', batch)
                    batch = []
            if batch:
                placeholders = ','.join(['?'] * len(header))
                cursor.executemany(f'INSERT INTO {table_name} VALUES ({placeholders})', batch)

conn.commit()
conn.close()
print("Database created successfully.")