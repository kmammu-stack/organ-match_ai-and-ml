import pandas as pd
import numpy as np
from faker import Faker
import random
import json

fake = Faker()
np.random.seed(42)
random.seed(42)

BLOOD_TYPES = ['A', 'B', 'AB', 'O']
BLOOD_DIST  = [0.34, 0.09, 0.04, 0.53]

HLA_ALLELES = {
    'A':  list(range(1, 35)),
    'B':  list(range(1, 60)),
    'C':  list(range(1, 20)),
    'DR': list(range(1, 20)),
    'DQ': list(range(1, 10)),
    'DP': list(range(1, 10)),
}

def random_hla():
    return {locus: random.sample(alleles, 2) 
            for locus, alleles in HLA_ALLELES.items()}

def generate_donors(n=500):
    records = []
    for _ in range(n):
        records.append({
            'donor_id':           fake.uuid4(),
            'blood_type':         str(np.random.choice(BLOOD_TYPES, p=BLOOD_DIST)),
            'age':                int(np.random.randint(18, 70)),
            'hla':                json.dumps(random_hla()),
            'donor_type':         str(np.random.choice(['living', 'deceased'], p=[0.35, 0.65])),
            'cold_ischemia_time': int(np.random.randint(4, 36)),
            'location_lat':       round(float(fake.latitude()), 6),
            'location_lon':       round(float(fake.longitude()), 6),
        })
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()
    return df

def generate_recipients(n=1000):
    records = []
    for _ in range(n):
        records.append({
            'recipient_id':  fake.uuid4(),
            'blood_type':    str(np.random.choice(BLOOD_TYPES, p=BLOOD_DIST)),
            'age':           int(np.random.randint(5, 75)),
            'hla':           json.dumps(random_hla()),
            'gfr':           int(np.random.randint(5, 30)),
            'pra':           round(float(np.random.uniform(0, 100)), 2),
            'urgency_score': int(np.random.randint(1, 5)),
            'wait_time':     int(np.random.randint(1, 3650)),
            'location_lat':  round(float(fake.latitude()), 6),
            'location_lon':  round(float(fake.longitude()), 6),
        })
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()
    return df

if __name__ == '__main__':
    donors     = generate_donors(500)
    recipients = generate_recipients(1000)

    # Verify no spaces in any column
    print("Donor columns:",     donors.columns.tolist())
    print("Recipient columns:", recipients.columns.tolist())
    print("Donor blood types:", donors['blood_type'].unique())
    print("Recipient blood types:", recipients['blood_type'].unique())

    donors.to_csv('donors.csv',         index=False)
    recipients.to_csv('recipients.csv', index=False)

    print(f"\nDonors:     {len(donors)} rows")
    print(f"Recipients: {len(recipients)} rows")
    print("\nDonor blood type distribution:")
    print(donors['blood_type'].value_counts())