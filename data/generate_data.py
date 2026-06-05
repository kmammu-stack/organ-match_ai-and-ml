import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

BLOOD_TYPES = ['A', 'B', 'AB', 'O']
BLOOD_DIST  = [0.34, 0.09, 0.04, 0.53]   # real UNOS distribution

HLA_ALLELES = {
    'A':  list(range(1, 35)),
    'B':  list(range(1, 60)),
    'C':  list(range(1, 20)),
    'DR': list(range(1, 20)),
    'DQ': list(range(1, 10)),
    'DP': list(range(1, 10)),
}

def random_hla():
    return {locus: random.sample(alleles, 2) for locus, alleles in HLA_ALLELES.items()}

def generate_donors(n=500):
    records = []
    for _ in range(n):
        records.append({
            'donor_id':       fake.uuid4(),
            'blood_type':     np.random.choice(BLOOD_TYPES, p=BLOOD_DIST),
            'age':            np.random.randint(18, 70),
            'hla':            random_hla(),
            'donor_type':     np.random.choice(['living', 'deceased'], p=[0.35, 0.65]),
            'cold_ischemia_time': np.random.randint(4, 36),  # hours
            'location_lat':   float(fake.latitude()),
            'location_lon':   float(fake.longitude()),
        })
    return pd.DataFrame(records)

def generate_recipients(n=1000):
    records = []
    for _ in range(n):
        records.append({
            'recipient_id':  fake.uuid4(),
            'blood_type':    np.random.choice(BLOOD_TYPES, p=BLOOD_DIST),
            'age':           np.random.randint(5, 75),
            'hla':           random_hla(),
            'gfr':           np.random.randint(5, 30),         # kidney function
            'pra':           np.random.uniform(0, 100),        # sensitization %
            'urgency_score': np.random.randint(1, 5),          # 1=low, 4=critical
            'wait_time':     np.random.randint(1, 3650),       # days on waitlist
            'location_lat':  float(fake.latitude()),
            'location_lon':  float(fake.longitude()),
        })
    return pd.DataFrame(records)

if __name__ == '__main__':
    donors     = generate_donors(500)
    recipients = generate_recipients(1000)
    donors.to_csv('data/donors.csv', index=False)
    recipients.to_csv('data/recipients.csv', index=False)
    print(f"Donors:     {len(donors)} rows")
    print(f"Recipients: {len(recipients)} rows")
    print("\nDonor blood type distribution:")
    print(donors['blood_type'].value_counts())