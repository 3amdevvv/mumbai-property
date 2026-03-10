import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

print("Loading dataset...")
df = pd.read_csv("housing.csv")

# Convert prices
df['price_in_lakhs'] = df.apply(
    lambda r: r['price'] * 100 if r['price_unit'] == 'Cr' else r['price'], axis=1
)

# Basic cleaning
df = df[(df['area'] >= 100) & (df['area'] <= 10000)]

# Encoders
le_region = LabelEncoder()
le_type   = LabelEncoder()
le_status = LabelEncoder()
le_age    = LabelEncoder()

df['region_enc'] = le_region.fit_transform(df['region'])
df['type_enc']   = le_type.fit_transform(df['type'])
df['status_enc'] = le_status.fit_transform(df['status'])
df['age_enc']    = le_age.fit_transform(df['age'])

region_avg_map = df.groupby('region')['price_in_lakhs'].mean().to_dict()
df['region_avg'] = df['region'].map(region_avg_map)

FEATURES = ['bhk','area','region_enc','type_enc','status_enc','age_enc','region_avg']
X = df[FEATURES]
y = df['price_in_lakhs']

print("Training model...")
model = RandomForestRegressor(n_estimators=40,max_depth=12, random_state=42)
model.fit(X, y)

print("Saving model...")

pickle.dump({
    "model": model,
    "region_enc_map": dict(zip(le_region.classes_, le_region.transform(le_region.classes_))),
    "type_enc_map": dict(zip(le_type.classes_, le_type.transform(le_type.classes_))),
    "status_enc_map": dict(zip(le_status.classes_, le_status.transform(le_status.classes_))),
    "age_enc_map": dict(zip(le_age.classes_, le_age.transform(le_age.classes_))),
    "region_avg_map": region_avg_map
}, open("model.pkl", "wb"))

print("✅ Model saved as model.pkl")