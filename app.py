"""
Mumbai House Price Explorer — Flask Backend
==========================================
Run:  python app.py
API available at: http://localhost:5000

Endpoints:
  GET  /api/regions           → All region stats for map
  POST /api/budget            → Budget-based locality finder
  POST /api/predict           → ML price prediction
"""
from flask import render_template
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os
import pickle



app = Flask(__name__)
CORS(app)  # Allow frontend on any port to call this API

@app.route('/')
def home():
    return render_template("ai-advisor")

@app.route('/budget')
def budget_page():
    return render_template("budget")

@app.route('/map')
def map_page():
    return render_template("map")

# ──────────────────────────────────────────
#  1. LOAD & PREPARE DATA (runs once on startup)
# ──────────────────────────────────────────
print("⏳ Loading dataset...")
df = pd.read_csv("housing.csv")


# Unify prices to Lakhs
df['price_in_lakhs'] = df.apply(
    lambda r: r['price'] * 100 if r['price_unit'] == 'Cr' else r['price'], axis=1
)

# Remove outliers
Q1, Q3 = df['price_in_lakhs'].quantile(0.01), df['price_in_lakhs'].quantile(0.99)
df = df[(df['price_in_lakhs'] >= Q1) & (df['price_in_lakhs'] <= Q3)]
df = df[(df['area'] >= 100) & (df['area'] <= 10000)]
df['price_per_sqft'] = (df['price_in_lakhs'] * 100000) / df['area']

# Region-level statistics
region_base = df.groupby('region').agg(
    avg_psf     =('price_per_sqft', 'median'),
    dom_type    =('type', lambda x: x.mode()[0]),
    ready_pct   =('status', lambda x: round((x == 'Ready to move').sum() / len(x) * 100, 1)),
    count       =('price_in_lakhs', 'count'),
    avg_price   =('price_in_lakhs', 'median'),
).reset_index()

for bhk in [1, 2, 3]:
    sub = df[df['bhk'] == bhk].groupby('region')['price_in_lakhs'].median().reset_index()
    sub.columns = ['region', f'bhk{bhk}']
    region_base = region_base.merge(sub, on='region', how='left')

region_base = region_base[region_base['count'] >= 5]
region_base[['bhk1','bhk2','bhk3']] = region_base[['bhk1','bhk2','bhk3']].fillna(0).round(2)
region_base['avg_psf']   = region_base['avg_psf'].round(0)
region_base['avg_price'] = region_base['avg_price'].round(2)

# Price bands (33rd / 66th percentile)
P33 = region_base['avg_psf'].quantile(0.33)
P66 = region_base['avg_psf'].quantile(0.66)

def color_band(psf):
    if psf < P33:  return 'green'
    if psf < P66:  return 'yellow'
    return 'blue'

region_base['color'] = region_base['avg_psf'].apply(color_band)
REGIONS = region_base.to_dict('records')
print(f"✅ {len(REGIONS)} regions loaded. Bands: green<₹{P33:.0f}, yellow<₹{P66:.0f}")

# ──────────────────────────────────────────
#  2. TRAIN ML MODEL (runs once on startup)
# ──────────────────────────────────────────
print("⏳ Loading trained model...")

model_data = pickle.load(open("model.pkl", "rb"))

model = model_data["model"]
region_enc_map = model_data["region_enc_map"]
type_enc_map = model_data["type_enc_map"]
status_enc_map = model_data["status_enc_map"]
age_enc_map = model_data["age_enc_map"]
region_avg_map = model_data["region_avg_map"]

print("✅ Model loaded!")

# Helper maps for encoding

# ──────────────────────────────────────────
#  3. API ROUTES
# ──────────────────────────────────────────

@app.route('/api/regions', methods=['GET'])
def get_regions():
    """Return all region stats + price band thresholds."""
    return jsonify({
        "bands": {"green": int(P33), "yellow": int(P66)},
        "regions": REGIONS
    })


@app.route('/api/budget', methods=['POST'])
def budget_finder():
    """
    Body: { "budget": 150, "bhk": "2" }
    Returns matching localities sorted by fit.
    """
    data   = request.get_json()
    budget = float(data.get('budget', 0))
    bhk    = str(data.get('bhk', 'any'))

    if budget <= 0:
        return jsonify({"error": "Budget must be > 0"}), 400

    results = []
    for r in REGIONS:
        if bhk == '1':   target = r['bhk1'] if r['bhk1'] > 0 else r['avg_price']
        elif bhk == '2': target = r['bhk2'] if r['bhk2'] > 0 else r['avg_price']
        elif bhk == '3': target = r['bhk3'] if r['bhk3'] > 0 else r['avg_price']
        else:            target = r['avg_price']

        if target <= 0:
            continue

        diff = target - budget
        pct  = diff / budget

        if diff <= 0:     match = 'perfect'
        elif pct <= 0.20: match = 'good'
        elif pct <= 0.35: match = 'stretch'
        else:             continue

        results.append({**r, "target_price": round(target, 2), "match": match})

    results.sort(key=lambda x: ({'perfect':0,'good':1,'stretch':2}[x['match']], abs(x['target_price'] - budget)))

    return jsonify({"count": len(results), "results": results})


@app.route('/api/predict', methods=['POST'])
def predict_price():
    """
    Body: {
      "bhk": 2, "area": 700, "region": "Andheri West",
      "type": "Apartment", "status": "Ready to move", "age": "New"
    }
    Returns predicted price in Lakhs.
    """
    data = request.get_json()

    region = data.get('region', '')
    ptype  = data.get('type', 'Apartment')
    status = data.get('status', 'Ready to move')
    age    = data.get('age', 'New')

    # Validate region
    if region not in region_enc_map:
        known = list(region_enc_map.keys())[:10]
        return jsonify({"error": f"Unknown region '{region}'. Examples: {known}"}), 400

    input_row = pd.DataFrame([{
        'bhk':        int(data.get('bhk', 2)),
        'area':       int(data.get('area', 700)),
        'region_enc': region_enc_map[region],
        'type_enc':   type_enc_map.get(ptype, 0),
        'status_enc': status_enc_map.get(status, 0),
        'age_enc':    age_enc_map.get(age, 0),
        'region_avg': region_avg_map.get(region, df['price_in_lakhs'].mean()),
    }])

    predicted_lakhs = float(model.predict(input_row)[0])
    predicted_cr    = predicted_lakhs / 100

    return jsonify({
        "region":          region,
        "bhk":             int(data.get('bhk', 2)),
        "area":            int(data.get('area', 700)),
        "type":            ptype,
        "status":          status,
        "age":             age,
        "predicted_lakhs": round(predicted_lakhs, 2),
        "predicted_cr":    round(predicted_cr, 3),
        "price_band":      color_band(
            predicted_lakhs * 100000 / int(data.get('area', 700))
        )
    })


@app.route('/api/regions/list', methods=['GET'])
def list_regions():
    """Return just the region names — useful for dropdown population."""
    return jsonify(sorted([r['region'] for r in REGIONS]))


@app.route('/')
def index():
    return """
    <h2>Mumbai Property API 🏠</h2>
    <ul>
      <li>GET  /api/regions         — All region stats + price bands</li>
      <li>GET  /api/regions/list    — Just region names</li>
      <li>POST /api/budget          — Budget-based locality finder</li>
      <li>POST /api/predict         — ML price prediction</li>
    </ul>
    """


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
