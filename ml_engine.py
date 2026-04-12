"""
PredictIQ - ML Engine
Loads and runs all 15 dedicated ML models
"""

import pickle
import numpy as np
import os

def load_all_models():
    models = {}
    if not os.path.exists("models"):
        return models
    model_files = [f.replace("_model.pkl","") for f in os.listdir("models") if f.endswith("_model.pkl")]
    for key in model_files:
        try:
            models[key] = {
                "clf":     pickle.load(open(f"models/{key}_model.pkl","rb")),
                "scaler":  pickle.load(open(f"models/{key}_scaler.pkl","rb")),
                "encoder": pickle.load(open(f"models/{key}_encoder.pkl","rb")),
            }
        except:
            pass
    return models

def ml_predict(reading, model_key, models):
    m = models.get(model_key)
    if not m or "_features" not in reading:
        return "Unknown", 0.0
    try:
        features = np.array([reading["_features"]])
        if features.shape[1] != m["scaler"].n_features_in_:
            return "Unknown", 0.0
        scaled = m["scaler"].transform(features)
        pred   = m["clf"].predict(scaled)[0]
        proba  = m["clf"].predict_proba(scaled)[0]
        conf   = round(float(max(proba)) * 100, 1)
        label  = m["encoder"].inverse_transform([pred])[0]
        return label, conf
    except:
        return "Unknown", 0.0

def get_status(value, sensor):
    w, b, c = sensor["thresholds"]
    if sensor["inverse"]:
        if value <= c:   return "critical"
        elif value <= b: return "bad"
        elif value <= w: return "warning"
        else:            return "good"
    else:
        if value >= c:   return "critical"
        elif value >= b: return "bad"
        elif value >= w: return "warning"
        else:            return "good"

def overall_status(reading, sensors):
    order = ["good","warning","bad","critical"]
    worst = "good"
    for s in sensors:
        st_ = get_status(reading.get(s["key"], 0), s)
        if order.index(st_) > order.index(worst):
            worst = st_
    return worst

def health_score(reading, sensors):
    scores = {"good":100,"warning":65,"bad":35,"critical":10}
    vals = [scores[get_status(reading.get(s["key"],0),s)] for s in sensors]
    return int(sum(vals)/len(vals))

def severity_score(reading, sensors, lam=0.1, k=1):
    total = 0
    for s in sensors:
        status = get_status(reading.get(s["key"],0), s)
        S   = {"good":0.1,"warning":0.4,"bad":0.7,"critical":1.0}[status]
        W   = 1/len(sensors)
        IRI = {"good":0.1,"warning":0.5,"bad":0.8,"critical":1.0}[status]
        total += (1-lam)**k * S * W * IRI
    return round(min(total*10, 10), 2)
