"""
PredictIQ - Train All Dedicated Models
Run: python train_all.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

FEATURES_CNC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

FAILURE_COLS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

def build_cnc_label(row):
    for col, name in FAILURE_COLS.items():
        if row[col] == 1:
            return name
    return "No Failure"

def load_cnc():
    df = pd.read_csv("data/ai4i2020.csv")
    df["label"] = df.apply(build_cnc_label, axis=1)
    X = df[FEATURES_CNC].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_enc, scaler, le

def save_model(clf, scaler, le, name):
    pickle.dump(clf,    open(f"models/{name}_model.pkl",   "wb"))
    pickle.dump(scaler, open(f"models/{name}_scaler.pkl",  "wb"))
    pickle.dump(le,     open(f"models/{name}_encoder.pkl", "wb"))
    print(f"Saved: models/{name}_model.pkl")

def train_cnc_lathe():
    print("\n== CNC Lathe — XGBoost ==")
    X, y, scaler, le = load_cnc()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="mlogloss", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")
    save_model(clf, scaler, le, "cnc_lathe")

def train_cnc_milling():
    print("\n== CNC Milling — Random Forest ==")
    X, y, scaler, le = load_cnc()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")
    save_model(clf, scaler, le, "cnc_milling")

def train_cnc_drilling():
    print("\n== CNC Drilling — Gradient Boosting ==")
    X, y, scaler, le = load_cnc()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      subsample=0.8, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")
    save_model(clf, scaler, le, "cnc_drilling")

def load_aerospace():
    cols = ["engine_id","cycle"] + [f"set{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]
    df = pd.read_csv("data/CMaps/train_FD001.txt", sep=r"\s+", header=None, names=cols, engine="python")
    max_c = df.groupby("engine_id")["cycle"].max().reset_index()
    max_c.columns = ["engine_id","max_cycle"]
    df = df.merge(max_c, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    def label(r):
        if r > 100:  return "Good"
        elif r > 50: return "Warning"
        elif r > 20: return "Bad"
        else:        return "Critical"
    df["label"] = df["rul"].apply(label)
    FEATURES = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
    X = df[FEATURES].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_enc, scaler, le

def train_aerospace_component(name, model_key, random_state=42):
    print(f"\n== {name} ==")
    X, y, scaler, le = load_aerospace()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    if "xgb" in model_key:
        clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                            eval_metric="mlogloss", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif "gb" in model_key:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                          learning_rate=0.1, random_state=42)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")
    save_model(clf, scaler, le, model_key)

def load_energy():
    df = pd.read_csv("data/T1.csv")
    df.columns = ["datetime","power","wind_speed","theoretical_power","wind_direction"]
    df = df.dropna()
    def label(row):
        p, w = row["power"], row["wind_speed"]
        if p < 0 or (w > 3 and p < 100): return "Critical"
        elif p < 500 and w > 8:           return "Bad"
        elif p < 800 and w > 10:          return "Warning"
        else:                             return "Good"
    df["label"] = df.apply(label, axis=1)
    FEATURES = ["wind_speed","theoretical_power","wind_direction","power"]
    X = df[FEATURES].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_enc, scaler, le

def train_energy_component(name, model_key, random_state=42):
    print(f"\n== {name} ==")
    X, y, scaler, le = load_energy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    if "xgb" in model_key:
        clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                            eval_metric="mlogloss", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif "gb" in model_key:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                          learning_rate=0.1, random_state=42)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")
    save_model(clf, scaler, le, model_key)

if __name__ == "__main__":
    print("PredictIQ — Training All Dedicated Models")
    print("="*50)

    # CNC — 3 dedicated models
    train_cnc_lathe()
    train_cnc_milling()
    train_cnc_drilling()

    # Aerospace components — 4 models
    train_aerospace_component("Aerospace Compressor — LSTM sim via RF", "aero_compressor", 42)
    train_aerospace_component("Aerospace Turbine — CNN+LSTM sim via GB", "aero_turbine_gb", 10)
    train_aerospace_component("Aerospace Fan — 1D CNN sim via RF", "aero_fan", 20)
    train_aerospace_component("Aerospace Combustion — XGBoost", "aero_combustion_xgb", 30)

    # Aeronautical components — 4 models
    train_aerospace_component("Aeronautical Compressor", "aeron_compressor", 50)
    train_aerospace_component("Aeronautical Turbine", "aeron_turbine_gb", 60)
    train_aerospace_component("Aeronautical Fan", "aeron_fan", 70)
    train_aerospace_component("Aeronautical Combustion", "aeron_combustion_xgb", 80)

    # Energy components — 4 models
    train_energy_component("Energy Gearbox — CNN+LSTM sim via GB", "energy_gearbox_gb", 42)
    train_energy_component("Energy Generator — XGBoost", "energy_generator_xgb", 10)
    train_energy_component("Energy Blades — CNN sim via RF", "energy_blades", 20)
    train_energy_component("Energy Bearings — 1D CNN sim via RF", "energy_bearings", 30)

    print("\n" + "="*50)
    print("ALL MODELS TRAINED!")
    for f in sorted(os.listdir("models")):
        if "_model" in f:
            print(f"  {f}")
    print("\nRun: python -m streamlit run app.py")
