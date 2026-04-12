"""
PredictIQ - Data Simulator
Reads real datasets row by row, simulates live sensor feed
"""

import pandas as pd
import numpy as np
import random
import streamlit as st

@st.cache_data
def load_cnc_data():
    return pd.read_csv("data/ai4i2020.csv")

@st.cache_data
def load_aerospace_data():
    cols = ["engine_id","cycle"] + [f"set{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]
    df = pd.read_csv("data/CMaps/train_FD001.txt", sep=r"\s+", header=None, names=cols, engine="python")
    max_c = df.groupby("engine_id")["cycle"].max().reset_index()
    max_c.columns = ["engine_id","max_cycle"]
    df = df.merge(max_c, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    return df

@st.cache_data
def load_energy_data():
    df = pd.read_csv("data/T1.csv")
    df.columns = ["datetime","power","wind_speed","theoretical_power","wind_direction"]
    return df.dropna()

def get_reading(industry_key, component_key, tick, simulate_failure=False):
    drift = min(tick * 0.03, 20)
    if simulate_failure:
        drift = 60
    r = {"component": component_key, "tick": tick}

    if industry_key == "CNC Manufacturing":
        try:
            df = load_cnc_data()
            row = df.iloc[tick % len(df)]
            r["temperature"] = round(float(row["Process temperature [K]"]) + (drift*0.3 if simulate_failure else 0), 1)
            r["rpm"]         = int(row["Rotational speed [rpm]"])
            r["torque"]      = round(float(row["Torque [Nm]"]) + (drift*0.5 if simulate_failure else 0), 1)
            r["tool_wear"]   = min(int(row["Tool wear [min]"]) + int(drift*2), 280)
            r["vibration"]   = round(random.uniform(0.3,1.5) + drift*0.08, 2)
            r["_features"]   = [
                float(row["Air temperature [K]"]),
                float(row["Process temperature [K]"]),
                float(row["Rotational speed [rpm]"]),
                float(row["Torque [Nm]"]),
                float(row["Tool wear [min]"])
            ]
        except:
            r.update({"temperature":308,"rpm":1500,"torque":40,"tool_wear":100,"vibration":0.8})

    elif industry_key in ["Aerospace","Aeronautical"]:
        try:
            df = load_aerospace_data()
            row = df.iloc[tick % len(df)]
            rul = max(float(row["rul"]) - drift*2, 0) if simulate_failure else float(row["rul"])
            r["rul"]        = round(rul, 1)
            r["temperature"]= round(float(row["s4"]) + drift*3, 1)
            r["fan_speed"]  = round(2100 + random.uniform(-50,50) + drift*5, 0)
            r["pressure"]   = round(float(row["s9"]) + drift*2, 1)
            r["vibration"]  = round(random.uniform(0.3,1.2) + drift*0.06, 2)
            r["fuel_flow"]  = round(2.0 + random.uniform(0,0.2) + drift*0.02, 3)
            r["efficiency"] = max(round(98 - drift*0.5 - random.uniform(0,2), 1), 50)
            r["_features"]  = [float(row[f"s{i}"]) for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
        except:
            r.update({"rul":100,"temperature":1350,"fan_speed":2100,"pressure":550,
                       "vibration":0.8,"fuel_flow":2.0,"efficiency":95})

    elif industry_key == "Energy Sector":
        try:
            df = load_energy_data()
            row = df.iloc[tick % len(df)]
            r["wind_speed"]  = round(float(row["wind_speed"]), 1)
            r["power"]       = max(round(float(row["power"]) - drift*20, 0), 0) if simulate_failure else round(float(row["power"]), 0)
            r["temperature"] = round(35 + drift*0.5 + random.uniform(-2,2), 1)
            r["rotor_speed"] = round(12 + (float(row["wind_speed"])/25)*8 + random.uniform(-1,1), 1)
            r["vibration"]   = round(random.uniform(0.5,2.0) + drift*0.08, 2)
            r["current"]     = round(200 + (float(row["power"])/2000)*200 + random.uniform(-10,10), 1)
            r["efficiency"]  = max(round(92 - drift*0.4 - random.uniform(0,2), 1), 50)
            r["pitch_angle"] = round(random.uniform(5,15) + drift*0.3, 1)
            r["_features"]   = [float(row["wind_speed"]), float(row["theoretical_power"]),
                                float(row["wind_direction"]), float(row["power"])]
        except:
            r.update({"wind_speed":8,"power":800,"temperature":45,"rotor_speed":15,
                       "vibration":1.5,"current":250,"efficiency":88,"pitch_angle":10})
    return r

def has_changed(prev, curr, sensors):
    """Check if any sensor value has meaningfully changed."""
    if not prev:
        return True
    for s in sensors:
        key = s["key"]
        old = prev.get(key, 0)
        new = curr.get(key, 0)
        if abs(new - old) > 0.5:
            return True
    return False
