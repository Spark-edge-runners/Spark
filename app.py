"""
PredictIQ - Multi-Industry Predictive Maintenance Platform
Run: python -m streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import datetime
import random
import io
from dotenv import load_dotenv

load_dotenv()

from config import INDUSTRIES, SC, SE, SL
from ml_engine import load_all_models, ml_predict, get_status, overall_status, health_score, severity_score
from simulator import get_reading, has_changed
from fault_database import get_faults_for_component, get_preventive_tips
from chatbot import chat_with_groq, get_auto_message

st.set_page_config(page_title="PredictIQ", page_icon="🔧", layout="wide")

# ── REQUIRED PARAMETERS PER MACHINE ──────────────────────────────────────────
# Maps component name → list of {key, label, unit, default, min_val, max_val, help}
MANUAL_INPUT_FIELDS = {
    "CNC Lathe": [
        {"key": "Air temperature [K]",          "label": "Air Temperature",     "unit": "K",   "default": 298.1, "min_val": 280.0,  "max_val": 340.0,  "help": "Ambient air temperature around the machine"},
        {"key": "Process temperature [K]",       "label": "Process Temperature", "unit": "K",   "default": 308.4, "min_val": 290.0,  "max_val": 360.0,  "help": "Temperature at the cutting zone / spindle area"},
        {"key": "Rotational speed [rpm]",        "label": "Spindle RPM",         "unit": "RPM", "default": 1550,  "min_val": 100.0,  "max_val": 3000.0, "help": "Rotational speed of the lathe spindle"},
        {"key": "Torque [Nm]",                   "label": "Torque",              "unit": "Nm",  "default": 40.2,  "min_val": 1.0,    "max_val": 100.0,  "help": "Torque applied at the cutting tool"},
        {"key": "Tool wear [min]",               "label": "Tool Wear",           "unit": "min", "default": 95,    "min_val": 0.0,    "max_val": 280.0,  "help": "Cumulative tool usage time in minutes"},
    ],
    "CNC Milling Machine": [
        {"key": "Air temperature [K]",          "label": "Air Temperature",     "unit": "K",   "default": 298.1, "min_val": 280.0,  "max_val": 340.0,  "help": "Ambient air temperature around the machine"},
        {"key": "Process temperature [K]",       "label": "Process Temperature", "unit": "K",   "default": 308.4, "min_val": 290.0,  "max_val": 360.0,  "help": "Temperature at the milling zone"},
        {"key": "Rotational speed [rpm]",        "label": "Spindle RPM",         "unit": "RPM", "default": 1550,  "min_val": 100.0,  "max_val": 3000.0, "help": "Milling spindle rotational speed"},
        {"key": "Torque [Nm]",                   "label": "Torque",              "unit": "Nm",  "default": 40.2,  "min_val": 1.0,    "max_val": 100.0,  "help": "Torque at the milling cutter"},
        {"key": "Tool wear [min]",               "label": "Tool Wear",           "unit": "min", "default": 95,    "min_val": 0.0,    "max_val": 280.0,  "help": "Accumulated cutter wear time"},
    ],
    "CNC Drilling Machine": [
        {"key": "Air temperature [K]",          "label": "Air Temperature",     "unit": "K",   "default": 298.1, "min_val": 280.0,  "max_val": 340.0,  "help": "Ambient air temperature"},
        {"key": "Process temperature [K]",       "label": "Process Temperature", "unit": "K",   "default": 308.4, "min_val": 290.0,  "max_val": 360.0,  "help": "Temperature at the drill tip zone"},
        {"key": "Rotational speed [rpm]",        "label": "Drill RPM",           "unit": "RPM", "default": 1550,  "min_val": 100.0,  "max_val": 3000.0, "help": "Drill bit rotational speed"},
        {"key": "Torque [Nm]",                   "label": "Torque",              "unit": "Nm",  "default": 40.2,  "min_val": 1.0,    "max_val": 100.0,  "help": "Drilling torque"},
        {"key": "Tool wear [min]",               "label": "Tool Wear",           "unit": "min", "default": 95,    "min_val": 0.0,    "max_val": 280.0,  "help": "Drill bit wear time"},
    ],
    "Compressor": [
        {"key": "s2",  "label": "Fan Inlet Temp (T2)",        "unit": "°R",   "default": 518.67, "min_val": 400.0,  "max_val": 700.0,  "help": "Total temperature at fan inlet"},
        {"key": "s3",  "label": "LPC Outlet Temp (T24)",      "unit": "°R",   "default": 642.68, "min_val": 500.0,  "max_val": 900.0,  "help": "Total temperature at LPC outlet"},
        {"key": "s4",  "label": "HPC Outlet Temp (T30)",      "unit": "°R",   "default": 1583.3, "min_val": 1000.0, "max_val": 2000.0, "help": "Total temperature at HPC outlet"},
        {"key": "s7",  "label": "HPC Outlet Pressure (P30)",  "unit": "psia", "default": 553.74, "min_val": 300.0,  "max_val": 800.0,  "help": "Total pressure at HPC outlet"},
        {"key": "s8",  "label": "Fan Speed (Nf)",             "unit": "rpm",  "default": 2388.1, "min_val": 1500.0, "max_val": 3000.0, "help": "Physical fan speed"},
        {"key": "s9",  "label": "Core Speed (Nc)",            "unit": "rpm",  "default": 9059.9, "min_val": 7000.0, "max_val": 11000.0,"help": "Physical core speed"},
        {"key": "s11", "label": "HPC Inlet Static Temp",      "unit": "°R",   "default": 392.14, "min_val": 300.0,  "max_val": 600.0,  "help": "Static temperature at HPC inlet"},
        {"key": "s12", "label": "Fuel Flow (Wf)",             "unit": "pps",  "default": 521.66, "min_val": 200.0,  "max_val": 800.0,  "help": "Ratio of fuel flow to Ps30"},
        {"key": "s13", "label": "Corrected Fan Speed",        "unit": "rpm",  "default": 2387.9, "min_val": 1500.0, "max_val": 3000.0, "help": "Corrected fan speed"},
        {"key": "s14", "label": "Corrected Core Speed",       "unit": "rpm",  "default": 8138.6, "min_val": 6000.0, "max_val": 10000.0,"help": "Corrected core speed"},
        {"key": "s15", "label": "Bypass Ratio",               "unit": "-",    "default": 8.4195, "min_val": 5.0,    "max_val": 12.0,   "help": "Bypass ratio"},
        {"key": "s17", "label": "Bleed Enthalpy",             "unit": "-",    "default": 0.3987, "min_val": 0.1,    "max_val": 0.7,    "help": "Bleed enthalpy"},
        {"key": "s20", "label": "HPT Coolant Bleed",          "unit": "-",    "default": 39.06,  "min_val": 20.0,   "max_val": 60.0,   "help": "HPT coolant bleed"},
        {"key": "s21", "label": "LPT Coolant Bleed",          "unit": "-",    "default": 23.42,  "min_val": 10.0,   "max_val": 40.0,   "help": "LPT coolant bleed"},
    ],
    "Turbine": [
        {"key": "s2",  "label": "Fan Inlet Temp (T2)",        "unit": "°R",   "default": 518.67, "min_val": 400.0,  "max_val": 700.0,  "help": "Total temperature at fan inlet"},
        {"key": "s3",  "label": "LPC Outlet Temp (T24)",      "unit": "°R",   "default": 642.68, "min_val": 500.0,  "max_val": 900.0,  "help": "Total temperature at LPC outlet"},
        {"key": "s4",  "label": "HPC Outlet Temp (T30)",      "unit": "°R",   "default": 1583.3, "min_val": 1000.0, "max_val": 2000.0, "help": "Total temperature at HPC outlet"},
        {"key": "s7",  "label": "HPC Outlet Pressure (P30)",  "unit": "psia", "default": 553.74, "min_val": 300.0,  "max_val": 800.0,  "help": "Total pressure at HPC outlet"},
        {"key": "s8",  "label": "Fan Speed (Nf)",             "unit": "rpm",  "default": 2388.1, "min_val": 1500.0, "max_val": 3000.0, "help": "Physical fan speed"},
        {"key": "s9",  "label": "Core Speed (Nc)",            "unit": "rpm",  "default": 9059.9, "min_val": 7000.0, "max_val": 11000.0,"help": "Physical core speed"},
        {"key": "s11", "label": "HPC Inlet Static Temp",      "unit": "°R",   "default": 392.14, "min_val": 300.0,  "max_val": 600.0,  "help": "Static temperature at HPC inlet"},
        {"key": "s12", "label": "Fuel Flow (Wf)",             "unit": "pps",  "default": 521.66, "min_val": 200.0,  "max_val": 800.0,  "help": "Ratio of fuel flow to Ps30"},
        {"key": "s13", "label": "Corrected Fan Speed",        "unit": "rpm",  "default": 2387.9, "min_val": 1500.0, "max_val": 3000.0, "help": "Corrected fan speed"},
        {"key": "s14", "label": "Corrected Core Speed",       "unit": "rpm",  "default": 8138.6, "min_val": 6000.0, "max_val": 10000.0,"help": "Corrected core speed"},
        {"key": "s15", "label": "Bypass Ratio",               "unit": "-",    "default": 8.4195, "min_val": 5.0,    "max_val": 12.0,   "help": "Bypass ratio"},
        {"key": "s17", "label": "Bleed Enthalpy",             "unit": "-",    "default": 0.3987, "min_val": 0.1,    "max_val": 0.7,    "help": "Bleed enthalpy"},
        {"key": "s20", "label": "HPT Coolant Bleed",          "unit": "-",    "default": 39.06,  "min_val": 20.0,   "max_val": 60.0,   "help": "HPT coolant bleed"},
        {"key": "s21", "label": "LPT Coolant Bleed",          "unit": "-",    "default": 23.42,  "min_val": 10.0,   "max_val": 40.0,   "help": "LPT coolant bleed"},
    ],
    "Fan": [
        {"key": "s2",  "label": "Fan Inlet Temp (T2)",        "unit": "°R",   "default": 518.67, "min_val": 400.0,  "max_val": 700.0,  "help": "Total temperature at fan inlet"},
        {"key": "s3",  "label": "LPC Outlet Temp (T24)",      "unit": "°R",   "default": 642.68, "min_val": 500.0,  "max_val": 900.0,  "help": "Total temperature at LPC outlet"},
        {"key": "s4",  "label": "HPC Outlet Temp (T30)",      "unit": "°R",   "default": 1583.3, "min_val": 1000.0, "max_val": 2000.0, "help": "HPC outlet temperature"},
        {"key": "s7",  "label": "HPC Outlet Pressure (P30)",  "unit": "psia", "default": 553.74, "min_val": 300.0,  "max_val": 800.0,  "help": "Pressure at HPC outlet"},
        {"key": "s8",  "label": "Fan Speed (Nf)",             "unit": "rpm",  "default": 2388.1, "min_val": 1500.0, "max_val": 3000.0, "help": "Physical fan speed"},
        {"key": "s9",  "label": "Core Speed (Nc)",            "unit": "rpm",  "default": 9059.9, "min_val": 7000.0, "max_val": 11000.0,"help": "Physical core speed"},
        {"key": "s11", "label": "HPC Inlet Static Temp",      "unit": "°R",   "default": 392.14, "min_val": 300.0,  "max_val": 600.0,  "help": "Static temp at HPC inlet"},
        {"key": "s12", "label": "Fuel Flow (Wf)",             "unit": "pps",  "default": 521.66, "min_val": 200.0,  "max_val": 800.0,  "help": "Fuel flow ratio"},
        {"key": "s13", "label": "Corrected Fan Speed",        "unit": "rpm",  "default": 2387.9, "min_val": 1500.0, "max_val": 3000.0, "help": "Corrected fan speed"},
        {"key": "s14", "label": "Corrected Core Speed",       "unit": "rpm",  "default": 8138.6, "min_val": 6000.0, "max_val": 10000.0,"help": "Corrected core speed"},
        {"key": "s15", "label": "Bypass Ratio",               "unit": "-",    "default": 8.4195, "min_val": 5.0,    "max_val": 12.0,   "help": "Bypass ratio"},
        {"key": "s17", "label": "Bleed Enthalpy",             "unit": "-",    "default": 0.3987, "min_val": 0.1,    "max_val": 0.7,    "help": "Bleed enthalpy"},
        {"key": "s20", "label": "HPT Coolant Bleed",          "unit": "-",    "default": 39.06,  "min_val": 20.0,   "max_val": 60.0,   "help": "HPT coolant bleed"},
        {"key": "s21", "label": "LPT Coolant Bleed",          "unit": "-",    "default": 23.42,  "min_val": 10.0,   "max_val": 40.0,   "help": "LPT coolant bleed"},
    ],
    "Combustion Chamber": [
        {"key": "s2",  "label": "Fan Inlet Temp (T2)",        "unit": "°R",   "default": 518.67, "min_val": 400.0,  "max_val": 700.0,  "help": "Total temperature at fan inlet"},
        {"key": "s3",  "label": "LPC Outlet Temp (T24)",      "unit": "°R",   "default": 642.68, "min_val": 500.0,  "max_val": 900.0,  "help": "LPC outlet temperature"},
        {"key": "s4",  "label": "EGT Temp (T50)",             "unit": "°R",   "default": 1583.3, "min_val": 1000.0, "max_val": 2000.0, "help": "Exhaust gas temperature — key health indicator"},
        {"key": "s7",  "label": "HPC Outlet Pressure (P30)",  "unit": "psia", "default": 553.74, "min_val": 300.0,  "max_val": 800.0,  "help": "Pressure at HPC outlet"},
        {"key": "s8",  "label": "Fan Speed (Nf)",             "unit": "rpm",  "default": 2388.1, "min_val": 1500.0, "max_val": 3000.0, "help": "Physical fan speed"},
        {"key": "s9",  "label": "Core Speed (Nc)",            "unit": "rpm",  "default": 9059.9, "min_val": 7000.0, "max_val": 11000.0,"help": "Physical core speed"},
        {"key": "s11", "label": "HPC Inlet Static Temp",      "unit": "°R",   "default": 392.14, "min_val": 300.0,  "max_val": 600.0,  "help": "Static temp at HPC inlet"},
        {"key": "s12", "label": "Fuel Flow (Wf)",             "unit": "pps",  "default": 521.66, "min_val": 200.0,  "max_val": 800.0,  "help": "Fuel flow ratio to Ps30"},
        {"key": "s13", "label": "Corrected Fan Speed",        "unit": "rpm",  "default": 2387.9, "min_val": 1500.0, "max_val": 3000.0, "help": "Corrected fan speed"},
        {"key": "s14", "label": "Corrected Core Speed",       "unit": "rpm",  "default": 8138.6, "min_val": 6000.0, "max_val": 10000.0,"help": "Corrected core speed"},
        {"key": "s15", "label": "Bypass Ratio",               "unit": "-",    "default": 8.4195, "min_val": 5.0,    "max_val": 12.0,   "help": "Bypass ratio"},
        {"key": "s17", "label": "Bleed Enthalpy",             "unit": "-",    "default": 0.3987, "min_val": 0.1,    "max_val": 0.7,    "help": "Bleed enthalpy"},
        {"key": "s20", "label": "HPT Coolant Bleed",          "unit": "-",    "default": 39.06,  "min_val": 20.0,   "max_val": 60.0,   "help": "HPT coolant bleed"},
        {"key": "s21", "label": "LPT Coolant Bleed",          "unit": "-",    "default": 23.42,  "min_val": 10.0,   "max_val": 40.0,   "help": "LPT coolant bleed"},
    ],
    "Gearbox": [
        {"key": "wind_speed",        "label": "Wind Speed",          "unit": "m/s", "default": 8.8,  "min_val": 0.0,  "max_val": 30.0,   "help": "Current wind speed measured at hub height"},
        {"key": "theoretical_power", "label": "Theoretical Power",   "unit": "kW",  "default": 920,  "min_val": 0.0,  "max_val": 3500.0, "help": "Expected power output at current wind speed"},
        {"key": "wind_direction",    "label": "Wind Direction",      "unit": "°",   "default": 134,  "min_val": 0.0,  "max_val": 360.0,  "help": "Wind direction in degrees from north"},
        {"key": "power",             "label": "Actual Power Output", "unit": "kW",  "default": 850,  "min_val": 0.0,  "max_val": 3500.0, "help": "Actual measured power output"},
    ],
    "Generator": [
        {"key": "wind_speed",        "label": "Wind Speed",          "unit": "m/s", "default": 8.8,  "min_val": 0.0,  "max_val": 30.0,   "help": "Current wind speed at hub height"},
        {"key": "theoretical_power", "label": "Theoretical Power",   "unit": "kW",  "default": 920,  "min_val": 0.0,  "max_val": 3500.0, "help": "Expected power at current wind conditions"},
        {"key": "wind_direction",    "label": "Wind Direction",      "unit": "°",   "default": 134,  "min_val": 0.0,  "max_val": 360.0,  "help": "Wind direction in degrees"},
        {"key": "power",             "label": "Actual Power Output", "unit": "kW",  "default": 850,  "min_val": 0.0,  "max_val": 3500.0, "help": "Measured generator output"},
    ],
    "Blades": [
        {"key": "wind_speed",        "label": "Wind Speed",          "unit": "m/s", "default": 8.8,  "min_val": 0.0,  "max_val": 30.0,   "help": "Current wind speed at hub height"},
        {"key": "theoretical_power", "label": "Theoretical Power",   "unit": "kW",  "default": 920,  "min_val": 0.0,  "max_val": 3500.0, "help": "Expected power at current wind speed"},
        {"key": "wind_direction",    "label": "Wind Direction",      "unit": "°",   "default": 134,  "min_val": 0.0,  "max_val": 360.0,  "help": "Wind direction in degrees"},
        {"key": "power",             "label": "Actual Power Output", "unit": "kW",  "default": 850,  "min_val": 0.0,  "max_val": 3500.0, "help": "Measured output power"},
    ],
    "Bearings": [
        {"key": "wind_speed",        "label": "Wind Speed",          "unit": "m/s", "default": 8.8,  "min_val": 0.0,  "max_val": 30.0,   "help": "Current wind speed"},
        {"key": "theoretical_power", "label": "Theoretical Power",   "unit": "kW",  "default": 920,  "min_val": 0.0,  "max_val": 3500.0, "help": "Expected power output"},
        {"key": "wind_direction",    "label": "Wind Direction",      "unit": "°",   "default": 134,  "min_val": 0.0,  "max_val": 360.0,  "help": "Wind direction in degrees"},
        {"key": "power",             "label": "Actual Power Output", "unit": "kW",  "default": 850,  "min_val": 0.0,  "max_val": 3500.0, "help": "Actual measured power"},
    ],
}

# ── Map component keys to feature extraction function (matching api.py) ──────
def build_features_from_manual(component, values):
    """Convert manually entered values dict to _features list for ml_predict."""
    if component in ["CNC Lathe", "CNC Milling Machine", "CNC Drilling Machine"]:
        return [
            values.get("Air temperature [K]", 298.1),
            values.get("Process temperature [K]", 308.4),
            values.get("Rotational speed [rpm]", 1550),
            values.get("Torque [Nm]", 40.2),
            values.get("Tool wear [min]", 95),
        ]
    elif component in ["Compressor", "Turbine", "Fan", "Combustion Chamber"]:
        return [values.get(f"s{i}", 0) for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    elif component in ["Gearbox", "Generator", "Blades", "Bearings"]:
        return [
            values.get("wind_speed", 8.8),
            values.get("theoretical_power", 920),
            values.get("wind_direction", 134),
            values.get("power", 850),
        ]
    return []

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Base & Layout */
.main { padding-top: 0rem }
.block-container { padding-top: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; max-width: 1400px }
* { font-family: 'Segoe UI', system-ui, sans-serif }

/* Cards */
.stat-card { background: #1a1a2e; border-radius: 12px; padding: 18px 16px; border: 1px solid #2e2e4e; text-align: center }
.stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px }
.stat-value { font-size: 34px; font-weight: 800; line-height: 1.1 }
.stat-sub { font-size: 11px; color: #555; margin-top: 6px }

/* Status colors */
.c-good     { color: #00c853 }
.c-warning  { color: #ffa726 }
.c-bad      { color: #ef5350 }
.c-critical { color: #ff1744 }

/* Sensor row */
.sensor-row { background: #1a1a2e; border-radius: 8px; padding: 12px 16px; border: 1px solid #2e2e4e; margin-bottom: 7px; display: flex; justify-content: space-between; align-items: center }
.sensor-name { font-size: 13px; color: #aaa; flex: 1 }
.sensor-val { font-size: 18px; font-weight: 700; min-width: 90px; text-align: right }
.sensor-unit { font-size: 12px; color: #555; margin-left: 6px; min-width: 40px }
.sensor-badge { font-size: 11px; padding: 3px 10px; border-radius: 10px; margin-left: 10px; font-weight: 600; white-space: nowrap }
.badge-good     { background: #0a2e0a; color: #00c853 }
.badge-warning  { background: #2e1f0a; color: #ffa726 }
.badge-bad      { background: #2e0a0a; color: #ef5350 }
.badge-critical { background: #1a0000; color: #ff1744 }

/* Chat */
.msg-auto { background: #1a1a2e; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #7c4dff; font-size: 13px; color: #ddd; line-height: 1.6 }
.msg-user { background: #0d2137; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #00bcd4; font-size: 13px; color: #ddd; line-height: 1.6 }
.msg-ai   { background: #0a2e0a; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #00c853; font-size: 13px; color: #ddd; line-height: 1.6 }
.msg-time { font-size: 10px; color: #555; margin-top: 5px }

/* Fault card */
.fault-card { background: #1a1a2e; border-radius: 8px; padding: 14px; border: 1px solid #2e2e4e; margin-bottom: 10px }
.fault-title { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 8px }
.fault-row { font-size: 12px; color: #aaa; margin: 4px 0; line-height: 1.5 }
.fault-label { color: #888; font-size: 11px }

/* Home / component cards */
.home-card { background: #1a1a2e; border-radius: 16px; padding: 26px; text-align: center; border: 1px solid #2e2e4e; margin-bottom: 14px }
.comp-card { background: #1a1a2e; border-radius: 12px; padding: 18px; text-align: center; border: 1px solid #2e2e4e; margin-bottom: 10px }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #12121f; border-radius: 8px; padding: 4px }
.stTabs [data-baseweb="tab"] { color: #888; border-radius: 6px; font-size: 13px }
.stTabs [aria-selected="true"] { background: #1a1a2e; color: #fff }

/* Top bar */
.topbar { background: #12121f; border-radius: 12px; padding: 14px 20px; border: 1px solid #2e2e4e; margin-bottom: 14px }

/* RUL bar */
.rul-wrap { background: #1a1a2e; border-radius: 8px; padding: 14px 16px; border: 1px solid #2e2e4e; margin-bottom: 10px }
.rul-track { background: #2e2e4e; border-radius: 4px; height: 10px; margin-top: 10px }

/* Alert item */
.alert-item { background: #1a1a2e; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; border-left: 3px solid #2e2e4e; font-size: 13px; color: #aaa; line-height: 1.5 }

/* Input section styling */
.input-section { background: #12121f; border-radius: 14px; padding: 22px; border: 1px solid #2e2e4e; margin-bottom: 16px }
.input-header { font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 4px }
.input-sub { font-size: 12px; color: #888; margin-bottom: 16px }
.field-group { background: #1a1a2e; border-radius: 10px; padding: 16px; border: 1px solid #2e2e4e; margin-bottom: 10px }
.field-label { font-size: 12px; font-weight: 600; color: #ccc; margin-bottom: 4px; display: flex; justify-content: space-between }
.field-unit { font-size: 11px; color: #7c4dff; background: #1e1e3e; padding: 1px 8px; border-radius: 10px }
.field-help { font-size: 11px; color: #555; margin-top: 2px }

/* Result card */
.result-card { border-radius: 14px; padding: 22px; border: 2px solid; text-align: center; margin-bottom: 16px }
.result-prediction { font-size: 24px; font-weight: 800; margin: 8px 0 4px }
.result-confidence { font-size: 14px; margin-bottom: 6px }

/* Mode selector */
.mode-btn { padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; border: 1px solid #2e2e4e; transition: all .2s }

/* CSV preview */
.csv-table { font-size: 12px; width: 100% }
.csv-result-row { padding: 8px 12px; border-radius: 6px; margin-bottom: 4px; font-size: 13px }

/* Stacked bar for CSV */
.csv-summary-bar { display: flex; border-radius: 8px; overflow: hidden; height: 28px; margin: 8px 0 }
.csv-bar-segment { display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #fff; min-width: 30px }

/* Section divider */
.section-divider { border: none; border-top: 1px solid #2e2e4e; margin: 14px 0 }
</style>
""", unsafe_allow_html=True)

# ── Load models once ─────────────────────────────────────
@st.cache_resource
def get_models():
    return load_all_models()

MODELS = get_models()

# ── Session state ────────────────────────────────────────
for k,v in [
    ("page","home"),("industry",None),("component",None),
    ("tick",0),("history",[]),("alerts",[]),
    ("running",True),("simulate_failure",False),
    ("chat_messages",[]),("prev_reading",None),
    ("auto_msg",""),("tip_index",0),
    ("input_mode","select"),   # "select", "live", "manual", "csv"
    ("manual_result",None),    # result dict from manual entry
    ("csv_results",None),      # list of result dicts from CSV
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div style='text-align:center;padding:30px 0 20px'>
        <div style='font-size:52px'>🔧</div>
        <div style='font-size:34px;font-weight:900;color:#fff;margin:10px 0 6px;letter-spacing:-0.5px'>PredictIQ</div>
        <div style='font-size:15px;color:#888'>Multi-Industry Predictive Maintenance Platform</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    cols = st.columns(2)
    for i,(name,cfg) in enumerate(INDUSTRIES.items()):
        with cols[i%2]:
            comps = " · ".join(list(cfg["components"].keys()))
            n = len(cfg["components"])
            st.markdown(f"""
            <div class='home-card'>
                <div style='font-size:48px'>{cfg['icon']}</div>
                <div style='font-size:19px;font-weight:700;color:#fff;margin:12px 0 6px'>{name}</div>
                <div style='font-size:12px;color:#888;line-height:1.6'>{comps}</div>
                <div style='margin-top:12px'>
                    <span style='background:#1e1e3e;color:#7c4dff;font-size:11px;padding:3px 12px;border-radius:10px'>{n} ML Models</span>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"{cfg['icon']} Enter {name}", key=f"h_{name}", use_container_width=True):
                st.session_state.industry = name
                st.session_state.page = "component"
                st.session_state.tick = 0
                st.session_state.history = []
                st.session_state.alerts = []
                st.session_state.chat_messages = []
                st.session_state.simulate_failure = False
                st.session_state.input_mode = "select"
                st.session_state.manual_result = None
                st.session_state.csv_results = None
                st.rerun()
            st.markdown("")

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Industries","4")
    with c2: st.metric("ML Models","15")
    with c3: st.metric("Components","15")
    with c4: st.metric("Datasets","3")

# ══════════════════════════════════════════════════════════
# COMPONENT SELECT
# ══════════════════════════════════════════════════════════
def show_component():
    cfg = INDUSTRIES[st.session_state.industry]
    if st.button("← Back to Home"):
        st.session_state.page = "home"; st.rerun()

    st.markdown(f"## {cfg['icon']} {st.session_state.industry}")
    st.markdown("#### Select a component to diagnose")
    st.divider()

    cols = st.columns(2)
    for i,(comp_name,comp_cfg) in enumerate(cfg["components"].items()):
        with cols[i%2]:
            m = MODELS.get(comp_cfg["model_key"])
            dot = "🟢" if m else "🔴"
            st.markdown(f"""
            <div class='comp-card'>
                <div style='font-size:38px'>{comp_cfg['icon']}</div>
                <div style='font-size:16px;font-weight:700;color:#fff;margin:10px 0 4px'>{comp_name}</div>
                <div style='font-size:12px;color:#888'>{comp_cfg['algorithm']}</div>
                <div style='font-size:11px;color:#555;margin-top:6px'>{dot} {comp_cfg['dataset']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Select {comp_name}", key=f"c_{comp_name}", use_container_width=True):
                st.session_state.component = comp_name
                st.session_state.page = "input_select"
                st.session_state.input_mode = "select"
                st.session_state.manual_result = None
                st.session_state.csv_results = None
                st.rerun()
            st.markdown("")

# ══════════════════════════════════════════════════════════
# INPUT MODE SELECTION  (new page)
# ══════════════════════════════════════════════════════════
def show_input_select():
    ik   = st.session_state.industry
    ck   = st.session_state.component
    cfg  = INDUSTRIES[ik]
    comp = cfg["components"][ck]

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = "component"; st.rerun()

    st.markdown(f"""
    <div style='background:#12121f;border-radius:14px;padding:24px 28px;border:1px solid #2e2e4e;margin-bottom:20px'>
        <div style='font-size:22px;font-weight:800;color:#fff;margin-bottom:4px'>
            {comp['icon']} {ck}
            <span style='font-size:13px;font-weight:400;color:#888;margin-left:10px'>{cfg['icon']} {ik}</span>
        </div>
        <div style='font-size:13px;color:#888'>Algorithm: <span style='color:#7c4dff'>{comp['algorithm']}</span>
        &nbsp;·&nbsp; Dataset: <span style='color:#aaa'>{comp['dataset']}</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📥 How would you like to provide sensor data?")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='input-section' style='text-align:center;cursor:pointer'>
            <div style='font-size:40px;margin-bottom:12px'>🔴</div>
            <div style='font-size:16px;font-weight:700;color:#fff'>Live Simulation</div>
            <div style='font-size:12px;color:#888;margin-top:6px;line-height:1.6'>
                Real-time sensor feed from<br>industry datasets with live charts<br>and auto ML predictions
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("▶ Start Live Monitor", use_container_width=True, key="mode_live"):
            st.session_state.tick = 0
            st.session_state.history = []
            st.session_state.alerts = []
            st.session_state.chat_messages = []
            st.session_state.simulate_failure = False
            st.session_state.prev_reading = None
            st.session_state.page = "dashboard"
            st.rerun()

    with c2:
        st.markdown("""
        <div class='input-section' style='text-align:center;cursor:pointer'>
            <div style='font-size:40px;margin-bottom:12px'>✏️</div>
            <div style='font-size:16px;font-weight:700;color:#fff'>Manual Entry</div>
            <div style='font-size:12px;color:#888;margin-top:6px;line-height:1.6'>
                Enter individual sensor values<br>manually and get an instant<br>ML diagnosis result
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("✏️ Enter Data Manually", use_container_width=True, key="mode_manual"):
            st.session_state.page = "manual_input"
            st.rerun()

    with c3:
        st.markdown("""
        <div class='input-section' style='text-align:center;cursor:pointer'>
            <div style='font-size:40px;margin-bottom:12px'>📂</div>
            <div style='font-size:16px;font-weight:700;color:#fff'>CSV File Upload</div>
            <div style='font-size:12px;color:#888;margin-top:6px;line-height:1.6'>
                Upload a CSV with multiple<br>readings and get batch predictions<br>with a summary report
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("📂 Upload CSV File", use_container_width=True, key="mode_csv"):
            st.session_state.page = "csv_input"
            st.rerun()

# ══════════════════════════════════════════════════════════
# MANUAL INPUT PAGE  (new page)
# ══════════════════════════════════════════════════════════
def show_manual_input():
    ik   = st.session_state.industry
    ck   = st.session_state.component
    cfg  = INDUSTRIES[ik]
    comp = cfg["components"][ck]
    sensors = comp["sensors"]

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = "input_select"; st.rerun()

    # Header
    st.markdown(f"""
    <div class='topbar'>
        <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
            <div>
                <span style='font-size:20px;font-weight:800;color:#fff'>✏️ Manual Sensor Entry</span>
                <span style='color:#555;margin:0 10px'>|</span>
                <span style='color:#aaa;font-size:14px'>{cfg['icon']} {ik} › {comp['icon']} {ck}</span>
            </div>
            <div style='font-size:12px;color:#888'>Enter sensor readings to get an instant prediction</div>
        </div>
    </div>""", unsafe_allow_html=True)

    fields = MANUAL_INPUT_FIELDS.get(ck, [])
    if not fields:
        st.warning(f"Manual entry not yet configured for {ck}. Please use Live Simulation mode.")
        return

    left_col, right_col = st.columns([1.05, 0.95])

    with left_col:
        st.markdown(f"""
        <div class='input-section'>
            <div class='input-header'>🔢 Sensor Parameters for {ck}</div>
            <div class='input-sub'>Enter the sensor readings from your machine. All fields are required.</div>
        </div>""", unsafe_allow_html=True)

        values = {}
        half = (len(fields) + 1) // 2
        col_a, col_b = st.columns(2)

        for i, field in enumerate(fields):
            target_col = col_a if i < half else col_b
            with target_col:
                st.markdown(f"""
                <div style='margin-bottom:4px'>
                    <span style='font-size:12px;font-weight:600;color:#ccc'>{field['label']}</span>
                    <span style='font-size:11px;color:#7c4dff;background:#1e1e3e;padding:1px 8px;border-radius:10px;margin-left:6px'>{field['unit']}</span>
                </div>
                <div style='font-size:11px;color:#555;margin-bottom:4px'>{field['help']}</div>
                """, unsafe_allow_html=True)
                val = st.number_input(
                    f"{field['label']} ({field['unit']})",
                    min_value=float(field["min_val"]),
                    max_value=float(field["max_val"]),
                    value=float(field["default"]),
                    step=float((field["max_val"] - field["min_val"]) / 200),
                    key=f"manual_{field['key']}",
                    label_visibility="collapsed"
                )
                values[field["key"]] = val

        st.markdown("")
        predict_btn = st.button("🔍 Analyse & Predict", use_container_width=True, type="primary")

        if predict_btn:
            features = build_features_from_manual(ck, values)
            reading_for_ml = {**values, "_features": features, "component": ck, "tick": 0}
            # Also add derived sensor display keys
            if ck in ["CNC Lathe", "CNC Milling Machine", "CNC Drilling Machine"]:
                reading_for_ml["temperature"] = values.get("Process temperature [K]", 308.4)
                reading_for_ml["rpm"]         = values.get("Rotational speed [rpm]", 1550)
                reading_for_ml["torque"]      = values.get("Torque [Nm]", 40.2)
                reading_for_ml["tool_wear"]   = values.get("Tool wear [min]", 95)
                reading_for_ml["vibration"]   = 0.8
            elif ck in ["Compressor","Turbine","Fan","Combustion Chamber"]:
                reading_for_ml["temperature"] = values.get("s4", 1583.3)
                reading_for_ml["pressure"]    = values.get("s9", 9059.9)
                reading_for_ml["fan_speed"]   = values.get("s8", 2388.1)
                reading_for_ml["vibration"]   = 0.8
                reading_for_ml["efficiency"]  = 95.0
                reading_for_ml["fuel_flow"]   = 2.0
                reading_for_ml["rul"]         = 120.0
            elif ck in ["Gearbox","Generator","Blades","Bearings"]:
                reading_for_ml["power"]       = values.get("power", 850)
                reading_for_ml["wind_speed"]  = values.get("wind_speed", 8.8)
                reading_for_ml["temperature"] = 45.0
                reading_for_ml["vibration"]   = 1.0
                reading_for_ml["efficiency"]  = 90.0
                reading_for_ml["rotor_speed"] = 14.0
                reading_for_ml["current"]     = 220.0
                reading_for_ml["pitch_angle"] = 10.0

            ml_pred, conf = ml_predict(reading_for_ml, comp["model_key"], MODELS)
            status = overall_status(reading_for_ml, sensors)
            score  = health_score(reading_for_ml, sensors)
            sev    = severity_score(reading_for_ml, sensors)
            faults = get_faults_for_component(ik, ck)
            tips   = get_preventive_tips(ik, ck)

            st.session_state.manual_result = {
                "ml_pred": ml_pred, "conf": conf, "status": status,
                "score": score, "sev": sev, "reading": reading_for_ml,
                "faults": faults, "tips": tips, "values": values,
            }
            st.rerun()

    with right_col:
        result = st.session_state.manual_result

        if not result:
            st.markdown("""
            <div style='background:#12121f;border-radius:14px;border:1px dashed #2e2e4e;padding:40px 24px;text-align:center;margin-top:8px'>
                <div style='font-size:40px;margin-bottom:12px'>📊</div>
                <div style='font-size:15px;color:#888'>Enter sensor values on the left<br>and click <b style="color:#7c4dff">Analyse & Predict</b><br>to see the ML diagnosis here.</div>
            </div>""", unsafe_allow_html=True)
        else:
            status = result["status"]
            sc = SC[status]
            ml_pred = result["ml_pred"]
            conf    = result["conf"]
            score   = result["score"]
            sev     = result["sev"]

            # Big result card
            st.markdown(f"""
            <div class='result-card' style='background:{sc}11;border-color:{sc}55'>
                <div style='font-size:13px;color:{sc};text-transform:uppercase;letter-spacing:.1em;font-weight:700'>{SE[status]} {SL[status]}</div>
                <div class='result-prediction' style='color:{sc}'>{ml_pred}</div>
                <div class='result-confidence' style='color:#aaa'>ML Confidence: <b style='color:{sc}'>{conf}%</b> · {comp['algorithm']}</div>
            </div>""", unsafe_allow_html=True)

            # Score cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>Health Score</div>
                    <div class='stat-value c-{status}'>{score}</div>
                    <div class='stat-sub'>out of 100</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>Severity</div>
                    <div class='stat-value c-{status}'>{sev}</div>
                    <div class='stat-sub'>out of 10</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>Confidence</div>
                    <div class='stat-value' style='color:#7c4dff'>{conf}%</div>
                    <div class='stat-sub'>{comp['algorithm']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Sensor readings vs thresholds
            st.markdown("<div style='font-size:12px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>📡 Sensor Status Breakdown</div>", unsafe_allow_html=True)
            reading = result["reading"]
            for sensor in sensors:
                val = reading.get(sensor["key"], 0)
                s   = get_status(val, sensor)
                color = SC[s]
                st.markdown(f"""
                <div class='sensor-row'>
                    <span class='sensor-name'>{sensor['label']}</span>
                    <span class='sensor-val' style='color:{color}'>{round(val, 2)}</span>
                    <span class='sensor-unit'>{sensor['unit']}</span>
                    <span class='sensor-badge badge-{s}'>{SE[s]} {s.upper()}</span>
                </div>""", unsafe_allow_html=True)

            # Fault diagnosis or tips
            if status in ["bad","critical"] and result["faults"]:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:12px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>⚠️ Likely Faults & Repairs</div>", unsafe_allow_html=True)
                for f in result["faults"][:3]:
                    diff_color = {"Easy":"#00c853","Medium":"#ffa726","Hard":"#ef5350","Expert":"#ff1744"}.get(f['difficulty'],"#888")
                    st.markdown(f"""
                    <div class='fault-card'>
                        <div class='fault-title'>🔴 {f['fault']}</div>
                        <div class='fault-row'><span class='fault-label'>Causes: </span>{f['causes'].replace(';',' · ')}</div>
                        <div class='fault-row'><span class='fault-label'>Repair: </span>{f['repair'].replace(';',' → ')}</div>
                        <div style='margin-top:8px'>
                            <span style='background:{diff_color}22;color:{diff_color};font-size:11px;padding:2px 10px;border-radius:8px'>{f['difficulty']}</span>
                            <span style='color:#555;font-size:12px;margin-left:8px'>⏱ {f['time_min']} min</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            elif result["tips"]:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:12px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px'>💡 Preventive Maintenance Tips</div>", unsafe_allow_html=True)
                for tip in result["tips"][:4]:
                    st.markdown(f"<div class='msg-auto'>💡 {tip}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CSV INPUT PAGE  (new page)
# ══════════════════════════════════════════════════════════
def show_csv_input():
    ik   = st.session_state.industry
    ck   = st.session_state.component
    cfg  = INDUSTRIES[ik]
    comp = cfg["components"][ck]
    sensors = comp["sensors"]

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = "input_select"; st.rerun()

    st.markdown(f"""
    <div class='topbar'>
        <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
            <div>
                <span style='font-size:20px;font-weight:800;color:#fff'>📂 CSV Batch Analysis</span>
                <span style='color:#555;margin:0 10px'>|</span>
                <span style='color:#aaa;font-size:14px'>{cfg['icon']} {ik} › {comp['icon']} {ck}</span>
            </div>
            <div style='font-size:12px;color:#888'>Upload a CSV to analyse multiple readings at once</div>
        </div>
    </div>""", unsafe_allow_html=True)

    fields = MANUAL_INPUT_FIELDS.get(ck, [])

    # Show expected columns
    expected_cols = [f['key'] for f in fields]
    col_preview = ", ".join(expected_cols)

    st.markdown(f"""
    <div class='input-section'>
        <div class='input-header'>📋 Expected CSV Columns for {ck}</div>
        <div class='input-sub'>Your CSV must have these column headers (order doesn't matter):</div>
        <div style='background:#0d0d1a;border-radius:8px;padding:12px 16px;font-family:monospace;font-size:12px;color:#7c4dff;border:1px solid #2e2e4e;word-break:break-all;line-height:1.8'>
            {col_preview}
        </div>
    </div>""", unsafe_allow_html=True)

    # Download sample CSV button
    sample_data = {f["key"]: [f["default"]] * 3 for f in fields}
    sample_df = pd.DataFrame(sample_data)
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download Sample CSV Template for {ck}",
        data=csv_bytes,
        file_name=f"predictiq_sample_{ck.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.markdown("")
    uploaded = st.file_uploader(
        f"Upload your CSV file",
        type=["csv"],
        key=f"csv_upload_{ck}",
        help="Upload a CSV file with sensor readings"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f"<div style='font-size:13px;color:#888;margin-bottom:8px'>✅ Loaded <b style='color:#fff'>{len(df)}</b> rows × <b style='color:#fff'>{len(df.columns)}</b> columns</div>", unsafe_allow_html=True)

            # Check missing columns
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.info(f"Found columns: {', '.join(df.columns.tolist())}")
                return

            # Show preview
            with st.expander("👁️ Preview uploaded data (first 5 rows)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

            if st.button("🔍 Run Batch Analysis", use_container_width=True, type="primary"):
                results = []
                progress = st.progress(0, text="Analysing rows...")
                n = len(df)

                for idx, row in df.iterrows():
                    values = {f["key"]: float(row.get(f["key"], f["default"])) for f in fields}
                    features = build_features_from_manual(ck, values)
                    reading_for_ml = {**values, "_features": features, "component": ck, "tick": idx}

                    # Populate display sensor keys
                    if ck in ["CNC Lathe", "CNC Milling Machine", "CNC Drilling Machine"]:
                        reading_for_ml["temperature"] = values.get("Process temperature [K]", 308.4)
                        reading_for_ml["rpm"]         = values.get("Rotational speed [rpm]", 1550)
                        reading_for_ml["torque"]      = values.get("Torque [Nm]", 40.2)
                        reading_for_ml["tool_wear"]   = values.get("Tool wear [min]", 95)
                        reading_for_ml["vibration"]   = 0.8
                    elif ck in ["Compressor","Turbine","Fan","Combustion Chamber"]:
                        reading_for_ml["temperature"] = values.get("s4", 1583.3)
                        reading_for_ml["pressure"]    = values.get("s9", 9059.9)
                        reading_for_ml["fan_speed"]   = values.get("s8", 2388.1)
                        reading_for_ml["vibration"]   = 0.8
                        reading_for_ml["efficiency"]  = 95.0
                        reading_for_ml["fuel_flow"]   = 2.0
                        reading_for_ml["rul"]         = 120.0
                    elif ck in ["Gearbox","Generator","Blades","Bearings"]:
                        reading_for_ml["power"]       = values.get("power", 850)
                        reading_for_ml["wind_speed"]  = values.get("wind_speed", 8.8)
                        reading_for_ml["temperature"] = 45.0
                        reading_for_ml["vibration"]   = 1.0
                        reading_for_ml["efficiency"]  = 90.0
                        reading_for_ml["rotor_speed"] = 14.0
                        reading_for_ml["current"]     = 220.0
                        reading_for_ml["pitch_angle"] = 10.0

                    ml_pred, ml_conf = ml_predict(reading_for_ml, comp["model_key"], MODELS)
                    status  = overall_status(reading_for_ml, sensors)
                    score   = health_score(reading_for_ml, sensors)

                    results.append({
                        "row": idx + 1,
                        "prediction": ml_pred,
                        "confidence": ml_conf,
                        "status": status,
                        "health_score": score,
                        "values": values,
                    })
                    progress.progress((idx + 1) / n, text=f"Analysing row {idx+1}/{n}...")

                progress.empty()
                st.session_state.csv_results = results
                st.rerun()

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    # Show batch results
    if st.session_state.csv_results:
        results = st.session_state.csv_results
        n_total    = len(results)
        counts = {"good":0,"warning":0,"bad":0,"critical":0}
        for r in results:
            counts[r["status"]] = counts.get(r["status"],0) + 1

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px;font-weight:700;color:#fff;margin-bottom:14px'>📊 Batch Results — {n_total} readings analysed</div>", unsafe_allow_html=True)

        # Summary metric cards
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Total Rows</div>
                <div class='stat-value' style='color:#fff'>{n_total}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Healthy</div>
                <div class='stat-value c-good'>{counts['good']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Warning</div>
                <div class='stat-value c-warning'>{counts['warning']}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Degraded</div>
                <div class='stat-value c-bad'>{counts['bad']}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Critical</div>
                <div class='stat-value c-critical'>{counts['critical']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Visual summary bar
        if n_total > 0:
            segs = ""
            for s, label, color in [("good","Healthy","#00c853"),("warning","Warn","#ffa726"),("bad","Degrad","#ef5350"),("critical","Crit","#ff1744")]:
                pct = counts[s] / n_total * 100
                if pct > 0:
                    segs += f"<div class='csv-bar-segment' style='background:{color};width:{pct:.1f}%'>{counts[s]}</div>"
            st.markdown(f"<div class='csv-summary-bar'>{segs}</div>", unsafe_allow_html=True)

        # Health score trend chart
        rows_x    = [r["row"]          for r in results]
        scores_y  = [r["health_score"] for r in results]
        bar_col   = [SC[r["status"]]   for r in results]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rows_x, y=scores_y,
            marker_color=bar_col,
            text=[f"{s}" for s in scores_y],
            textposition="outside",
            textfont=dict(size=10, color="#aaa"),
        ))
        fig.update_layout(
            title=dict(text="Health Score per Reading", font=dict(color="#aaa", size=14)),
            margin=dict(l=10, r=10, t=40, b=30),
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Row", color="#888", gridcolor="#2e2e4e", tickfont=dict(size=11)),
            yaxis=dict(title="Health Score", range=[0, 115], color="#888", gridcolor="#2e2e4e"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-row result table
        st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px'>Detailed results per row:</div>", unsafe_allow_html=True)
        table_data = []
        for r in results:
            s = r["status"]
            table_data.append({
                "Row": r["row"],
                "Prediction": r["prediction"],
                "Confidence (%)": r["confidence"],
                "Status": SL[s],
                "Health Score": r["health_score"],
            })
        table_df = pd.DataFrame(table_data)

        def style_row(val):
            mapping = {"HEALTHY":"color:#00c853","WARNING":"color:#ffa726","DEGRADED":"color:#ef5350","CRITICAL":"color:#ff1744"}
            return mapping.get(val,"")

        st.dataframe(
            table_df.style.applymap(style_row, subset=["Status"]),
            use_container_width=True,
            height=min(400, 40 + len(table_df) * 36),
        )

        # Download results CSV
        result_csv = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results CSV",
            data=result_csv,
            file_name=f"predictiq_results_{ck.lower().replace(' ','_')}.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════
# LIVE DASHBOARD  (unchanged)
# ══════════════════════════════════════════════════════════
def show_dashboard():
    ik   = st.session_state.industry
    ck   = st.session_state.component
    cfg  = INDUSTRIES[ik]
    comp = cfg["components"][ck]
    sensors = comp["sensors"]

    new_reading = get_reading(ik, ck, st.session_state.tick, st.session_state.simulate_failure)
    changed = has_changed(st.session_state.prev_reading, new_reading, sensors)

    if changed:
        st.session_state.prev_reading = new_reading
        st.session_state.history.append(new_reading)
        if len(st.session_state.history) > 60:
            st.session_state.history = st.session_state.history[-60:]

    reading  = new_reading
    status   = overall_status(reading, sensors)
    score    = health_score(reading, sensors)
    sev      = severity_score(reading, sensors)
    ml_pred, conf = ml_predict(reading, comp["model_key"], MODELS)
    rul_val  = reading.get("rul", None)
    faults   = get_faults_for_component(ik, ck)
    tips     = get_preventive_tips(ik, ck)

    auto_msg = get_auto_message(ik, ck, status, reading, sensors)

    if not st.session_state.chat_messages or st.session_state.chat_messages[-1]["content"] != auto_msg:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_messages.append({
            "role": "auto", "content": auto_msg, "time": ts
        })
        if len(st.session_state.chat_messages) > 50:
            st.session_state.chat_messages = st.session_state.chat_messages[-50:]

    # TOP BAR
    st.markdown(f"""
    <div class='topbar'>
        <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
            <div>
                <span style='font-size:20px;font-weight:800;color:#fff'>🔧 PredictIQ</span>
                <span style='color:#555;margin:0 10px'>|</span>
                <span style='color:#aaa;font-size:14px'>{cfg['icon']} {ik}</span>
                <span style='color:#555;margin:0 10px'>›</span>
                <span style='color:#fff;font-size:14px'>{comp['icon']} {ck}</span>
            </div>
            <div>
                <span style='background:{SC[status]}22;color:{SC[status]};font-size:13px;font-weight:700;padding:6px 18px;border-radius:20px;border:1px solid {SC[status]}55'>
                    {SE[status]} {SL[status]}
                </span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>Health Score</div>
                <div class='stat-value c-{status}'>{score}</div>
                <div class='stat-sub'>out of 100</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>VEHMS Severity</div>
                <div class='stat-value c-{status}'>{sev}</div>
                <div class='stat-sub'>out of 10</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>ML Confidence</div>
                <div class='stat-value' style='color:#7c4dff'>{conf}%</div>
                <div class='stat-sub'>{comp['algorithm']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        cc1,cc2,cc3,cc4 = st.columns(4)
        with cc1:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = "input_select"; st.rerun()
        with cc2:
            if st.button("⏸ Pause" if st.session_state.running else "▶ Resume", use_container_width=True):
                st.session_state.running = not st.session_state.running
        with cc3:
            fail_lbl = "🔴 Stop Sim" if st.session_state.simulate_failure else "💥 Simulate Fail"
            if st.button(fail_lbl, use_container_width=True):
                st.session_state.simulate_failure = not st.session_state.simulate_failure
                st.session_state.tick = 0
                st.rerun()
        with cc4:
            if st.button("✏️ Manual Entry", use_container_width=True):
                st.session_state.page = "manual_input"
                st.rerun()

        st.markdown("")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sensors", "📈 Trends", "🔔 Alerts", "🔬 Diagnostics"])

        with tab1:
            for sensor in sensors:
                val = reading.get(sensor["key"], 0)
                s   = get_status(val, sensor)
                color = SC[s]
                st.markdown(f"""
                <div class='sensor-row'>
                    <span class='sensor-name'>{sensor['label']}</span>
                    <span class='sensor-val' style='color:{color}'>{round(val,2)}</span>
                    <span class='sensor-unit'>{sensor['unit']}</span>
                    <span class='sensor-badge badge-{s}'>{SE[s]} {s.upper()}</span>
                </div>""", unsafe_allow_html=True)

            if rul_val is not None:
                rul_pct = min(rul_val/150*100, 100)
                rul_color = SC[get_status(rul_val, {"thresholds":(100,50,20),"inverse":True})]
                st.markdown(f"""
                <div class='rul-wrap'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='font-size:13px;color:#888'>Remaining Useful Life</span>
                        <span style='font-size:15px;font-weight:700;color:{rul_color}'>{rul_val} cycles</span>
                    </div>
                    <div class='rul-track'>
                        <div style='background:{rul_color};width:{rul_pct:.0f}%;height:10px;border-radius:4px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab2:
            primary = sensors[0]
            if len(st.session_state.history) > 1:
                vals   = [r.get(primary["key"],0) for r in st.session_state.history[-40:]]
                colors = [SC[get_status(v,primary)] for v in vals]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(vals))), y=vals,
                    mode="lines+markers",
                    line=dict(color=cfg["color"],width=2),
                    marker=dict(size=5,color=colors),
                    fill="tozeroy",fillcolor=cfg["color"]+"18"))
                w = primary["thresholds"][0]
                fig.add_hline(y=w, line_dash="dash", line_color="#ffa726",
                              annotation_text="Warning",annotation_font_color="#ffa726")
                fig.update_layout(
                    title=dict(text=f"{primary['label']} trend",font=dict(color="#aaa",size=13)),
                    margin=dict(l=10,r=10,t=36,b=10),height=230,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="#2e2e4e",color="#888",title=primary['unit']),
                    xaxis=dict(gridcolor="#2e2e4e",color="#888",title="Sample"),
                    showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            sensor_names  = [s["label"] for s in sensors]
            sensor_stress = [{"good":0.1,"warning":0.5,"bad":0.75,"critical":1.0}[get_status(reading.get(s["key"],0),s)] for s in sensors]
            bar_colors    = [SC[get_status(reading.get(s["key"],0),s)] for s in sensors]
            fig2 = go.Figure(go.Bar(
                x=sensor_stress, y=sensor_names, orientation="h",
                marker_color=bar_colors,
                text=[f"{round(reading.get(s['key'],0),1)} {s['unit']}" for s in sensors],
                textposition="inside",
                textfont=dict(size=11)))
            fig2.update_layout(
                title=dict(text="Sensor stress level",font=dict(color="#aaa",size=13)),
                margin=dict(l=10,r=10,t=36,b=10),height=200,
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0,1],showgrid=False,color="#888",title="Stress Level"),
                yaxis=dict(color="#888"),showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            if status != "good":
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                alert = {"time":ts,"status":status,"component":ck,
                         "message":auto_msg.replace("✅","").replace("⚠️","").replace("❌","").replace("🚨","").strip()}
                if not st.session_state.alerts or st.session_state.alerts[-1]["message"] != alert["message"]:
                    st.session_state.alerts.append(alert)
                    if len(st.session_state.alerts) > 30:
                        st.session_state.alerts = st.session_state.alerts[-30:]

            if st.session_state.alerts:
                for a in reversed(st.session_state.alerts[-15:]):
                    bc = SC.get(a["status"],"#888")
                    st.markdown(f"""
                    <div class='alert-item' style='border-left-color:{bc}'>
                        {SE.get(a['status'],'⚠️')} &nbsp;
                        <span style='color:#555'>{a['time']}</span> &nbsp;
                        <span style='color:#7c4dff'>[{a['component']}]</span> &nbsp;
                        {a['message']}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No alerts yet — component running normally.")

        with tab4:
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:8px;padding:14px 18px;border:1px solid #2e2e4e;margin-bottom:12px'>
                <div style='font-size:12px;color:#888;margin-bottom:10px;text-transform:uppercase;letter-spacing:.06em'>Model Info</div>
                <div style='font-size:13px;color:#fff;margin-bottom:5px'>Algorithm: <span style='color:{cfg["color"]}'>{comp['algorithm']}</span></div>
                <div style='font-size:13px;color:#fff;margin-bottom:5px'>Prediction: <span style='color:{SC[status]}'>{ml_pred}</span></div>
                <div style='font-size:13px;color:#fff;margin-bottom:5px'>Confidence: <span style='color:{cfg["color"]}'>{conf}%</span></div>
                <div style='font-size:13px;color:#fff'>Dataset: <span style='color:#888'>{comp['dataset']}</span></div>
            </div>""", unsafe_allow_html=True)

            fig3 = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={"axis":{"range":[0,100],"tickcolor":"#888"},
                       "bar":{"color":SC[status]},
                       "steps":[{"range":[0,30],"color":"#2e0a0a"},
                                {"range":[30,60],"color":"#2e1f0a"},
                                {"range":[60,85],"color":"#1a2e0a"},
                                {"range":[85,100],"color":"#0a2e0a"}]},
                number={"font":{"color":SC[status],"size":32}}))
            fig3.update_layout(height=210,margin=dict(l=20,r=20,t=20,b=0),
                               paper_bgcolor="rgba(0,0,0,0)",font={"color":"#888"})
            st.plotly_chart(fig3, use_container_width=True)

    with right_col:
        status_color = SC[status]

        st.markdown(f"""
        <div style='background:#12121f;border-radius:12px;border:1px solid #2e2e4e;overflow:hidden;margin-bottom:12px'>
            <div style='background:#1a1a2e;padding:14px 18px;border-bottom:1px solid #2e2e4e'>
                <div style='font-size:14px;font-weight:700;color:#fff'>🤖 PredictIQ Assistant</div>
                <div style='font-size:12px;color:#888;margin-top:3px'>
                    Live AI for {ck} · Status:
                    <span style='color:{status_color};font-weight:700'>{SL[status]}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages[-8:]:
                role = msg["role"]
                content = msg["content"]
                ts = msg.get("time","")
                if role == "auto":
                    st.markdown(f"""<div class='msg-auto'>🤖 {content}<div class='msg-time'>{ts}</div></div>""", unsafe_allow_html=True)
                elif role == "user":
                    st.markdown(f"""<div class='msg-user'>👤 {content}<div class='msg-time'>{ts}</div></div>""", unsafe_allow_html=True)
                elif role == "assistant":
                    st.markdown(f"""<div class='msg-ai'>🤖 {content}<div class='msg-time'>{ts}</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        if status in ["bad","critical"] and faults:
            st.markdown("<div style='font-size:12px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin:8px 0 6px'>⚠️ Fault Diagnosis</div>", unsafe_allow_html=True)
            for f in faults[:3]:
                diff_color = {"Easy":"#00c853","Medium":"#ffa726","Hard":"#ef5350","Expert":"#ff1744"}.get(f['difficulty'],"#888")
                st.markdown(f"""
                <div class='fault-card'>
                    <div class='fault-title'>🔴 {f['fault']}</div>
                    <div class='fault-row'><span class='fault-label'>Causes:</span> {f['causes'].replace(';',' · ')}</div>
                    <div class='fault-row'><span class='fault-label'>Repair:</span> {f['repair'].replace(';',' → ')}</div>
                    <div style='margin-top:8px'>
                        <span style='background:{diff_color}22;color:{diff_color};font-size:11px;padding:2px 10px;border-radius:8px'>{f['difficulty']}</span>
                        <span style='color:#555;font-size:12px;margin-left:8px'>⏱ {f['time_min']} min</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        elif status == "warning":
            st.markdown("<div style='font-size:12px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin:8px 0 6px'>💡 Preventive Tips</div>", unsafe_allow_html=True)
            tip_idx = st.session_state.tip_index % len(tips) if tips else 0
            for tip in tips[tip_idx:tip_idx+3]:
                st.markdown(f"<div class='msg-auto'>💡 {tip}</div>", unsafe_allow_html=True)
        else:
            if tips:
                tip = tips[st.session_state.tick % len(tips)]
                st.markdown(f"<div class='msg-auto'>💡 Tip: {tip}</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("<div style='font-size:12px;color:#888;margin-bottom:4px'>Ask the AI assistant:</div>", unsafe_allow_html=True)
        user_input = st.text_input(
            "Message",
            placeholder="e.g. Should I stop the machine?",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.tick}"
        )
        send_col, clear_col = st.columns([3,1])
        with send_col:
            send = st.button("Send →", use_container_width=True)
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        if send and user_input.strip():
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_messages.append({"role":"user","content":user_input,"time":ts})
            history = [
                {"role": m["role"] if m["role"] in ["user","assistant"] else "assistant",
                 "content": m["content"]}
                for m in st.session_state.chat_messages[-10:]
                if m["role"] in ["user","assistant"]
            ]
            history.append({"role":"user","content":user_input})
            response = chat_with_groq(history, ik, ck, status, reading, sensors)
            st.session_state.chat_messages.append({
                "role":"assistant","content":response,
                "time":datetime.datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

    if st.session_state.running:
        st.session_state.tick += 1
        time.sleep(2)
        st.rerun()

# ── ROUTER ───────────────────────────────────────────────
page = st.session_state.page
if   page == "home":         show_home()
elif page == "component":    show_component()
elif page == "input_select": show_input_select()
elif page == "manual_input": show_manual_input()
elif page == "csv_input":    show_csv_input()
elif page == "dashboard":    show_dashboard()
