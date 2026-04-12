"""
PredictIQ - Industry & Component Configuration
"""

INDUSTRIES = {
    "CNC Manufacturing": {
        "icon": "⚙️", "color": "#7c4dff",
        "components": {
            "CNC Lathe": {
                "model_key": "cnc_lathe", "algorithm": "XGBoost", "icon": "🔩",
                "sensors": [
                    {"key":"temperature","label":"Process Temp","unit":"K","thresholds":(310,315,320),"inverse":False},
                    {"key":"rpm","label":"Spindle RPM","unit":"RPM","thresholds":(1500,2600,2800),"inverse":False},
                    {"key":"torque","label":"Torque","unit":"Nm","thresholds":(45,55,65),"inverse":False},
                    {"key":"tool_wear","label":"Tool Wear","unit":"min","thresholds":(150,180,220),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"Hz","thresholds":(2.0,3.5,5.0),"inverse":False},
                ],
                "dataset": "AI4I 2020",
            },
            "CNC Milling Machine": {
                "model_key": "cnc_milling", "algorithm": "Random Forest", "icon": "🔧",
                "sensors": [
                    {"key":"temperature","label":"Process Temp","unit":"K","thresholds":(310,315,320),"inverse":False},
                    {"key":"rpm","label":"Spindle RPM","unit":"RPM","thresholds":(1500,2600,2800),"inverse":False},
                    {"key":"torque","label":"Torque","unit":"Nm","thresholds":(45,55,65),"inverse":False},
                    {"key":"tool_wear","label":"Tool Wear","unit":"min","thresholds":(150,180,220),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"Hz","thresholds":(2.0,3.5,5.0),"inverse":False},
                ],
                "dataset": "AI4I 2020",
            },
            "CNC Drilling Machine": {
                "model_key": "cnc_drilling", "algorithm": "Gradient Boosting", "icon": "🪛",
                "sensors": [
                    {"key":"temperature","label":"Process Temp","unit":"K","thresholds":(310,315,320),"inverse":False},
                    {"key":"rpm","label":"Spindle RPM","unit":"RPM","thresholds":(1500,2600,2800),"inverse":False},
                    {"key":"torque","label":"Torque","unit":"Nm","thresholds":(45,55,65),"inverse":False},
                    {"key":"tool_wear","label":"Tool Wear","unit":"min","thresholds":(150,180,220),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"Hz","thresholds":(2.0,3.5,5.0),"inverse":False},
                ],
                "dataset": "AI4I 2020",
            },
        },
    },
    "Aerospace": {
        "icon": "🚀", "color": "#69f0ae",
        "components": {
            "Compressor": {
                "model_key": "aero_compressor", "algorithm": "Random Forest", "icon": "🌀",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"pressure","label":"Pressure P30","unit":"psia","thresholds":(600,650,700),"inverse":False},
                    {"key":"temperature","label":"Temp T30","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Turbine": {
                "model_key": "aero_turbine_gb", "algorithm": "Gradient Boosting", "icon": "🌪️",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"temperature","label":"Turbine Temp","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"fan_speed","label":"Core Speed N2","unit":"rpm","thresholds":(8000,9000,10000),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Fan": {
                "model_key": "aero_fan", "algorithm": "Random Forest", "icon": "💨",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"fan_speed","label":"Fan Speed N1","unit":"rpm","thresholds":(2200,2400,2600),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"temperature","label":"Inlet Temp T2","unit":"°R","thresholds":(600,700,800),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Combustion Chamber": {
                "model_key": "aero_combustion_xgb", "algorithm": "XGBoost", "icon": "🔥",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"temperature","label":"EGT Temp T50","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"fuel_flow","label":"Fuel Flow","unit":"pps","thresholds":(2.2,2.5,2.8),"inverse":False},
                    {"key":"pressure","label":"Pressure","unit":"psia","thresholds":(600,650,700),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
        },
    },
    "Aeronautical": {
        "icon": "✈️", "color": "#00bcd4",
        "components": {
            "Compressor": {
                "model_key": "aeron_compressor", "algorithm": "Random Forest", "icon": "🌀",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"pressure","label":"Pressure P30","unit":"psia","thresholds":(600,650,700),"inverse":False},
                    {"key":"temperature","label":"Temp T30","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Turbine": {
                "model_key": "aeron_turbine_gb", "algorithm": "Gradient Boosting", "icon": "🌪️",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"temperature","label":"Turbine Temp","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"fan_speed","label":"Core Speed N2","unit":"rpm","thresholds":(8000,9000,10000),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Fan": {
                "model_key": "aeron_fan", "algorithm": "Random Forest", "icon": "💨",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"fan_speed","label":"Fan Speed N1","unit":"rpm","thresholds":(2200,2400,2600),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(95,85,75),"inverse":True},
                    {"key":"temperature","label":"Inlet Temp","unit":"°R","thresholds":(600,700,800),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
            "Combustion Chamber": {
                "model_key": "aeron_combustion_xgb", "algorithm": "XGBoost", "icon": "🔥",
                "sensors": [
                    {"key":"rul","label":"Remaining Life","unit":"cycles","thresholds":(100,50,20),"inverse":True},
                    {"key":"temperature","label":"EGT Temp T50","unit":"°R","thresholds":(1400,1500,1600),"inverse":False},
                    {"key":"fuel_flow","label":"Fuel Flow","unit":"pps","thresholds":(2.2,2.5,2.8),"inverse":False},
                    {"key":"pressure","label":"Pressure","unit":"psia","thresholds":(600,650,700),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"g","thresholds":(1.5,2.5,3.5),"inverse":False},
                ],
                "dataset": "NASA CMAPSS",
            },
        },
    },
    "Energy Sector": {
        "icon": "⚡", "color": "#ffd740",
        "components": {
            "Gearbox": {
                "model_key": "energy_gearbox_gb", "algorithm": "Gradient Boosting", "icon": "⚙️",
                "sensors": [
                    {"key":"wind_speed","label":"Wind Speed","unit":"m/s","thresholds":(15,20,25),"inverse":False},
                    {"key":"vibration","label":"Vibration","unit":"mm/s","thresholds":(3,6,10),"inverse":False},
                    {"key":"temperature","label":"Gearbox Temp","unit":"°C","thresholds":(60,75,85),"inverse":False},
                    {"key":"power","label":"Power Output","unit":"kW","thresholds":(500,200,50),"inverse":True},
                    {"key":"rotor_speed","label":"Rotor Speed","unit":"RPM","thresholds":(18,22,26),"inverse":False},
                ],
                "dataset": "NREL SCADA",
            },
            "Generator": {
                "model_key": "energy_generator_xgb", "algorithm": "XGBoost", "icon": "🔌",
                "sensors": [
                    {"key":"wind_speed","label":"Wind Speed","unit":"m/s","thresholds":(15,20,25),"inverse":False},
                    {"key":"power","label":"Power Output","unit":"kW","thresholds":(500,200,50),"inverse":True},
                    {"key":"temperature","label":"Generator Temp","unit":"°C","thresholds":(60,75,85),"inverse":False},
                    {"key":"current","label":"Current","unit":"A","thresholds":(400,500,600),"inverse":False},
                    {"key":"efficiency","label":"Efficiency","unit":"%","thresholds":(90,80,70),"inverse":True},
                ],
                "dataset": "NREL SCADA",
            },
            "Blades": {
                "model_key": "energy_blades", "algorithm": "Random Forest", "icon": "🌬️",
                "sensors": [
                    {"key":"wind_speed","label":"Wind Speed","unit":"m/s","thresholds":(15,20,25),"inverse":False},
                    {"key":"vibration","label":"Blade Vibration","unit":"mm/s","thresholds":(3,6,10),"inverse":False},
                    {"key":"pitch_angle","label":"Pitch Angle","unit":"°","thresholds":(20,30,40),"inverse":False},
                    {"key":"power","label":"Power Output","unit":"kW","thresholds":(500,200,50),"inverse":True},
                    {"key":"rotor_speed","label":"Rotor Speed","unit":"RPM","thresholds":(18,22,26),"inverse":False},
                ],
                "dataset": "NREL SCADA",
            },
            "Bearings": {
                "model_key": "energy_bearings", "algorithm": "Random Forest", "icon": "🔩",
                "sensors": [
                    {"key":"vibration","label":"Vibration","unit":"mm/s","thresholds":(3,6,10),"inverse":False},
                    {"key":"temperature","label":"Bearing Temp","unit":"°C","thresholds":(60,75,85),"inverse":False},
                    {"key":"rotor_speed","label":"Rotor Speed","unit":"RPM","thresholds":(18,22,26),"inverse":False},
                    {"key":"wind_speed","label":"Wind Speed","unit":"m/s","thresholds":(15,20,25),"inverse":False},
                    {"key":"power","label":"Power Output","unit":"kW","thresholds":(500,200,50),"inverse":True},
                ],
                "dataset": "NREL SCADA",
            },
        },
    },
}

SC = {"good":"#00c853","warning":"#ffa726","bad":"#ef5350","critical":"#ff1744"}
SE = {"good":"✅","warning":"⚠️","bad":"❌","critical":"🚨"}
SL = {"good":"HEALTHY","warning":"WARNING","bad":"DEGRADED","critical":"CRITICAL"}
