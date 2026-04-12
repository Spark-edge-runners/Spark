"""
PredictIQ - Flask API Backend
Run: python api.py  →  http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pickle, numpy as np, os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Load models ───────────────────────────────────────────
def load_model(name):
    try:
        clf    = pickle.load(open(f"models/{name}_model.pkl",   "rb"))
        scaler = pickle.load(open(f"models/{name}_scaler.pkl",  "rb"))
        enc    = pickle.load(open(f"models/{name}_encoder.pkl", "rb"))
        print(f"  ✅ {name}")
        return {"clf": clf, "scaler": scaler, "encoder": enc}
    except Exception as e:
        print(f"  ⚠️  {name}: {e}")
        return None

MODELS = {
    "cnc_lathe":          load_model("cnc_lathe"),
    "cnc_milling":        load_model("cnc_milling"),
    "cnc_drilling":       load_model("cnc_drilling"),
    "aero_compressor":    load_model("aero_compressor"),
    "aero_turbine":       load_model("aero_turbine_gb"),
    "aero_combustion":    load_model("aero_combustion_xgb"),
    "energy_gearbox":     load_model("energy_gearbox_gb"),
    "energy_generator":   load_model("energy_generator_xgb"),
    "energy_bearings":    load_model("energy_bearings"),
}

STATUS_MAP = {
    "No Failure":               {"status": "good",     "color": "#00c853"},
    "Heat Dissipation Failure": {"status": "warning",  "color": "#ffa726"},
    "Power Failure":            {"status": "bad",      "color": "#ef5350"},
    "Overstrain Failure":       {"status": "critical", "color": "#ff1744"},
    "Tool Wear Failure":        {"status": "critical", "color": "#ff1744"},
    "Random Failures":          {"status": "critical", "color": "#ff1744"},
    "Good":                     {"status": "good",     "color": "#00c853"},
    "Warning":                  {"status": "warning",  "color": "#ffa726"},
    "Bad":                      {"status": "bad",      "color": "#ef5350"},
    "Critical":                 {"status": "critical", "color": "#ff1744"},
}

def run_predict(model_key, features):
    m = MODELS.get(model_key)
    if not m:
        return {"error": f"Model '{model_key}' not loaded", "prediction": "Unknown", "confidence": 0, "status": "good"}
    try:
        X      = np.array([features])
        scaled = m["scaler"].transform(X)
        pred   = m["clf"].predict(scaled)[0]
        proba  = m["clf"].predict_proba(scaled)[0]
        label  = m["encoder"].inverse_transform([pred])[0]
        conf   = round(float(max(proba)) * 100, 1)
        info   = STATUS_MAP.get(label, {"status": "good", "color": "#00c853"})
        return {"prediction": label, "confidence": conf, "status": info["status"],
                "color": info["color"], "message": f"{label} ({conf}% confidence)"}
    except Exception as e:
        return {"error": str(e), "prediction": "Error", "confidence": 0, "status": "good"}

# ══════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard/cnc")
def cnc_dashboard():
    return render_template("cnc_dashboard.html")

@app.route("/dashboard/aerospace")
@app.route("/dashboard/aeronautical")
def aerospace_dashboard():
    return render_template("aerospace_dashboard.html")

@app.route("/dashboard/energy")
def energy_dashboard():
    return render_template("energy_dashboard.html")

# ══════════════════════════════════════════════════════════
# CNC PREDICT ENDPOINTS
# ══════════════════════════════════════════════════════════

def cnc_features(data):
    return [
        data.get("Air temperature [K]", 298.1),
        data.get("Process temperature [K]", 308.4),
        data.get("Rotational speed [rpm]", 1550),
        data.get("Torque [Nm]", 40.2),
        data.get("Tool wear [min]", 95),
    ]

@app.route("/predict/cnc/model", methods=["POST"])
def predict_cnc():
    return jsonify(run_predict("cnc_lathe", cnc_features(request.json)))

@app.route("/predict/cnc/lathe", methods=["POST"])
def predict_cnc_lathe():
    return jsonify(run_predict("cnc_lathe", cnc_features(request.json)))

@app.route("/predict/cnc/milling", methods=["POST"])
def predict_cnc_milling():
    return jsonify(run_predict("cnc_milling", cnc_features(request.json)))

@app.route("/predict/cnc/drilling", methods=["POST"])
def predict_cnc_drilling():
    return jsonify(run_predict("cnc_drilling", cnc_features(request.json)))

# ══════════════════════════════════════════════════════════
# AEROSPACE PREDICT ENDPOINTS
# ══════════════════════════════════════════════════════════

def aero_features(data):
    return [data.get(f"s{i}", 0) for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]

@app.route("/predict/aerospace/model", methods=["POST"])
@app.route("/predict/aerospace/compressor", methods=["POST"])
def predict_aero_compressor():
    return jsonify(run_predict("aero_compressor", aero_features(request.json)))

@app.route("/predict/aerospace/turbine", methods=["POST"])
def predict_aero_turbine():
    return jsonify(run_predict("aero_turbine", aero_features(request.json)))

@app.route("/predict/aerospace/combustion", methods=["POST"])
def predict_aero_combustion():
    return jsonify(run_predict("aero_combustion", aero_features(request.json)))

# ══════════════════════════════════════════════════════════
# ENERGY PREDICT ENDPOINTS
# ══════════════════════════════════════════════════════════

def energy_features(data):
    return [
        data.get("wind_speed", 8.8),
        data.get("theoretical_power", 920),
        data.get("wind_direction", 134),
        data.get("power", 850),
    ]

@app.route("/predict/energy/model", methods=["POST"])
@app.route("/predict/energy/gearbox", methods=["POST"])
def predict_energy_gearbox():
    return jsonify(run_predict("energy_gearbox", energy_features(request.json)))

@app.route("/predict/energy/generator", methods=["POST"])
def predict_energy_generator():
    return jsonify(run_predict("energy_generator", energy_features(request.json)))

@app.route("/predict/energy/bearings", methods=["POST"])
def predict_energy_bearings():
    return jsonify(run_predict("energy_bearings", energy_features(request.json)))

# ══════════════════════════════════════════════════════════
# CHAT ENDPOINT
# ══════════════════════════════════════════════════════════

@app.route("/chat", methods=["POST"])
def chat():
    import requests as req
    data      = request.json
    msg       = data.get("message", "")
    industry  = data.get("industry", "")
    component = data.get("component", "")
    status    = data.get("status", "good")
    reading   = data.get("reading", {})
    api_key   = os.getenv("GROQ_API_KEY", "")

    if not api_key or "paste" in api_key.lower():
        return jsonify({"response": "⚠️ Groq API key not set. Open .env and paste your key from console.groq.com"})

    sensor_text = "\n".join([f"  {k}: {v}" for k, v in reading.items() if k != "tick"])
    system = f"""You are PredictIQ, an expert AI maintenance assistant for {industry} - {component}.
Current status: {status.upper()}
Live sensor readings:
{sensor_text}

Rules:
- Be concise (2-3 sentences max)
- Give clear, actionable advice
- If critical/bad: immediate action steps
- If normal/warning: preventive advice"""

    try:
        r = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": msg}
            ], "max_tokens": 150, "temperature": 0.3},
            timeout=10
        )
        result = r.json()
        return jsonify({"response": result["choices"][0]["message"]["content"]})
    except Exception as e:
        return jsonify({"response": f"AI unavailable: {str(e)[:60]}"})

# ══════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════

@app.route("/health")
def health():
    loaded = [k for k, v in MODELS.items() if v is not None]
    return jsonify({"status": "running", "models_loaded": len(loaded), "models": loaded})

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🔧 PredictIQ API Server")
    print("=" * 40)
    loaded = [k for k, v in MODELS.items() if v is not None]
    print(f"✅ {len(loaded)}/9 models loaded")
    print("=" * 40)
    print("→ http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, port=5000)