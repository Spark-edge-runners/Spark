============================================================
  PredictIQ — Complete System
============================================================

TWO WAYS TO RUN:

1. FRONTEND + API (Landing page with 3D models):
   pip install -r requirements.txt
   python api.py
   → Open http://localhost:5000

2. LIVE DASHBOARD (Real-time monitoring):
   python -m streamlit run app.py
   → Open http://localhost:8501

SETUP:
   - Open .env → paste your Groq API key
   - That's it! Models and datasets already included.

GET GROQ API KEY (FREE):
   console.groq.com → Create API Key → paste in .env

FOLDER STRUCTURE:
   api.py              Flask backend + frontend server
   app.py              Streamlit live dashboard
   config.py           Industry/component configuration
   ml_engine.py        ML model loader and predictor
   simulator.py        Real dataset simulator
   fault_database.py   Fault diagnosis database
   chatbot.py          Groq AI chatbot module
   train_all.py        Retrain all 15 models
   templates/          HTML frontend
   static/             CSS, JS, 3D model files
   data/               All 3 datasets
   models/             All 15 trained ML models
   .env                Your API key (never share!)
============================================================
