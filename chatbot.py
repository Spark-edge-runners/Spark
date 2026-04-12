"""
PredictIQ - AI Chatbot Module
Uses Groq API with LLaMA 3 model
"""

import os
import requests
import json
from fault_database import get_faults_for_component, get_preventive_tips

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

def build_system_prompt(industry, component, status, reading, sensors):
    sensor_text = "\n".join([
        f"  - {s['label']}: {reading.get(s['key'], 'N/A')} {s['unit']}"
        for s in sensors
    ])
    faults = get_faults_for_component(industry, component)
    fault_text = "\n".join([
        f"  - {f['fault']}: causes={f['causes']}, repair={f['repair']}, time={f['time_min']}min"
        for f in faults[:5]
    ])

    return f"""You are PredictIQ Assistant — an expert industrial maintenance AI for factory supervisors.
You help operators understand machine health and take action quickly.

Current machine context:
- Industry: {industry}
- Component: {component}
- Status: {status.upper()}
- Live sensor readings:
{sensor_text}

Known fault patterns for this component:
{fault_text}

Your rules:
1. Always be concise — max 3-4 sentences per response
2. Use simple language a factory worker understands
3. If status is critical or bad — give immediate action steps
4. If status is normal/warning — give preventive advice
5. Never use technical jargon without explaining it
6. Always end with a clear action or reassurance
7. Be direct — no fluff, no lengthy explanations"""

def get_auto_message(industry, component, status, reading, sensors):
    """Generate automatic contextual message based on current status."""
    faults = get_faults_for_component(industry, component)
    tips = get_preventive_tips(industry, component)

    if status == "good":
        import random
        tip = random.choice(tips) if tips else "All systems normal."
        return f"✅ All systems normal. Tip: {tip}"

    elif status == "warning":
        bad_sensors = [
            s for s in sensors
            if _get_status(reading.get(s["key"], 0), s) == "warning"
        ]
        if bad_sensors:
            s = bad_sensors[0]
            val = reading.get(s["key"], 0)
            return f"⚠️ {s['label']} at {val} {s['unit']} — approaching limit. Monitor closely and prepare for maintenance."
        return "⚠️ Warning detected. Check sensor readings and prepare maintenance team."

    elif status == "bad":
        if faults:
            f = faults[0]
            return f"❌ {f['fault']} likely. Causes: {f['causes'].split(';')[0]}. Action: {f['repair'].split(';')[0]}. Est. time: {f['time_min']} min."
        return "❌ Component degrading. Reduce load and schedule immediate inspection."

    elif status == "critical":
        if faults:
            steps = []
            for i, f in enumerate(faults[:3]):
                repair = f['repair'].split(';')[0]
                steps.append(f"Step {i+1}: {repair}")
            return f"🚨 CRITICAL! {'. '.join(steps)}. Stop operation immediately!"
        return "🚨 CRITICAL condition! Stop machine immediately and call maintenance team!"

    return "Monitoring active. All readings being processed."

def _get_status(value, sensor):
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

def chat_with_groq(messages, industry, component, status, reading, sensors):
    """Send message to Groq API and get response."""
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not set. Add it to your .env file."

    system_prompt = build_system_prompt(industry, component, status, reading, sensors)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Connection error: {str(e)[:100]}"
