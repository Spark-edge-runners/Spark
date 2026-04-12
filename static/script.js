const CONFIG = {
  colors: [
    { stop: 0.0, color: [161, 140, 209] }, // purple
    { stop: 0.5, color: [255, 106, 159] }, // pink
    { stop: 1.0, color: [0, 212, 255] },    // cyan
  ],
  endpoints: {
    aerospace: {
      model: "/predict/aerospace/model",
      encoder: "/predict/aerospace/encoder",
      scaler: "/predict/aerospace/scaler",
      features: ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
    },
    cnc: {
      model: "/predict/cnc/model",
      encoder: "/predict/cnc/encoder",
      scaler: "/predict/cnc/scaler",
      features: ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
    },
    energy: {
      model: "/predict/energy/model",
      encoder: "/predict/energy/encoder",
      scaler: "/predict/energy/scaler",
      features: ["wind_speed", "theoretical_power", "wind_direction", "power"],
    },
  },
};

const state = {
  isAnimating: false,
  activePredictor: null,
  energyScene: null,
};

const body = document.body;
const cursorGlow = document.getElementById("cursor-glow");
const progressBar = document.getElementById("progress-bar");

function clamp(n, min = 0, max = 1) {
  return Math.max(min, Math.min(max, n));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ];
}

function interpolateFlow(normalized) {
  const t = clamp(normalized);
  if (t <= 0.5) return lerpColor(CONFIG.colors[0].color, CONFIG.colors[1].color, t / 0.5);
  return lerpColor(CONFIG.colors[1].color, CONFIG.colors[2].color, (t - 0.5) / 0.5);
}

function updateScrollFlow() {
  const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const normalized = clamp(window.scrollY / maxScroll);
  const color = interpolateFlow(normalized);
  document.documentElement.style.setProperty("--flow", `${color[0]}, ${color[1]}, ${color[2]}`);
  progressBar.style.width = `${Math.round(normalized * 100)}%`;
}

window.addEventListener("scroll", updateScrollFlow, { passive: true });
window.addEventListener("resize", updateScrollFlow);
updateScrollFlow();

document.addEventListener("mousemove", (e) => {
  cursorGlow.style.transform = `translate(${e.clientX}px, ${e.clientY}px) translate(-50%, -50%)`;
  cursorGlow.style.opacity = "0.62";
});

document.addEventListener("mouseleave", () => {
  cursorGlow.style.opacity = "0.25";
});

/* =========================
   CNC TAB SWITCHING
========================= */
document.querySelectorAll("[data-tab]").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const tab = btn.dataset.tab;

    document.querySelectorAll("[data-tab]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    document.querySelectorAll(".cnc-tab").forEach((frame) => {
      frame.classList.toggle("active", frame.dataset.cncView === tab);
    });
  });
});

/* =========================
   ENERGY TURBINE
========================= */
function createEnergyScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b14);

  const camera = new THREE.PerspectiveCamera(40, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
  camera.position.set(0, 2.4, 8.5);

  const ambient = new THREE.AmbientLight(0xffffff, 1.7);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xd7f7ff, 2.2);
  key.position.set(4, 6, 6);
  scene.add(key);

  const rim = new THREE.PointLight(0x00d4ff, 1.4, 20);
  rim.position.set(0, 2, 5);
  scene.add(rim);

  const group = new THREE.Group();
  scene.add(group);

  const matBase = new THREE.MeshStandardMaterial({ color: 0x5f6d85, roughness: 0.45, metalness: 0.75 });
  const matBlade = new THREE.MeshStandardMaterial({ color: 0xdff8ff, roughness: 0.18, metalness: 0.35 });
  const matHub = new THREE.MeshStandardMaterial({ color: 0x00d4ff, roughness: 0.16, metalness: 0.7, emissive: 0x012733, emissiveIntensity: 0.4 });

  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.9, 5.2, 18), matBase);
  tower.position.y = -0.8;
  group.add(tower);

  const nacelle = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.9, 1.1), matBase);
  nacelle.position.set(0, 1.8, 0);
  group.add(nacelle);

  const hub = new THREE.Mesh(new THREE.SphereGeometry(0.35, 32, 32), matHub);
  hub.position.set(1.15, 1.8, 0);
  group.add(hub);

  const bladeGeo = new THREE.BoxGeometry(2.8, 0.12, 0.26);
  const blades = [];
  for (let i = 0; i < 3; i++) {
    const blade = new THREE.Mesh(bladeGeo, matBlade);
    blade.position.set(2.65, 1.8, 0);
    blade.rotation.z = (Math.PI * 2 / 3) * i;
    blade.rotation.y = (Math.PI * 2 / 3) * i;
    group.add(blade);
    blades.push(blade);
  }

  let targetX = 0;
  let targetY = 0;
  let currentX = 0;
  let currentY = 0;

  canvas.parentElement.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    targetY = ((e.clientX - rect.left) / rect.width - 0.5) * 1.2;
    targetX = ((e.clientY - rect.top) / rect.height - 0.5) * 0.9;
  });

  canvas.parentElement.addEventListener("mouseleave", () => {
    targetX = 0;
    targetY = 0;
  });

  function resize() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function animate() {
    currentX += (targetX - currentX) * 0.05;
    currentY += (targetY - currentY) * 0.05;

    group.rotation.x = -0.15 + currentX;
    group.rotation.y = currentY;
    blades.forEach((b, i) => {
      b.rotation.x += 0.02 + i * 0.001;
    });

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  resize();
  window.addEventListener("resize", resize);

  animate();

  return { scene, camera, renderer, group };
}

const energyCanvas = document.getElementById("energyCanvas");
state.energyScene = createEnergyScene(energyCanvas);

/* =========================
   EXPLOSION / OPEN PANEL
========================= */
function rippleAt(target, x, y) {
  const ripple = document.createElement("span");
  ripple.className = "ripple";
  const rect = target.getBoundingClientRect();
  ripple.style.left = `${x - rect.left}px`;
  ripple.style.top = `${y - rect.top}px`;
  target.appendChild(ripple);
  setTimeout(() => ripple.remove(), 820);
}

function openPanel(domain) {
  document.querySelectorAll(".component-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== domain);
  });
  const panel = document.querySelector(`.component-panel[data-panel="${domain}"]`);
  if (panel) panel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function explodeSection(domain, section, x, y) {
  if (state.isAnimating) return;
  state.isAnimating = true;

  rippleAt(section, x, y);

  const shell = section.querySelector(".stage-shell");
  if (shell) shell.classList.add("exploding");
  body.classList.add("exploding");

  setTimeout(() => {
    openPanel(domain);
  }, 980);

  setTimeout(() => {
    if (shell) shell.classList.remove("exploding");
    body.classList.remove("exploding");
    state.isAnimating = false;
  }, 1350);
}

document.querySelectorAll(".section").forEach((section) => {
  section.addEventListener("click", (event) => {
    if (event.target.closest("button, a, input, select, textarea, iframe, .component-card, .tab-btn, .close-panel")) return;
    explodeSection(section.dataset.domain, section, event.clientX, event.clientY);
  });
});

document.querySelectorAll("[data-open-components]").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const domain = btn.dataset.openComponents;
    const section = document.querySelector(`.section[data-domain="${domain}"]`);
    if (section) {
      explodeSection(domain, section, window.innerWidth * 0.5, section.getBoundingClientRect().top + 120);
    }
  });
});

document.querySelectorAll("[data-close]").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const domain = btn.dataset.close;
    const panel = document.querySelector(`.component-panel[data-panel="${domain}"]`);
    if (panel) panel.classList.add("hidden");
  });
});

/* =========================
   PREDICTOR PANELS
========================= */
function getDefaultFields(domain) {
  if (domain === "aerospace") {
    return [
      { id: "s2", label: "s2", value: 642.3, step: 0.01 },
      { id: "s3", label: "s3", value: 1588.9, step: 0.01 },
      { id: "s4", label: "s4", value: 1401.2, step: 0.01 },
      { id: "s7", label: "s7", value: 21.6, step: 0.01 },
      { id: "s8", label: "s8", value: 552.6, step: 0.01 },
      { id: "s9", label: "s9", value: 2388.0, step: 0.01 },
      { id: "s11", label: "s11", value: 9048.5, step: 0.01 },
      { id: "s12", label: "s12", value: 1.31, step: 0.01 },
      { id: "s13", label: "s13", value: 47.3, step: 0.01 },
      { id: "s14", label: "s14", value: 521.8, step: 0.01 },
      { id: "s15", label: "s15", value: 2388.1, step: 0.01 },
      { id: "s17", label: "s17", value: 8135.0, step: 0.01 },
      { id: "s20", label: "s20", value: 8.42, step: 0.01 },
      { id: "s21", label: "s21", value: 23.38, step: 0.01 },
    ];
  }

  if (domain === "cnc") {
    return [
      { id: "air_temp", label: "Air temperature [K]", value: 298.1, step: 0.1 },
      { id: "process_temp", label: "Process temperature [K]", value: 308.4, step: 0.1 },
      { id: "rpm", label: "Rotational speed [rpm]", value: 1550, step: 1 },
      { id: "torque", label: "Torque [Nm]", value: 40.2, step: 0.1 },
      { id: "tool_wear", label: "Tool wear [min]", value: 95, step: 1 },
    ];
  }

  return [
    { id: "wind_speed", label: "wind_speed", value: 8.8, step: 0.1 },
    { id: "theoretical_power", label: "theoretical_power", value: 920, step: 1 },
    { id: "wind_direction", label: "wind_direction", value: 134, step: 1 },
    { id: "power", label: "power", value: 850, step: 1 },
  ];
}

function buildPredictorUI(domain, endpoint) {
  const shell = document.querySelector(`.predictor-shell[data-predictor="${domain}"]`);
  if (!shell) return;

  const fields = getDefaultFields(domain).map((field) => `
    <div class="field">
      <label for="${field.id}">${field.label}</label>
      <input id="${field.id}" name="${field.id}" type="number" step="${field.step || "any"}" value="${field.value}" />
    </div>
  `).join("");

  shell.innerHTML = `
    <div class="predictor-head">
      <h4>${domain.toUpperCase()} prediction screen</h4>
      <p class="note">Posting to <code>${endpoint}</code> with the feature order used by your backend.</p>
    </div>

    <div class="form-grid">
      ${fields}
    </div>

    <div class="form-actions">
      <button class="primary-btn" data-run="${domain}">Run prediction</button>
      <button class="secondary-btn" data-fill="${domain}">Load sample data</button>
    </div>

    <div class="prediction-output" id="output-${domain}">Waiting for input…</div>
  `;

  shell.classList.remove("hidden");
  state.activePredictor = domain;

  shell.querySelector(`[data-fill="${domain}"]`).addEventListener("click", () => {
    getDefaultFields(domain).forEach((field) => {
      const el = document.getElementById(field.id);
      if (el) el.value = field.value;
    });
  });

  shell.querySelector(`[data-run="${domain}"]`).addEventListener("click", () => runPrediction(domain, endpoint));
}

document.querySelectorAll(".component-card").forEach((card) => {
  card.addEventListener("click", (event) => {
    event.stopPropagation();
    const domain = card.dataset.domain;
    const endpoint = card.dataset.endpoint;
    const panel = document.querySelector(`.component-panel[data-panel="${domain}"]`);
    if (!panel) return;

    const predictor = panel.querySelector(`.predictor-shell[data-predictor="${domain}"]`);
    predictor.classList.remove("hidden");
    buildPredictorUI(domain, endpoint);
    predictor.scrollIntoView({ behavior: "smooth", block: "center" });
  });
});

function buildPayload(domain) {
  if (domain === "aerospace") {
    const payload = {};
    CONFIG.endpoints.aerospace.features.forEach((key) => {
      payload[key] = Number(document.getElementById(key)?.value ?? 0);
    });
    return payload;
  }

  if (domain === "cnc") {
    return {
      "Air temperature [K]": Number(document.getElementById("air_temp")?.value ?? 0),
      "Process temperature [K]": Number(document.getElementById("process_temp")?.value ?? 0),
      "Rotational speed [rpm]": Number(document.getElementById("rpm")?.value ?? 0),
      "Torque [Nm]": Number(document.getElementById("torque")?.value ?? 0),
      "Tool wear [min]": Number(document.getElementById("tool_wear")?.value ?? 0),
    };
  }

  return {
    wind_speed: Number(document.getElementById("wind_speed")?.value ?? 0),
    theoretical_power: Number(document.getElementById("theoretical_power")?.value ?? 0),
    wind_direction: Number(document.getElementById("wind_direction")?.value ?? 0),
    power: Number(document.getElementById("power")?.value ?? 0),
  };
}

function formatPredictionResponse(data) {
  if (typeof data === "string") return data;
  if (data?.prediction !== undefined) return `Prediction: ${data.prediction}\n\n${data.message || "Inference completed successfully."}`;
  if (data?.label !== undefined) return `Label: ${data.label}\n\n${data.message || "Inference completed successfully."}`;
  return JSON.stringify(data, null, 2);
}

async function runPrediction(domain, endpoint) {
  const output = document.getElementById(`output-${domain}`);
  const payload = buildPayload(domain);
  output.textContent = "Running prediction…";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { raw: text };
    }

    output.textContent = formatPredictionResponse(parsed);
  } catch (err) {
    output.textContent =
      "Backend unavailable.\n\n" +
      "Payload sent:\n" +
      JSON.stringify(payload, null, 2) +
      "\n\nConnect your Flask/FastAPI backend to return JSON.";
  }
}

/* Auto-fill samples so the panels never feel empty */
setTimeout(() => {
  ["aerospace", "cnc", "energy"].forEach((domain) => {
    getDefaultFields(domain).forEach((field) => {
      const el = document.getElementById(field.id);
      if (el) el.value = field.value;
    });
  });
}, 120);