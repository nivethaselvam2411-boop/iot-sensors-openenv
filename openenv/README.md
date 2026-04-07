# 🚀 IoT Sensor Maintenance under Uncertainty (OpenEnv)

## 📌 Overview

This project implements an **OpenEnv-compatible reinforcement learning environment** for intelligent IoT sensor maintenance.

Unlike traditional rule-based systems, this environment introduces:

* 🔍 **Hidden fault types**
* 🎲 **Ambiguous sensor readings**
* 💰 **Cost-aware decision making**
* ⏱️ **Time penalties**
* 🔁 **Multi-step fault resolution**

Agents must **infer the underlying issue** and choose optimal actions (`reboot`, `calibrate`, `ignore`) to restore system health efficiently.

---

## 🎯 Problem Statement

In real-world IoT systems:

* Sensor failures are **not directly observable**
* Readings are often **noisy or misleading**
* Maintenance actions have **costs and trade-offs**

This environment simulates such conditions, requiring agents to:

> Make intelligent, sequential decisions under uncertainty.

---

## 🧠 Key Features

### 1. Hidden Fault Modeling

Each sensor has an underlying fault type:

* `stuck` → requires reboot
* `drift` → requires calibration
* `noise` → requires multiple calibrations

⚠️ Fault type is **not visible** to the agent.

---

### 2. Ambiguous Observations

Sensor values are **overlapping and noisy**, meaning:

* Same value can represent different faults
* Agents must **infer**, not directly map

---

### 3. Cost-Aware Actions

Each action has a cost:

| Action    | Cost |
| --------- | ---- |
| calibrate | low  |
| reboot    | high |
| ignore    | zero |

Agents must balance:

> correctness vs cost

---

### 4. Multi-Step Resolution

Some faults (e.g., `noise`) require:

* repeated actions
* persistence

---

### 5. Time Penalty

Each step incurs a penalty:

* Encourages **faster solutions**
* Discourages unnecessary actions

---

## ⚙️ Environment API

### 🔄 Reset

```bash
POST /reset
```

**Body:**

```json
{
  "difficulty": "easy | medium | hard",
  "seed": 42
}
```

---

### ▶️ Step

```bash
POST /step
```

**Body:**

```json
{
  "sensor_id": "sensor_1",
  "command": "reboot | calibrate | ignore"
}
```

---

### 📊 State

```bash
GET /state
```

---

## 🧪 Tasks

### 🟢 Easy

* Single sensor
* Hidden fault
* Minimal ambiguity

### 🟡 Medium

* Multiple sensors
* Drift faults
* Noisy observations

### 🔴 Hard

* Mixed fault types
* Multi-step fixes
* Stochastic behavior

---

## 🏆 Reward Design

Reward is based on:

* ✅ Correct fixes
* ❌ Wrong actions (penalty)
* 💰 Action cost
* ⏱️ Time penalty

Final reward is normalized to **[0, 1]**

---

## 🤖 Agent Strategy (Inference)

The provided agent uses:

* **Adaptive decision making**
* **Action memory per sensor**
* **Cost-aware reasoning**

Strategy:

1. Try `calibrate` first (cheap)
2. Repeat if needed (noise handling)
3. Escalate to `reboot` if unresolved

---

## 🏗️ Project Structure

```bash
.
├── app.py           # FastAPI environment
├── models.py        # Data models
├── inference.py     # Agent logic
├── openenv.yaml     # Environment specification
├── Dockerfile       # Deployment
├── requirements.txt # Dependencies
```

---

## 🚀 Deployment

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

---

### Run with Docker

```bash
docker build -t iot-env .
docker run -p 7860:7860 iot-env
```

---

## 🌍 Real-World Relevance

This environment models:

* Predictive maintenance systems
* Industrial IoT monitoring
* Autonomous system diagnostics

---

## 🧠 Why This Matters

Most environments are:
❌ deterministic
❌ easily solvable
❌ not realistic

This environment is:
✅ uncertain
✅ cost-sensitive
✅ decision-heavy

---

## 🏁 Conclusion

This project demonstrates how to design a **realistic, decision-driven RL environment** that goes beyond simple rule-based simulations.

It challenges agents to:

> Think, adapt, and optimize under real-world constraints.

---

## 👤 Author

**Nivetha Selvam**
**Nithish**
