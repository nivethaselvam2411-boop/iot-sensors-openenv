import os
import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://nivz24-iot-sensor-maintenanc.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

client = OpenAI(base_url=API_BASE_URL, api_key="not-needed")

# 🔥 Memory to track attempts (important for reasoning)
sensor_attempts = {}

def get_smart_action(sensors):
    global sensor_attempts

    # 🔥 Get active sensors
    active_sensors = [(sid, val) for sid, val in sensors.items() if abs(val) > 1e-3]

    if not active_sensors:
        return {"sensor_id": "sensor_1", "command": "ignore"}

    # 🔥 Sort sensors by abnormality
    active_sensors.sort(key=lambda x: abs(x[1]), reverse=True)

    for sid, val in active_sensors:

        if sid not in sensor_attempts:
            sensor_attempts[sid] = {"calibrate": 0, "reboot": 0}

        attempts = sensor_attempts[sid]

        # 🔥 Smarter strategy
        if attempts["calibrate"] == 0:
            attempts["calibrate"] += 1
            return {"sensor_id": sid, "command": "calibrate"}

        elif attempts["calibrate"] == 1:
            attempts["calibrate"] += 1
            return {"sensor_id": sid, "command": "calibrate"}

        elif attempts["reboot"] == 0:
            attempts["reboot"] += 1
            return {"sensor_id": sid, "command": "reboot"}

        # 🔥 MOVE TO NEXT SENSOR (IMPORTANT)
        else:
            continue

    # fallback
    return {"sensor_id": active_sensors[0][0], "command": "ignore"}

def run_task(task_id: str):
    global sensor_attempts
    sensor_attempts = {}  # reset memory per task

    print(f"[START] Task: {task_id}")

    response = client.post(
        "/reset",
        cast_to=httpx.Response,
        body={"difficulty": task_id, "seed": 42}
    )

    data = response.json()
    obs = data.get("observation", data)

    step_count = 0
    done = False
    total_reward = 0.0

    while not done and step_count < 10:
        step_count += 1

        action = get_smart_action(obs["sensors"])

        step_resp = client.post(
            "/step",
            cast_to=httpx.Response,
            body=action
        )

        result = step_resp.json()

        obs = result["observation"]
        reward = result["reward"]["value"]
        done = result["done"]

        total_reward += reward

        print(
            f"[STEP] {step_count}: "
            f"{action['sensor_id']} → {action['command']} | "
            f"Reward={reward} | "
            f"StepCount={obs.get('step_count')}"
        )

    print(f"[END] Task: {task_id}, Total Reward: {round(total_reward, 2)}\n")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_task(task)
        except Exception as e:
            print(f"Error running {task}: {e}")