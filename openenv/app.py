from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from threading import Lock
import random
import time

from models import Action, Observation, Reward


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    seed: Optional[int] = None


class EnvState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty: Literal["easy", "medium", "hard"]
    total_issues: int
    fixed_issues: int
    remaining_issues: int
    sensors: Dict[str, float]
    message: str


class SensorEnv:
    def __init__(self) -> None:
        self._rng = random.Random()
        self._difficulty: Literal["easy", "medium", "hard"] = "easy"
        self._sensor_values: Dict[str, float] = {}
        self._issue_map: Dict[str, Optional[dict]] = {}
        self._total_issues: int = 0
        self._fixed_issues: int = 0
        self._step_count: int = 0
        self._lock = Lock()

        # NEW: cost + penalty
        self.ACTION_COST = {
            "reboot": 0.3,
            "calibrate": 0.1,
            "ignore": 0.0
        }
        self.STEP_PENALTY = 0.05

    def reset(self, difficulty: Literal["easy", "medium", "hard"] = "easy", seed: Optional[int] = None) -> Observation:
        with self._lock:
            if seed is not None:
                self._rng.seed(seed)

            self._difficulty = difficulty
            self._sensor_values.clear()
            self._issue_map.clear()
            self._fixed_issues = 0
            self._step_count = 0

            if difficulty == "easy":
                self._sensor_values = {"sensor_1": self._rng.uniform(80, 120)}
                self._issue_map = {
                    "sensor_1": {"type": "stuck", "severity": 1.0}
                }

            elif difficulty == "medium":
                self._sensor_values = {
                    "sensor_1": self._rng.uniform(80, 120),
                    "sensor_2": self._rng.uniform(80, 120),
                    "sensor_3": self._rng.uniform(80, 120),
                }
                self._issue_map = {
                    k: {"type": "drift", "severity": 1.0}
                    for k in self._sensor_values.keys()
                }

            elif difficulty == "hard":
                self._sensor_values = {
                    "sensor_1": self._rng.uniform(80, 120),
                    "sensor_2": self._rng.uniform(80, 120),
                    "sensor_3": self._rng.uniform(80, 120),
                    "sensor_4": self._rng.uniform(80, 120),
                    "sensor_5": 0.0,
                }
                self._issue_map = {
                    "sensor_1": {"type": "stuck", "severity": 1.0},
                    "sensor_2": {"type": "stuck", "severity": 1.0},
                    "sensor_3": {"type": "drift", "severity": 1.0},
                    "sensor_4": {"type": "noise", "severity": 1.0},
                    "sensor_5": None,
                }

            else:
                raise ValueError("Unsupported difficulty")

            self._total_issues = sum(1 for v in self._issue_map.values() if v is not None)

            return Observation(
                sensors=dict(self._sensor_values),
                status_message=f"Reset {difficulty}. Issues: {self._total_issues}.",
                step_count=self._step_count
            )

    def step(self, action: Action) -> Tuple[Observation, Reward]:
        with self._lock:
            self._step_count += 1

            sid = action.sensor_id
            cmd = action.command
            now = time.time()

            if sid not in self._sensor_values:
                return Observation(
                    sensors=dict(self._sensor_values),
                    status_message="Invalid sensor",
                    step_count=self._step_count
                ), Reward(value=0.0, reason="Invalid sensor")

            issue = self._issue_map.get(sid)
            reward_value = 0.0
            reason = "No progress"

            if issue is None:
                reward_value -= 0.1
                reason = "Unnecessary action"

            else:
                issue_type = issue["type"]

                if issue_type == "stuck" and cmd == "reboot":
                    self._issue_map[sid] = None
                    self._sensor_values[sid] = 0.0
                    self._fixed_issues += 1
                    reward_value += 1.0 / self._total_issues
                    reason = "Fixed stuck sensor"

                elif issue_type == "drift" and cmd == "calibrate":
                    self._issue_map[sid] = None
                    self._sensor_values[sid] = 0.0
                    self._fixed_issues += 1
                    reward_value += 1.0 / self._total_issues
                    reason = "Fixed drift sensor"

                elif issue_type == "noise" and cmd == "calibrate":
                    issue["severity"] -= 0.5
                    if issue["severity"] <= 0:
                        self._issue_map[sid] = None
                        self._sensor_values[sid] = 0.0
                        self._fixed_issues += 1
                        reward_value += 1.0 / self._total_issues
                        reason = "Noise resolved"
                    else:
                        reason = "Partial noise reduction"

                else:
                    reward_value -= 0.2
                    reason = "Wrong action"

            # Apply cost + time penalty
            reward_value -= self.ACTION_COST[cmd]
            reward_value -= self.STEP_PENALTY

            reward_value = max(0.0, min(1.0, reward_value))

            remaining = sum(1 for v in self._issue_map.values() if v is not None)

            return Observation(
                sensors=dict(self._sensor_values),
                status_message=f"[{now:.0f}] {reason}. Remaining: {remaining}",
                step_count=self._step_count
            ), Reward(value=round(reward_value, 4), reason=reason)

    def state(self) -> EnvState:
        with self._lock:
            remaining = sum(1 for v in self._issue_map.values() if v is not None)
            return EnvState(
                difficulty=self._difficulty,
                total_issues=self._total_issues,
                fixed_issues=self._fixed_issues,
                remaining_issues=remaining,
                sensors=dict(self._sensor_values),
                message="Current state",
            )


app = FastAPI(title="OpenEnv - IoT Sensor Maintenance", version="2.0")
ENV = SensorEnv()


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    return ENV.reset(difficulty=req.difficulty, seed=req.seed)


@app.post("/step")
def step(action: Action) -> Dict[str, object]:
    obs, rew = ENV.step(action)
    done = ENV.state().remaining_issues == 0
    return {
        "observation": obs.model_dump(),
        "reward": rew.model_dump(),
        "done": done,
    }


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    return ENV.state()