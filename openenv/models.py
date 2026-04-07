from typing import Dict, Literal, Optional 
from pydantic import BaseModel, Field, ConfigDict


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensor_id: str = Field(..., description="The target sensor identifier")
    command: Literal["reboot", "calibrate", "ignore"] = Field(
        ..., description="The maintenance command to issue"
    )


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensors: Dict[str, float] = Field(
        ..., description="Latest reading per sensor (noisy / ambiguous)"
    )
    status_message: str = Field(
        ..., description="Summary of last action and environment response"
    )
    step_count: int = Field(
        ..., description="Number of steps taken in the current episode"
    )


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalized reward including cost and penalties"
    )
    reason: str = Field(
        ..., description="Explanation for reward (fix, penalty, wrong action, etc.)"
    )


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    seed:  Optional [int ]= None