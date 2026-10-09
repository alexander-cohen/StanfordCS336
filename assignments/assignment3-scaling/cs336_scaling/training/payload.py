from typing import Literal
from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.schemas.experiment import FailReason


class CompletedPayload(FrozenForbidExtraModel):
    used_runtime_seconds: float
    val_losses: list[float]
    result_type: Literal["completed"]


class FailedPayload(FrozenForbidExtraModel):
    used_runtime_seconds: float
    reason: FailReason
    result_type: Literal["failed"]


type FinishPayload = CompletedPayload | FailedPayload
