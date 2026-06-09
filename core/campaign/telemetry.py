"""Read W&B run telemetry via wandb.Api.

The only module in core.campaign that touches the network.  wandb.Api is
injected through `api_factory` so unit tests pass a fake and never hit
the wire (WANDB_MODE=disabled does not stub Api reads).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Row = dict[str, Any]

DEFAULT_KEYS: tuple[str, ...] = (
    "eval/success_rate",
    "rollout/ep_rew_mean",
    "train/loss",
    "_step",
    "_runtime",
)


@dataclass(frozen=True)
class CampaignTelemetry:
    rows: list[Row]
    summary: dict[str, Any]


def read_telemetry(
    run_path: str,
    *,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    api_factory: Callable[[], Any] | None = None,
) -> CampaignTelemetry:
    """Fetch history rows + summary for a W&B run.

    run_path: "entity/project/run_id" (or the shortest unambiguous form).
    api_factory: zero-arg callable returning a wandb.Api-like object;
                 defaults to wandb.Api (imported lazily).
    """
    if api_factory is None:
        import wandb

        api_factory = wandb.Api
    api = api_factory()
    run = api.run(run_path)
    rows = [dict(r) for r in run.history(keys=list(keys), pandas=False)]
    summary = dict(run.summary)
    return CampaignTelemetry(rows=rows, summary=summary)
