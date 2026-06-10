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

    # wandb's run.history(keys=...) returns ZERO rows if ANY requested key was
    # never logged by the run (e.g. rollout/ep_rew_mean for an env that only
    # logs rollout/success_rate), or if the requested metrics never co-occur at
    # a single step (eval/* and train/* land on disjoint steps).  Either case
    # silently blinds the monitor — every kill-rule then evaluates against an
    # empty row list and returns a vacuous kill:false.  Fetch each metric key
    # independently and merge by _step so one absent/disjoint metric can't zero
    # out the whole fetch.  (_step/_runtime ride along with every call.)
    metric_keys = [k for k in keys if k not in ("_step", "_runtime")]
    merged: dict[Any, Row] = {}
    for mk in metric_keys:
        for r in run.history(keys=[mk, "_step", "_runtime"], pandas=False):
            row = merged.setdefault(r.get("_step"), {})
            row.update({k: v for k, v in r.items() if v is not None})
    rows = [merged[s] for s in sorted(merged, key=lambda s: (s is None, s))]
    summary = dict(run.summary)
    return CampaignTelemetry(rows=rows, summary=summary)
