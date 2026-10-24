"""Batch representation adapter; execution is owned by OHAGI training.runner."""

import jax.numpy as jnp
from ohagi.training.engine import EpochPlan
from ohagi.training.runner import train_epoch as train_epoch


def epoch_plan(parameters, optimizer_state, batches, step, dtype):
    """Describe the next KKTHN epoch for execution by the shared OHAGI engine."""

    def prepare(batch):
        inputs, targets, mask, count = batch
        targets = (
            jnp.zeros((len(inputs), 0), dtype=dtype)
            if targets is None
            else jnp.asarray(targets, dtype=dtype)
        )
        return (jnp.asarray(inputs, dtype=dtype), targets, jnp.asarray(mask)), count

    def update(state, batch):
        params, optimizer, metrics = step(*state, batch)
        return (params, optimizer), metrics

    return EpochPlan((parameters, optimizer_state), batches, update, prepare=prepare)


def run_epoch(*args, **kwargs):
    """Execute a KKTHN training epoch through the shared epoch runner."""
    from ohagi.training.engine import Engine

    return Engine.execute(epoch_plan(*args, **kwargs))
