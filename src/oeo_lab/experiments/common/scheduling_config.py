"""Shared worker admission settings for all experiments using common.runner."""

MEMORY_RESERVE_GIB = 10.0
MEMORY_RESUME_MARGIN_GIB = 5.0


def effective_settings(experiment_settings):
    """Keep each experiment's task/worker choices and apply shared memory policy."""
    return dict(experiment_settings, min_free_gb=MEMORY_RESERVE_GIB,
                memory_resume_margin_gb=MEMORY_RESUME_MARGIN_GIB)
