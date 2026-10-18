# BABD task scheduling

BABD enables `scheduler='babd_rc_oeo'` and the configured `rc_oeo_share` in
`src/oeo_lab/experiments/classification/babd_config.py` (`EXECUTION`).
`n_jobs` remains the total number of concurrent task processes, not CPU affinity.

After completed checkpoints are filtered, RC-OEO tasks are grouped by seed
in their original plan order. Within a seed, distinct (training budget, bins)
combinations are ranked by budget*bins and dispatched alternately from the
large and small ends. Each combination contributes one fold per round before
its next fold is selected. Thus the first round mixes sizes instead of launching
all folds/seeds of the largest setting together. Missing completed combinations
are skipped. Seed groups control queue order, not a barrier waiting for all
running tasks from one seed to finish. Other models retain their original order.
This is a resource-cost heuristic, not a memory estimate or an OOM guarantee.

While both pending queues are nonempty, RC-OEO receives floor(rc_oeo_share*n_jobs)
slots, clamped to [1,n_jobs-1], and other models receive the rest. For n_jobs=1,
RC-OEO is dispatched first. As soon as either pending queue is empty, all free
slots can be used by the other queue, even while processes from the empty queue
are still running. Running tasks are never preempted. Total concurrency and the
existing available-memory threshold remain enforced. The memory threshold does
not reserve future memory and cannot guarantee prevention of OOM.

The balanced and imbalanced experiments still run separately in the existing
order. No estimator, input data, split, score or experimental grid is changed.
Other experiments without this scheduler setting retain FIFO scheduling.

Already running parent processes retain their loaded scheduling code. The new
policy applies on the next launch. This change does not terminate any experiment
or rewrite historical checkpoints. Source hashes participate in checkpoint
identity: old checkpoints may not be reused after this source change. No broad
source-hash bypass is introduced. Scheduling changes can affect wall-clock timing
through contention; preserve run provenance when comparing timings.

Validation: `PYTHONPATH=src python tests/classification/test_babd_scheduler.py`.
