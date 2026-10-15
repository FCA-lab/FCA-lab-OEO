"""One process per task; bounded concurrency and recoverable checkpoints."""
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from .checkpoints import completed, save

def _execute(worker, task, path):
    try:
        result = worker(task)
        save(path, {"task": task, "status": "ok", "result": result})
    except Exception:
        save(path, {"task": task, "status": "failed", "error": traceback.format_exc()})
        raise SystemExit(1)

def available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None

def run_tasks(tasks, worker, directory, settings, template="{task_id}.json"):
    ctx = mp.get_context("spawn")
    pending = []
    for task in tasks:
        path = Path(directory) / template.format(task_id=task["id"])
        if not completed(path, task):
            pending.append((task, path))
    active = []
    blocked_since = None
    try:
        while pending or active:
            for item in active[:]:
                process, task, path, started = item
                timeout = settings.get("task_timeout_sec")
                expired = timeout is not None and time.monotonic() - started > timeout
                if expired and process.is_alive():
                    process.terminate(); process.join()
                    save(path, {"task": task, "status": "timeout"})
                if not process.is_alive():
                    process.join()
                    if not Path(path).exists():
                        save(path, {"task": task, "status": "failed", "error": f"exit code {process.exitcode}"})
                    active.remove(item)
                    status = json.loads(Path(path).read_text()).get("status", "unknown")
                    print(f"Finished {task['id']} (status={status}, exit={process.exitcode})", flush=True)
            free = available_gb()
            memory_ok = free is None or free >= settings["min_free_gb"]
            if pending and len(active) < settings["n_jobs"] and memory_ok:
                task, path = pending.pop(0)
                process = ctx.Process(target=_execute, args=(worker, task, path))
                process.start()
                active.append((process, task, path, time.monotonic()))
                blocked_since = None
                time.sleep(settings["worker_ramp_sec"])
            else:
                if pending and not active and not memory_ok:
                    blocked_since = blocked_since or time.monotonic()
                    if time.monotonic() - blocked_since > settings["memory_wait_timeout_sec"]:
                        raise RuntimeError("Memory wait timed out; checkpoints preserved")
                time.sleep(0.1)
    finally:
        for process, *_ in active:
            if process.is_alive():
                process.terminate()
            process.join()
