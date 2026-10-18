"""One process per task; bounded concurrency and recoverable checkpoints."""
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from .checkpoints import completed, save
from .resource_guard import memory_status, memory_admission
from .scheduling_config import effective_settings

def _execute(worker, task, path):
    try:
        result = worker(task)
        save(path, {"task": task, "status": "ok", "result": result})
    except Exception:
        save(path, {"task": task, "status": "failed", "error": traceback.format_exc()})
        raise SystemExit(1)

def _is_rc_oeo(task):
    return task.get("model_name") == "rc-oeo"


def _prepare_pending(pending, settings):
    """BABD：按种子分组，组内交错大小组合，再轮转各组合的折号。

    训练规模乘分箱数仅作为排序估计，不是内存预测值。
    相同组合的各折不能连续占据队首；先让每种组合各分配一折。
    """
    from collections import deque
    if settings.get("scheduler") != "babd_rc_oeo":
        return pending
    share = settings.get("rc_oeo_share", 0.75)
    if not 0 < share < 1:
        raise ValueError("rc_oeo_share must be between 0 and 1")
    groups = {}
    others = []
    for item in pending:
        task = item[0]
        if not _is_rc_oeo(task):
            others.append(item)
            continue
        # 字典保留种子在原任务计划中的顺序，不强制按数值重排。
        seed = task.get("split", {}).get("seed", 0)
        combination = (task["train_budget"], task["max_bins"])
        groups.setdefault(seed, {}).setdefault(combination, []).append(item)
    ordered = []
    for combinations in groups.values():
        keys = deque(sorted(combinations, key=lambda k: (k[0]*k[1], k[0], k[1])))
        mixed = []
        while keys:
            mixed.append(keys.pop())  # 大组合先启动，随后穿插小组合。
            if keys:
                mixed.append(keys.popleft())
        queues = [deque(sorted(combinations[k],
                    key=lambda item: item[0].get("split", {}).get("fold", 1)))
                  for k in mixed]
        while queues:
            for queue in queues:
                ordered.append(queue.popleft())
            queues = [queue for queue in queues if queue]
    return ordered + others


def _next_pending_index(pending, active, settings):
    """两队列非空时保留配额；任一队列清空即允许另一组使用所有空闲名额。

    active 保存已经启动、尚未回收的进程。这里只选任务，不终止或抢占进程。
    单进程时优先 RC-OEO；普通实验仍使用 FIFO。
    """
    if settings.get("scheduler") != "babd_rc_oeo" or settings["n_jobs"] == 1:
        return 0
    first_rc = next((i for i, item in enumerate(pending) if _is_rc_oeo(item[0])), None)
    first_other = next((i for i, item in enumerate(pending) if not _is_rc_oeo(item[0])), None)
    if first_rc is None:
        return first_other
    if first_other is None:
        return first_rc
    jobs = settings["n_jobs"]
    rc_limit = max(1, min(jobs - 1, int(jobs * settings.get("rc_oeo_share", 0.75))))
    rc_active = sum(_is_rc_oeo(item[1]) for item in active)
    if rc_active < rc_limit:
        return first_rc
    if len(active) - rc_active < jobs - rc_limit:
        return first_other
    return None


def run_tasks(tasks, worker, directory, settings, template="{task_id}.json"):
    settings = effective_settings(settings)
    ctx = mp.get_context("spawn")
    pending = []
    for task in tasks:
        path = Path(directory) / template.format(task_id=task["id"])
        if not completed(path, task):
            pending.append((task, path))
    pending = _prepare_pending(pending, settings)
    if settings.get("scheduler") == "babd_rc_oeo":
        jobs = settings["n_jobs"]
        rc_limit = 1 if jobs == 1 else max(1, min(jobs - 1, int(jobs * settings.get("rc_oeo_share", 0.75))))
        print(f"[调度] RC-OEO 按种子分组、大小组合交错、折号轮转；并行名额 {rc_limit}+{jobs-rc_limit}；"
              "任一待分配队列清空后共享空闲名额；保留内存限制", flush=True)
    active = []
    blocked_since = None
    memory_blocked = False
    last_memory_notice = 0.0
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
            snapshot = memory_status()
            was_blocked = memory_blocked
            reserve = settings["min_free_gb"]
            margin = settings["memory_resume_margin_gb"]
            memory_ok = memory_admission(snapshot, reserve, was_blocked, margin)
            memory_blocked = not memory_ok
            now = time.monotonic()
            if pending and memory_blocked and (not was_blocked or now-last_memory_notice >= 60):
                detail = ("无法读取内存，暂停分配" if snapshot is None else
                    f"RAM可用={snapshot['available_gib']:.1f} GiB，"
                    f"swap已用={snapshot['swap_used_gib']:.1f} GiB，"
                    f"有效余量={snapshot['headroom_gib']:.1f} GiB")
                print(f"[内存暂停] {detail}；预留={reserve:.1f} GiB，"
                      f"恢复要求={reserve+margin:.1f} GiB；运行={len(active)}/{settings['n_jobs']}；"
                      "等待任务完成，不补充新任务", flush=True)
                last_memory_notice = now
            elif pending and was_blocked and memory_ok:
                print(f"[内存恢复] 有效余量={snapshot['headroom_gib']:.1f} GiB；恢复分配", flush=True)
            if pending and len(active) < settings["n_jobs"] and memory_ok:
                index = _next_pending_index(pending, active, settings)
                if index is None:
                    time.sleep(0.1)
                    continue
                task, path = pending.pop(index)
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
