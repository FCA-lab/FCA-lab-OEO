"""Atomic JSON checkpoints; never trust file existence alone."""
import hashlib
import json
import os
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def completed(path, task):
    try:
        value = json.loads(Path(path).read_text())
        return same_task(value.get("task"), task) and value.get("status") == "ok"
    except (OSError, ValueError):
        return False


# 2026-10-06：ID 生成、恢复检查和汇总均使用同一身份规则。
# 调度设置仍保存在原 task 中；仅比较身份时剔除，不修改历史执行记录。
# 源码/数据/划分变化仍阻止复用，修复前检查点不会被冒充为修复后的结果。
def task_identity(task):
    """任务身份忽略调度控制，保留数据、算法、划分、参数及运行环境。"""
    value = json.loads(json.dumps(task))
    value.pop('id', None)
    provenance = value.get('provenance', {})
    settings = provenance.get('settings', {})
    for key in ('n_jobs','min_free_gb','worker_ramp_sec','memory_wait_timeout_sec',
                'task_timeout_sec','resume','keep_partial','tasks_per_worker',
                'scheduler','rc_oeo_share','memory_resume_margin_gb'):
        settings.pop(key, None)
    # 2026-10-18: scheduling/storage changes do not change BABD calculations.
    from .resume_identity import normalize_provenance
    normalize_provenance(provenance)
    return value


def task_id(task):
    return digest(task_identity(task))[:24]


def same_task(saved, current):
    return isinstance(saved, dict) and task_identity(saved) == task_identity(current)


def recover_compatible_checkpoints(tasks, directory, template="{task_id}.json"):
    """Find successful legacy IDs once, preserving their original provenance.

    Scan one file at a time: do not retain historical sample indices in memory.
    Failed/unknown computation versions cannot satisfy the identity comparison.
    """
    import re
    directory = Path(directory)
    wanted = {task_id(t): t for t in tasks
              if not completed(directory/template.format(task_id=t['id']), t)}
    if not wanted or not directory.exists():
        return 0
    recovered = 0
    # Legacy checkpoints have 24-hex task IDs; never read task index JSON here.
    pattern = re.compile('^' + re.escape(template).replace(
        re.escape('{task_id}'), '[0-9a-f]{24}') + '$')
    paths = sorted(p for p in directory.iterdir() if pattern.match(p.name))
    print(f"[检查点检查] 待匹配 {len(wanted)} 个任务，扫描 {len(paths)} 个历史文件", flush=True)
    for i, path in enumerate(paths, 1):
        try:
            record = json.loads(path.read_text())
            saved = record.get('task')
            if record.get('status') != 'ok' or not isinstance(saved, dict):
                continue
            key = task_id(saved)
            current = wanted.get(key)
            if current is None or not same_task(saved, current):
                continue
            target = directory/template.format(task_id=current['id'])
            if target.exists():
                # Preserve a failed/incompatible attempt before installing a match.
                import time
                target.rename(target.with_name(target.stem+'.previous-'+str(time.time_ns())+'.json'))
            record['reused_from'] = str(path.name)
            save(target, record)
            del wanted[key]
            recovered += 1
        except (OSError, ValueError, TypeError, KeyError):
            # Corrupt checkpoint: leave it untouched and recompute the task.
            continue
        if i % 100 == 0:
            print(f"[检查点检查] 已扫描 {i}/{len(paths)}，找到 {recovered} 个兼容结果", flush=True)
        if not wanted:
            break
    print(f"[检查点恢复] 复用旧 ID 的成功结果 {recovered} 个", flush=True)
    return recovered
