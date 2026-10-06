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
                'task_timeout_sec','resume','keep_partial','tasks_per_worker'):
        settings.pop(key, None)
    return value


def task_id(task):
    return digest(task_identity(task))[:24]


def same_task(saved, current):
    return isinstance(saved, dict) and task_identity(saved) == task_identity(current)
