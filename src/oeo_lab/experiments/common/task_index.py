"""分类任务索引：参数清单与可复用的数据划分分开保存。

只改变磁盘表示，不改变传给模型的任务、任务 ID 或已有检查点。
"""
import hashlib
import json
import os
from pathlib import Path


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, separators=(',', ':'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_task_index(path, tasks):
    """重复索引数组单独落盘一次；数据配置、环境和模型配置共享引用。

    接受任务迭代器，迁移旧索引时不必把整个文件加载进内存。
    各内容文件先落盘，最后原子替换主索引，避免产生半成品引用。
    """
    path = Path(path)
    folder = path.with_name(path.stem + '_data')
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    written = set()

    def put(value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        key = hashlib.sha256(raw).hexdigest()
        if key not in written:
            target = folder / (key + '.json')
            # 内容寻址，不依赖任务顺序；重新运行时也核对原文件。
            if not target.exists() or target.read_bytes() != raw:
                tmp = target.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, target)
            written.add(key)
        return key

    for task in tasks:
        entry = dict(task)
        refs = {}
        for field in ('dataset', 'model', 'provenance'):
            if field in entry:
                refs[field] = put(entry.pop(field))
        if 'split' in entry:
            split = dict(entry.pop('split'))
            arrays = {}
            for field in ('train_indices', 'test_indices'):
                if field in split:
                    arrays[field] = put(split.pop(field))
            split['_index_arrays'] = arrays
            refs['split'] = put(split)
        entry['_index_refs'] = refs
        entries.append(entry)
    _atomic_json(path, {'format_version': 2, 'data_dir': folder.name, 'tasks': entries})


def iter_task_index(path):
    """逐个还原完整任务；仍支持旧的带缩进内嵌任务格式。

    不修改身份信息；返回的任务可继续用于已有 completed/same_task 检查。
    """
    path = Path(path)
    # 旧索引由 save(..., indent=2) 生成：每个任务结束行缩进为四格。
    # 对旧文件流式读取，避免一次解析数 GB JSON。
    with path.open(encoding='utf-8') as stream:
        first = stream.readline()
        if first.strip() == '{':
            if stream.readline().strip() != '"tasks": [':
                raise ValueError('Unsupported legacy task index')
            block = []
            for line in stream:
                if not block and line.strip() in (']', ']}', '],', '}'):
                    continue
                block.append(line)
                if line.rstrip() in ('    },', '    }'):
                    yield json.loads(''.join(block).rstrip().rstrip(','))
                    block = []
            if block:
                raise ValueError('Truncated legacy task index')
            return
        stream.seek(0)
        index = json.load(stream)
    if index.get('format_version') != 2:
        raise ValueError('Unsupported task index version')
    folder = path.parent / index['data_dir']

    def get(key):
        raw = (folder / (key + '.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != key:
            raise ValueError('Task index content checksum mismatch')
        return json.loads(raw)

    for stored in index['tasks']:
        task = dict(stored)
        for field, key in task.pop('_index_refs').items():
            task[field] = get(key)
        if 'split' in task:
            split = task['split']
            for field, key in split.pop('_index_arrays').items():
                split[field] = get(key)
        yield task
