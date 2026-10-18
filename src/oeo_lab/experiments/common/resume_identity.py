"""Separate BABD computation identity from scheduling and checkpoint transport.

Raw provenance is retained in every checkpoint. Historical source versions are
accepted only when their exact hash was audited against the current computation.
"""
import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PREFIX = 'src/oeo_lab/experiments/'
TRANSPORT = {PREFIX + 'common/' + name for name in (
    'checkpoints.py', 'provenance.py', 'task_index.py', 'resume_identity.py',
    'resource_guard.py', 'scheduling_config.py')}


def computation_hash(path, source):
    """Fingerprint calculation code while excluding shared scheduling knobs."""
    special = {PREFIX + 'classification/runner.py', PREFIX + 'common/runner.py'}
    special.update(PREFIX + 'classification/' + name for name in (
        'babd_config.py','uci_config.py','bitcoin_config.py','elliptic_config.py',
        'tabpfn_config.py','completeness_config.py','config.py'))
    if path not in special:
        return hashlib.sha256(source.encode()).hexdigest()
    tree = ast.parse(source)
    if path == PREFIX + 'common/runner.py':
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_execute']
    elif path == PREFIX + 'classification/runner.py':
        tree.body = [n for n in tree.body if not (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == 'run_classification_tasks')]
    else:
        if path.endswith('/babd_config.py'):
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'EXECUTION' for t in n.targets))]
        else:
            for node in tree.body:
                if not (isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'EXECUTION' for t in node.targets)):
                    continue
                value=node.value
                if isinstance(value, ast.Dict):
                    pairs=[(k,v) for k,v in zip(value.keys,value.values)
                           if not (isinstance(k, ast.Constant)
                           and k.value in {'min_free_gb','memory_resume_margin_gb'})]
                    value.keys=[k for k,_ in pairs];value.values=[v for _,v in pairs]
    return 'ast-v2:' + hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def legacy_aliases():
    return json.loads(Path(__file__).with_name('checkpoint_source_aliases.json').read_text())


def normalize_provenance(provenance):
    """Conservative migration: unknown old source hashes remain incompatible."""
    sources = provenance.get('sources')
    if not isinstance(sources, dict):
        return
    computed = provenance.get('computation_sources', {})
    aliases = legacy_aliases()
    identity = {}
    for path, raw_hash in sources.items():
        if path in TRANSPORT:
            continue
        identity[path] = aliases.get(path, {}).get(raw_hash, computed.get(path, raw_hash))
    provenance.pop('source_hash', None)
    provenance.pop('sources', None)
    provenance['computation_sources'] = identity
