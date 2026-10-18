"""通用运行环境检查与源码指纹采集，不包含具体实验配置。"""
import hashlib
import importlib.metadata
import platform
from .checkpoints import digest


def collect_provenance(root, source_patterns, packages, settings):
    """检查配置指定的包和源码是否存在，记录环境与内容指纹。

    检查范围是运行依赖是否可定位，并不等价于证明实验可复现。
    实验参数由调用方显式传入，避免读取其他实验的配置文件。
    """
    from .scheduling_config import effective_settings
    settings = effective_settings(settings)
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'Missing required package: {name}') from exc
    sources = set()
    for pattern in source_patterns:
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise FileNotFoundError(f'No source files match: {pattern}')
        sources.update(matches)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources)}
    result = {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'settings': settings, 'sources': hashes,
            'source_hash': digest(hashes)}

    # Preserve raw hashes for audit. Shared scheduling files do not affect task identity.
    from .resume_identity import computation_hash, TRANSPORT, PREFIX
    parsed_sources = {PREFIX + 'classification/runner.py', PREFIX + 'common/runner.py'}
    parsed_sources.update(PREFIX + 'classification/' + name for name in (
        'babd_config.py','uci_config.py','bitcoin_config.py','elliptic_config.py',
        'tabpfn_config.py','completeness_config.py','config.py'))
    result['computation_sources'] = {
        path: (computation_hash(path, (root/path).read_text())
               if path in parsed_sources else raw_hash)
        for path, raw_hash in hashes.items() if path not in TRANSPORT}
    return result
