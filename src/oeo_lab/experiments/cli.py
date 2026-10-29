"""Unified selection and optional supervision; scientific settings stay in modules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

EXPERIMENTS = {
    "completeness": ("规则完备性", "oeo_lab.experiments.classification.completeness"),
    "uci": ("UCI 分类", "oeo_lab.experiments.classification.uci_experiment"),
    "babd": ("BABD 分类", "oeo_lab.experiments.classification.babd_experiment"),
    "training-size": ("训练规模", "oeo_lab.experiments.classification.training_size_experiment"),
    "elliptic": ("Elliptic 合法/非法交易二分类", "oeo_lab.experiments.classification.elliptic_experiment"),
}


def direct_run(names, forwarded):
    """Run sequentially; stop on failure so the supervisor can recover."""
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for index, name in enumerate(names, 1):
        print(f"[实验 {index}/{len(names)}] {name}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", EXPERIMENTS[name][1], *forwarded],
            env=env, check=False)
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


def start_watch(names, forwarded):
    """Delegate to the installed public service, without copying its implementation."""
    tool = shutil.which("experiment-watch")
    if not tool:
        fallback = Path.home() / ".local/bin/experiment-watch"
        tool = str(fallback) if fallback.is_file() else None
    if not tool:
        print("未找到公共 experiment-watch；请安装，或添加 --no-watch 直接运行。",
              file=sys.stderr)
        return 2
    launcher = Path(__file__).resolve().parents[3] / "run_experiment.py"
    if not launcher.is_file():
        print("守护模式需要源码工程中的 run_experiment.py；可使用 --no-watch。",
              file=sys.stderr)
        return 2
    cwd = str(Path.cwd())
    # Stable service identity is independent of scientific checkpoint identity.
    identity = [sys.executable, str(launcher), cwd, names, forwarded]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:12]
    name = "fca-" + (names[0] if len(names) == 1 else "batch") + "-" + digest
    marker = "--watch-run=" + digest
    selection = [names[0]] if len(names) == 1 else ["--batch", *names]
    command = [sys.executable, str(launcher), "--no-watch", marker,
               *selection, *forwarded]
    config_path = Path.home() / ".config/experiment-watch" / (name + ".json")
    call = [tool, "start", "--name", name]
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get("command") != command or config.get("cwd") != cwd:
            print("已有守护配置与请求不一致，请检查：" + str(config_path), file=sys.stderr)
            return 2
        # Public service resumes a registered instance without replacing its command.
    else:
        call += ["--cwd", cwd, "--match-token=" + marker, "--", *command]
    result = subprocess.run(call, check=False)
    if result.returncode == 0:
        print("守护任务：" + name)
        print("查看：tmux attach -t watch-" + name)
        print("状态：" + shlex.join([tool, "status", "--name", name]))
        print("停止：" + shlex.join([tool, "stop", "--name", name]))
        print("日志：" + str(Path.home() / ".local/state/experiment-watch" / name))
    return result.returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="默认使用公共守护程序；--no-watch 直接运行。",
        epilog="批量：--batch uci elliptic；全部：all。批量参数会传给每组实验。")
    parser.add_argument("--no-watch", action="store_true", help="取消守护")
    parser.add_argument("--batch", nargs="+", choices=list(EXPERIMENTS),
                        help="按指定顺序执行多组实验")
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS, "all"])
    # These wrapper-only arguments are removed before forwarding to experiment parsers.
    no_watch = "--no-watch" in argv
    argv = [v for v in argv if v != "--no-watch" and not v.startswith("--watch-run=")]
    if not argv:
        if not sys.stdin.isatty():
            parser.print_help()
            return 2
        names = list(EXPERIMENTS)
        for index, name in enumerate(names, 1):
            print(f"{index}. {EXPERIMENTS[name][0]} ({name})")
        try:
            choice = input("编号/名称（空格分隔可多选，all 全部，q 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            return 130
        if choice.lower() == "q":
            return 0
        chosen = [names[int(v)-1] if v.isdigit() and 1 <= int(v) <= len(names)
                  else v for v in choice.split()]
        argv = chosen if len(chosen) == 1 else ["--batch", *chosen]
    if argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] == "--batch":
        selected, forwarded = [], argv[1:]
        while forwarded and not forwarded[0].startswith("-"):
            selected.append(forwarded.pop(0))
        if not selected:
            parser.error("--batch 需要实验名称")
    else:
        selected, forwarded = ([*EXPERIMENTS] if argv[0] == "all" else [argv[0]]), argv[1:]
    if any(name not in EXPERIMENTS for name in selected):
        parser.error("未知实验名称")
    selected = list(dict.fromkeys(selected))
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    # Inspection commands should never register services or restart automatically.
    inspection = any(v in ("-h", "--help", "--dry-run") for v in forwarded)
    try:
        if no_watch or inspection:
            return direct_run(selected, forwarded)
        return start_watch(selected, forwarded)
    except KeyboardInterrupt:
        return 130
