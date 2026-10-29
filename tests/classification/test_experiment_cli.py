"""Dispatcher tests: defaults and scientific settings belong to the modules."""
from types import SimpleNamespace
from unittest.mock import patch
import sys
import pytest
from oeo_lab.experiments import cli


@pytest.mark.parametrize("name", list(cli.EXPERIMENTS))
def test_forwards_arguments_and_status(name):
    with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
        assert cli.main([name, "--n-jobs", "30", "--dry-run"]) == 7
    command = run.call_args.args[0]
    assert command == [sys.executable, "-m", cli.EXPERIMENTS[name][1],
                       "--n-jobs", "30", "--dry-run"]
    assert "PYTHONPATH" in run.call_args.kwargs["env"]


def test_menu_uses_original_module_defaults():
    with patch.object(sys.stdin, "isatty", return_value=True), \
         patch("builtins.input", return_value="1"), \
         patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        assert cli.main(["--no-watch"]) == 0
    assert run.call_args.args[0] == [
        sys.executable, "-m", cli.EXPERIMENTS["completeness"][1]]


def test_noninteractive_does_not_start_experiment():
    with patch.object(sys.stdin, "isatty", return_value=False), \
         patch.object(cli.subprocess, "run") as run:
        assert cli.main([]) == 2
    run.assert_not_called()


def test_unknown_experiment_does_not_start():
    with patch.object(cli.subprocess, "run") as run, pytest.raises(SystemExit) as error:
        cli.main(["unknown"])
    assert error.value.code == 2
    run.assert_not_called()


def test_experiment_help_is_forwarded():
    with patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        assert cli.main(["uci", "--help"]) == 0
    assert run.call_args.args[0][-1] == "--help"


def test_batch_stops_on_failure():
    with patch.object(cli.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=0), SimpleNamespace(returncode=3)]) as run:
        assert cli.main(["--no-watch", "--batch", "uci", "elliptic", "babd"]) == 3
    assert run.call_count == 2
    assert run.call_args_list[1].args[0][2] == cli.EXPERIMENTS["elliptic"][1]


def test_all_defaults_to_one_guarded_batch():
    with patch.object(cli, "start_watch", return_value=0) as start:
        assert cli.main(["all"]) == 0
    start.assert_called_once_with(list(cli.EXPERIMENTS), [])


def test_batch_dry_run_never_registers_service():
    with patch.object(cli, "direct_run", return_value=0) as run, \
         patch.object(cli, "start_watch") as start:
        assert cli.main(["--batch", "uci", "elliptic", "--dry-run"]) == 0
    run.assert_called_once_with(["uci", "elliptic"], ["--dry-run"])
    start.assert_not_called()


def test_guard_command_and_resume(tmp_path):
    with patch.object(cli.Path, "home", return_value=tmp_path), \
         patch.object(cli.shutil, "which", return_value="/public/experiment-watch"), \
         patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        assert cli.start_watch(["uci", "elliptic"], ["--n-jobs", "2"]) == 0
        call = run.call_args.args[0]
        command = call[call.index("--")+1:]
        assert "--no-watch" in command
        assert any(v.startswith("--watch-run=") for v in command)
        assert command[-5:] == ["--batch", "uci", "elliptic", "--n-jobs", "2"]
        name = call[call.index("--name")+1]
        config = tmp_path / ".config/experiment-watch" / (name+".json")
        config.parent.mkdir(parents=True)
        config.write_text(cli.json.dumps({"command": command, "cwd": str(cli.Path.cwd())}))
        assert cli.start_watch(["uci", "elliptic"], ["--n-jobs", "2"]) == 0
        assert run.call_args.args[0] == ["/public/experiment-watch", "start", "--name", name]


def test_missing_guard_is_explicit(tmp_path):
    with patch.object(cli.Path, "home", return_value=tmp_path), \
         patch.object(cli.shutil, "which", return_value=None):
        assert cli.start_watch(["uci"], []) == 2
