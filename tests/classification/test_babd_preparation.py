"""Guard tests for the dataset preparation CLI; no real dataset is required."""
from pathlib import Path
import os
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/prepare_babd.py'


def invoke(*args, env=None):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, env=env)


def test_help():
    result = invoke('--help')
    assert result.returncode == 0
    assert '--output-dir' in result.stdout


def test_missing_input_has_no_output(tmp_path):
    output = tmp_path / 'output'
    result = invoke('--input', tmp_path / 'missing.csv', '--output-dir', output)
    assert result.returncode != 0
    assert 'Input file not found' in result.stderr
    assert not output.exists()


def test_wrong_checksum_has_no_output(tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('account,SW,label\na,SA,0\n')
    output = tmp_path / 'output'
    result = invoke('--input', source, '--output-dir', output)
    assert result.returncode != 0
    assert 'checksum differs' in result.stderr
    assert not output.exists()
    assert source.read_text() == 'account,SW,label\na,SA,0\n'


def test_existing_output_is_untouched(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    marker = output / 'BABD-13-clean.csv'
    marker.write_bytes(b'preserve existing data')
    result = invoke('--input', tmp_path / 'missing.csv', '--output-dir', output)
    assert result.returncode != 0
    assert 'Output directory already exists' in result.stderr
    assert marker.read_bytes() == b'preserve existing data'
    assert sorted(p.name for p in output.iterdir()) == ['BABD-13-clean.csv']


def test_default_output_uses_data_root(tmp_path):
    output = tmp_path / 'raw/public/babd13'
    output.mkdir(parents=True)
    env = dict(os.environ, OEO_DATA_ROOT=str(tmp_path))
    result = invoke('--input', tmp_path / 'missing.csv', env=env)
    assert result.returncode != 0
    assert str(output) in result.stderr
    assert 'Output directory already exists' in result.stderr
