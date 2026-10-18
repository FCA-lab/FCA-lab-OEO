from pathlib import Path
from oeo_lab.experiments.common.resource_guard import memory_status,memory_admission
from oeo_lab.experiments.common.scheduling_config import (
    MEMORY_RESERVE_GIB,MEMORY_RESUME_MARGIN_GIB,effective_settings)
from oeo_lab.experiments.common.checkpoints import same_task


def test_shared_settings_override_legacy_experiment_values():
    assert MEMORY_RESERVE_GIB==10 and MEMORY_RESUME_MARGIN_GIB==5
    for settings in ({'min_free_gb':2,'n_jobs':3}, {'n_jobs':30},{}):
        result=effective_settings(settings)
        assert result['min_free_gb']==10
        assert result['memory_resume_margin_gb']==5
        assert result.get('n_jobs')==settings.get('n_jobs')


def test_shared_guard_under_swap_pressure(tmp_path):
    path=tmp_path/'meminfo'
    path.write_text('MemTotal: 264241152 kB\nMemAvailable: 14680064 kB\n'
                    'SwapTotal: 41943040 kB\nSwapFree: 4194304 kB\n')
    snapshot=memory_status(path)
    assert snapshot['headroom_gib']==-22
    assert not memory_admission(snapshot,MEMORY_RESERVE_GIB)
    assert not memory_admission({'headroom_gib':14},10,True,5)
    assert memory_admission({'headroom_gib':15},10,True,5)


def test_other_experiment_identity_ignores_only_resource_policy():
    base={'dataset':{'sha256':'a'},'provenance':{'settings':{'min_free_gb':2}}}
    newer={'dataset':{'sha256':'a'},'provenance':{'settings':{
        'min_free_gb':10,'memory_resume_margin_gb':5}}}
    assert same_task(base,newer)
    newer['dataset']['sha256']='b'
    assert not same_task(base,newer)
