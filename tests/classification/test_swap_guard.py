from oeo_lab.experiments.common.runner import memory_status,memory_admission
from oeo_lab.experiments.common.checkpoints import same_task


def test_high_swap_blocks_even_with_available_ram(tmp_path):
    p=tmp_path/'meminfo'
    p.write_text('MemTotal: 264241152 kB\nMemAvailable: 14680064 kB\nSwapTotal: 41943040 kB\nSwapFree: 4194304 kB\n')
    s=memory_status(p)
    assert s['total_gib']==252 and s['available_gib']==14 and s['swap_used_gib']==36
    assert s['headroom_gib']==-22 and not memory_admission(s,10)


def test_pause_and_resume_hysteresis():
    blocked=False
    # No refill while falling; resume only once the additional margin is met.
    decisions=[]
    for headroom in [20,9,10,14,15,11,9]:
        ok=memory_admission({'headroom_gib':headroom},10,blocked,5)
        decisions.append(ok);blocked=not ok
    assert decisions==[True,False,False,False,True,True,False]


def test_unknown_memory_blocks(tmp_path):
    assert memory_status(tmp_path/'absent') is None
    assert not memory_admission(None,10)


def test_no_swap_and_dynamic_total(tmp_path):
    p=tmp_path/'meminfo'
    p.write_text('MemTotal: 33554432 kB\nMemAvailable: 16777216 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
    s=memory_status(p)
    assert s['total_gib']==32 and memory_admission(s,10)


def test_guard_setting_not_scientific_identity():
    a={'provenance':{'settings':{}}}
    b={'provenance':{'settings':{'memory_resume_margin_gb':5}}}
    assert same_task(a,b)
