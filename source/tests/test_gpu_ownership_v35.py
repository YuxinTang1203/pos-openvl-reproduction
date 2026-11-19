import pytest
from pyrualean.gpu_ownership_v35 import parse_inventory,can_start,assert_visible_gpu_owned

def xml(process=''):
    return '<nvidia_smi_log><gpu><uuid>GPU-test</uuid><fb_memory_usage><free>24000 MiB</free></fb_memory_usage><processes>'+process+'</processes></gpu></nvidia_smi_log>'

def test_empty_card_admitted_and_foreign_compute_blocks_even_with_free_memory():
    empty=parse_inventory(xml(),lambda p:0)[0]
    assert can_start(empty)
    busy=parse_inventory(xml('<process_info><pid>12</pid><type>C</type></process_info>'),lambda p:1014)[0]
    assert not can_start(busy)

def test_unknown_uid_and_graphics_context_block_new_child():
    def unavailable(pid):raise PermissionError('not readable')
    row=parse_inventory(xml('<process_info><pid>12</pid><type>C+G</type></process_info>'),unavailable)[0]
    assert row['processes'][0]['uid'] is None and not can_start(row)
    row=parse_inventory(xml('<process_info><pid>12</pid><type>G</type></process_info>'),lambda p:1014)[0]
    assert not can_start(row)

def test_resident_service_card_needs_explicit_release():
    from pyrualean.gpu_ownership_v35 import RESIDENT_UUIDS
    row=parse_inventory(xml(),lambda p:0)[0];row['uuid']=next(iter(RESIDENT_UUIDS))
    assert not can_start(row)
    assert can_start(row,released=[row['uuid']])

def test_logical_device_must_map_to_exact_admitted_uuid(monkeypatch):
    monkeypatch.setenv('PYRUA_ADMITTED_GPU_UUID','GPU-test');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError,match='physical UUID'):assert_visible_gpu_owned()


@pytest.mark.parametrize('current_owner', ['gone', 'unknown', 'foreign'])
def test_transient_missing_pid_is_rechecked_without_accepting_unknown_owner(monkeypatch, current_owner):
    import os
    import pyrualean.gpu_ownership_v35 as ownership
    monkeypatch.setenv('PYRUA_ADMITTED_GPU_UUID', 'GPU-test')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    processes = [] if current_owner == 'gone' else [{'pid': 12, 'uid': None if current_owner == 'unknown' else os.getuid()+1}]
    snapshots = iter([[{'uuid': 'GPU-test', 'processes': [{'pid': 12, 'uid': None}]}],
                      [{'uuid': 'GPU-test', 'processes': processes}]])
    monkeypatch.setattr(ownership, 'inventory', lambda: next(snapshots))
    if current_owner == 'gone':
        assert assert_visible_gpu_owned() == 'GPU-test'
    else:
        with pytest.raises(RuntimeError, match='Foreign or unknown'):
            assert_visible_gpu_owned()
