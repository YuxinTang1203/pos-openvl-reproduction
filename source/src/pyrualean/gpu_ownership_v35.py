"""GPU ownership admission, keyed by physical UUID, never by free memory alone."""
from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET

# Project AGENTS identifies these as resident model-service cards. An empty
# moment is not a release; explicit coordination is required before reuse.
RESIDENT_UUIDS={
    'GPU-e5aabee3-f710-bdda-3eec-9638ee78e870',
    'GPU-02cec457-47cc-7d3c-b353-7b9e118ca7f8',
    'GPU-8243dcba-720a-499e-5ddf-a53cbb512d70',
    'GPU-145620b5-e050-966b-437a-2688c5f38f8b',
}


def parse_inventory(xml, uid_lookup):
    root=ET.fromstring(xml);out=[]
    for index,g in enumerate(root.findall('gpu')):
        processes=[]
        for p in g.findall('./processes/process_info'):
            pid=int(p.findtext('pid'))
            try:uid=uid_lookup(pid)
            except OSError:uid=None
            processes.append({'pid':pid,'type':p.findtext('type'),'uid':uid})
        free=g.findtext('./fb_memory_usage/free') or '0 MiB'
        out.append({'index':index,'uuid':g.findtext('uuid'),'free_mib':int(free.split()[0]),'processes':processes})
    return out


def inventory():
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=20)
    return parse_inventory(xml,lambda pid:Path('/proc',str(pid)).stat().st_uid)


def can_start(gpu, *, released=(), min_free_mib=12000):
    # Starting a new child requires an entirely idle card, including same-UID
    # unrelated work. Per-user UUID locks provide a second, cooperative guard.
    if gpu['uuid'] in RESIDENT_UUIDS and gpu['uuid'] not in released:return False
    return not gpu['processes'] and gpu['free_mib']>=min_free_mib


def assert_visible_gpu_owned():
    expected=os.environ.get('PYRUA_ADMITTED_GPU_UUID')
    if not expected or not expected.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:
        raise RuntimeError('GPU model requires one explicitly admitted physical UUID; no CPU fallback')
    rows=[g for g in inventory() if g['uuid']==expected]
    if len(rows)!=1:raise RuntimeError('Admitted GPU UUID is unavailable')
    if any(p['uid'] is None for p in rows[0]['processes']):
        # A short-lived peer may exit between nvidia-smi and /proc lookup.
        # Recheck the inventory while retaining rejection of live unknown UIDs.
        rows=[g for g in inventory() if g['uuid']==expected]
        if len(rows)!=1:raise RuntimeError('Admitted GPU UUID is unavailable')
    for p in rows[0]['processes']:
        if p['uid'] is None or p['uid']!=os.getuid():
            raise RuntimeError('Foreign or unknown GPU process appeared; do not start/continue model inference')
    return expected
