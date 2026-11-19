"""Start the existing SAM3 backend on an explicitly selected, idle GPU."""
import argparse, os
from pathlib import Path
from pyrualean.gpu_ownership_v35 import inventory, can_start
p=argparse.ArgumentParser()
p.add_argument('--gpu-uuid',required=True)
p.add_argument('--port',type=int,default=8765)
p.add_argument('--min-free-mib',type=int,default=10000)
a=p.parse_args()
checkpoint=Path(os.environ.get('SAM3_CHECKPOINT_PATH','assets/sam3.pt')).resolve()
if not checkpoint.is_file(): raise SystemExit('Missing SAM3_CHECKPOINT_PATH')
card=next((g for g in inventory() if g['uuid']==a.gpu_uuid),None)
if card is None or not can_start(card,min_free_mib=a.min_free_mib):
 raise SystemExit('Requested GPU is unavailable, reserved, busy, or lacks memory')
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,PYRUA_ADMITTED_GPU_UUID=a.gpu_uuid,SAM3_CHECKPOINT_PATH=str(checkpoint))
import sys
sys.argv=['sam3_service','--gpu','0','--port',str(a.port)]
from pyrualean.hosts.sam3_service import main
main()
