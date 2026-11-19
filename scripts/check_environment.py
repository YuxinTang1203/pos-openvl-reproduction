"""Report requirements without downloading models or calling an LLM."""
import argparse, importlib.util, json, sys, shutil, os
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--gpu',action='store_true'); a=p.parse_args()
checks={'python>=3.10':sys.version_info>=(3,10)}
for name in ['numpy','PIL','pytest','yaml','httpx','mcp','pyrualean','openvlharness','pycocotools','cv2','matplotlib','scipy','torch','torchvision']:
 checks[name]=importlib.util.find_spec(name) is not None
if a.gpu:
 for name in ['torch','torchvision','sam3','openvlharness']:
  checks[name]=importlib.util.find_spec(name) is not None
 checks['nvidia-smi']=shutil.which('nvidia-smi') is not None
 checks['sam3_checkpoint']=Path(os.environ.get('SAM3_CHECKPOINT_PATH','assets/sam3.pt')).is_file()
 if checks.get('torch'):
  import torch
  checks['cuda_available']=torch.cuda.is_available()
print(json.dumps({'python':sys.version,'checks':checks},indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
