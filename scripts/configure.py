"""Generate matched PyRUA baseline / PoS+OpenVL configs and policy copies."""
import argparse
import json
import shutil
from pathlib import Path

import yaml
from pyrualean.migrations import validate_migration

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=Path, help='RoboDojo root containing XPolicyLab')
parser.add_argument('--variant', choices=['baseline', 'fusion'], default='fusion')
args = parser.parse_args()
if args.variant == 'baseline':
    config = {'kind': 'baseline'}
    policy_name = 'PyRUA_Baseline'
    output_name = 'runtime-resolved-baseline.json'
else:
    config = json.loads((ROOT / 'source/integrations/migrations/general_harness/ablation_fusion.json').read_text())
    config['openvl']['config'] = str(ROOT / 'source/integrations/migrations/openvl/capabilities_visual.yml')
    policy_name = 'PyRUA_Reproduction'
    output_name = 'runtime-resolved.json'
validate_migration(config)
# Check the destination before writing or copying anything.
destination = None
if args.benchmark:
    benchmark = args.benchmark.resolve()
    if not (benchmark / 'XPolicyLab/setup_policy_server.py').is_file():
        raise SystemExit('Expected a provisioned RoboDojo/XPolicyLab benchmark')
    destination = benchmark / 'XPolicyLab/policy' / policy_name
    if destination.exists():
        raise SystemExit('Refuse overwrite: ' + str(destination))
output = ROOT / 'configs' / output_name
output.write_text(json.dumps(config, indent=2) + '\n')
print('Generated', output)
if destination:
    shutil.copytree(ROOT / 'source/integrations/xpolicylab/PyRUA_Lean', destination)
    deployment = yaml.safe_load((destination / 'deploy.yml').read_text())
    deployment.update(policy_name=policy_name, migration=config,
                      result_dir=str(ROOT / 'outputs' / args.variant))
    (destination / 'deploy.yml').write_text(yaml.safe_dump(deployment, sort_keys=False))
    print('Installed policy', destination)
