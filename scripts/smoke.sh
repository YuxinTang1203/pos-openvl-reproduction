#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../source"
python -m pytest -q tests/test_v17_ablation.py tests/test_fusion_v17.py tests/test_openvl_routed.py tests/test_sam3_service.py tests/test_sam3_device.py
