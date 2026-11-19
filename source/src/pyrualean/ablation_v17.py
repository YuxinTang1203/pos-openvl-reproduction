"""Matched V17 experiment profiles; no new robot or model operation.

PoS uses ProgressLite directly. Fusion uses the existing OpenVL/PoS bridge.
The fixed reminders append only to the executor's instruction prompt, for every pair.
"""
from __future__ import annotations

import copy
import hashlib

from .minimal_fusion import MinimalFusionHarness, validate_options as validate_fusion
from .progress_lite import ProgressLiteHarness, validate_options as validate_pos
from .static_rule_reminders import STATIC_RULES

KIND = 'v17_ablation'
PROFILES = {'pos', 'fusion', 'next_action'}
ALLOWED = {'kind', 'profile', 'pos', 'openvl', 'static_rules', 'action_focus', 'reliability'}


def effective_config(config):
    """Translate explicit switches into an existing harness configuration."""
    if config['profile'] == 'pos':
        return {'kind': 'pos_progress_lite', **copy.deepcopy(config.get('pos', {}))}
    return {
        'kind': 'minimal_fusion', 'pos': copy.deepcopy(config.get('pos', {})),
        'openvl': copy.deepcopy(config['openvl']), 'dependency_mode': 'off',
        'observation_scope': 'next_action' if config['profile'] == 'next_action' else 'baseline',
        'action_focus': {'enabled': config.get('action_focus', False)},
        'reliability': {'enabled': config.get('reliability', False)},
    }


def validate_options(config):
    if not isinstance(config, dict) or config.get('kind') != KIND or set(config) - ALLOWED:
        raise ValueError('unknown V17 ablation option')
    if config.get('profile') not in PROFILES:
        raise ValueError('profile must be pos, fusion or next_action')
    for name in ('static_rules', 'action_focus', 'reliability'):
        if type(config.get(name, False)) is not bool:
            raise ValueError(name + ' must be boolean')
    if not isinstance(config.get('pos', {}), dict) or 'kind' in config.get('pos', {}):
        raise ValueError('pos must contain ProgressLite options without kind')
    if config['profile'] == 'pos':
        if 'openvl' in config or config.get('action_focus', False) or config.get('reliability', False):
            raise ValueError('pos control must not enable OpenVL, V15 focus or V15 reliability')
        validate_pos(effective_config(config))
    else:
        if not isinstance(config.get('openvl'), dict) or config['openvl'].get('enabled') is not True:
            raise ValueError('fusion profiles require explicit openvl.enabled=true')
        if config['profile'] == 'next_action' and config.get('action_focus', False):
            raise ValueError('test V15 focus separately from the next_action prompt')
        validate_fusion(effective_config(config))


class _StaticRules:
    def __init__(self, *args, static_rules=False, **kwargs):
        self.static_rules_enabled = static_rules
        super().__init__(*args, **kwargs)

    def prompt(self):
        base = super().prompt()
        return base + '\n' + STATIC_RULES if self.static_rules_enabled else base


class _PoSControl(_StaticRules, ProgressLiteHarness):
    pass


class _FusionControl(_StaticRules, MinimalFusionHarness):
    pass


def build_harness(kit, config, task, initial_images=()):
    validate_options(config)
    if config['profile'] == 'pos':
        cls = _PoSControl
    else:
        from .openvl_routed import OpenVLRoutedAdapter
        visual = config['openvl']
        kit.evidence_adapter = OpenVLRoutedAdapter(
            kit, config=visual['config'], initial_images=initial_images,
            ambiguity_score_margin=visual.get('ambiguity_score_margin', .10),
            duplicate_mask_iou=visual.get('duplicate_mask_iou', .90))
        cls = _FusionControl
    # No action wrapper, audit, tool, or auxiliary model call is added by this mixin.
    # Set the prompt switch before initialization in case an inherited init reads it.
    harness = cls(kit, effective_config(config), task, initial_images,
                  static_rules=config.get('static_rules', False))
    harness.provenance['v17_ablation'] = {
        'profile': config['profile'], 'static_rules': harness.static_rules_enabled,
        'static_rules_sha256': hashlib.sha256(STATIC_RULES.encode()).hexdigest(),
        'static_rules_injection': 'executor instruction prompt only; identical suffix across pairs',
        'v15_action_focus': config.get('action_focus', False),
        'v15_reliability': config.get('reliability', False),
        'dependency_mode': 'off', 'same_v17_projection_fix': True,
        'added_model_calls': 0, 'added_sensing_calls': 0, 'added_action_gates': False,
    }
    harness._log('v17_ablation_configured', config=harness.provenance['v17_ablation'])
    return harness
