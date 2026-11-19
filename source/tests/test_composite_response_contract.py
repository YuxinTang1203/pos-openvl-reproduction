"""V8 failed on declared lengths/extra fields; expose constraints before generation."""
import copy
import json

import pytest

from tests.test_composite_harness import setup
from tests.test_composite_accepted_review import review
from tests.test_pos_harness import Script


@pytest.mark.parametrize('role', ['pos_belief', 'pos_repair', 'pos_sentinel'])
def test_contract_is_derived_from_live_role_schema_without_touching_evidence(robo, tmp_path, role):
    kit = setup(robo, tmp_path, visual=False)
    h = kit.general_harness
    before = copy.deepcopy(h.belief)
    contract = json.loads(h.response_contract(role).split('COMPOSITE_RESPONSE_CONTRACT\n')[1])
    assert contract['role'] == role
    assert contract['constraints']['payload']['required'] == h.response_schema(role)['required']
    assert contract['constraints']['payload']['additionalProperties'] is False
    if role == 'pos_sentinel':
        assert contract['constraints']['payload.directive.next_step']['maxLength'] == 400
    else:
        assert contract['constraints']['payload.achievement[].target']['maxLength'] == 240
    assert len(h.response_contract(role)) < 5000
    assert h.belief == before


def test_contract_follows_schema_changes_instead_of_hardcoded_limits(robo, tmp_path, monkeypatch):
    h = setup(robo, tmp_path, visual=False).general_harness
    original = h.response_schema

    def changed(role):
        schema = original(role)
        schema['properties']['directive']['properties']['next_step']['maxLength'] = 123
        return schema

    monkeypatch.setattr(h, 'response_schema', changed)
    contract = json.loads(h.response_contract('pos_sentinel').split('COMPOSITE_RESPONSE_CONTRACT\n')[1])
    assert contract['constraints']['payload.directive.next_step']['maxLength'] == 123


@pytest.mark.parametrize('bad', ['extra_key', 'long_directive'])
def test_v8_invalid_reports_remain_rejected_without_host_rewrite(robo, tmp_path, bad):
    h = setup(robo, tmp_path, visual=False).general_harness
    h.options['repair_attempts'] = 0
    report = review()
    if bad == 'extra_key':
        report['progress_reason_note'] = 'Not declared by the original schema.'
    else:
        report['directive']['next_step'] = 'x' * 401
    original = copy.deepcopy(report)
    belief = copy.deepcopy(h.belief)
    script = Script(copy.deepcopy(h.belief), report)
    h.before_request(script)
    assert h.last_status == 'audit_rejected'
    assert h.revision == 0 and h.belief == belief and report == original
    assert all('COMPOSITE_RESPONSE_CONTRACT' in call['prompt'] for call in script.calls)
