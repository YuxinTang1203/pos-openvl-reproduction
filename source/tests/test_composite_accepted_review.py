"""Exercise actual accepted Sentinel reports, not only rejected top-level issues."""
import copy
import json

from tests.test_composite_harness import setup, args
from tests.test_pos_harness import Script, audit


def review(issues=()):
    return {**audit(None), "issues": list(issues), "directive": {
        "goal_id": "a-root", "mode": "observe",
        "next_step": "Inspect the relevant original view.",
        "decision_critical_missing": "Placement is unverified.",
        "why_change": "Public outcome evidence is required."}}


def test_accepted_caveats_reach_executor_and_next_pos_payload(robo, tmp_path):
    kit = setup(robo, tmp_path, visual=False)
    h = kit.general_harness
    issue = 'Projection is conditional on the supplied plane, not a measured height.'
    h.before_request(Script(copy.deepcopy(h.belief), review([issue])))
    assert h.revision == 1
    assert h.decision_context()['maintenance_feedback']['issues'] == [issue]
    assert h.decision_context()['last_committed_review']['issues'] == [issue]
    out = kit.execute_tool('python', args('print(robo.state())'))
    assert out.result['component_review_feedback']['issues'] == [issue]
    assert issue in h.evidence['obs-1']['text']
    model = Script(copy.deepcopy(h.belief), review())
    h.before_request(model)
    assert all(issue in json.dumps(c['payload']) for c in model.calls)


def test_unavailable_review_keeps_previous_accepted_caveats_historical(robo, tmp_path):
    kit = setup(robo, tmp_path, visual=False)
    h = kit.general_harness
    issue = 'Object identity remains an inference.'
    h.before_request(Script(copy.deepcopy(h.belief), review([issue])))
    original = copy.deepcopy(h.decision_context()['last_committed_review'])
    kit.execute_tool('python', args('robo.move_to([0,0,.6])'))
    before = copy.deepcopy(h.belief)
    h.before_request(Script(copy.deepcopy(h.belief), None))
    ctx = h.decision_context()
    assert ctx['maintenance_feedback']['status'] == 'audit_unavailable'
    assert ctx['maintenance_feedback']['issues'] == []
    assert ctx['last_committed_review'] == original
    assert original['motion_epoch'] < ctx['current_motion_epoch']
    assert h.belief == before
    out = kit.execute_tool('python', args('print(robo.state())'))
    assert out.result['last_committed_review'] == original
    assert json.loads(h.evidence['obs-2']['text'])['last_committed_review'] == original


def test_new_accepted_review_replaces_old_caveats_and_copies_are_isolated(robo, tmp_path):
    kit = setup(robo, tmp_path, visual=False)
    h = kit.general_harness
    h.before_request(Script(copy.deepcopy(h.belief), review(['old'])))
    kit.execute_tool('python', args('print(robo.state())'))
    h.before_request(Script(copy.deepcopy(h.belief), review()))
    ctx = h.decision_context()
    assert ctx['last_committed_review']['issues'] == []
    ctx['last_committed_review']['issues'].append('tampered')
    assert h.decision_context()['last_committed_review']['issues'] == []


def test_accepted_report_obeys_existing_diagnostic_size_bounds(robo, tmp_path):
    kit = setup(robo, tmp_path, visual=False)
    h = kit.general_harness
    h._log('committed', audit={'accepted': True, 'issues': ['x' * 1500] * 11})
    report = h.decision_context()['last_committed_review']
    assert len(report['issues']) == 8
    assert all(len(x) == 800 for x in report['issues'])
    assert report['omitted_issues'] == 3
