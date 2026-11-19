"""Robot-specific presentation and Zoom arguments; upstream storage stays intact."""
import copy
import math


def zoom_arguments(args):
    args = copy.deepcopy(args)
    box = args.get('coords')
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('coords must be [x1,y1,x2,y2] in source-image 0–1000 coordinates')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
        raise ValueError('coords must contain four finite numbers in [0,1000]')
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('coords require x1<x2 and y1<y2')
    margin = args.get('padding', 50)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError('padding must be a finite nonnegative number; minimum effective margin is 50')
    args['padding'] = max(50, margin)
    return args, ({'requested_padding': margin, 'effective_padding': 50} if margin < 50 else None)


def present_feedback(payload, refs, *, automatic):
    out = copy.deepcopy(payload)
    temporal = out.get('evidence_temporal_context', {})
    out['evidence_temporal_context'] = {
        k: temporal[k] for k in ('assessment_step', 'assessment_revision', 'images') if k in temporal}
    out['evidence_temporal_context']['guidance'] = 'Old coordinates are historical after actions; evidence_current(ref) checks usability. Zoom retains source time.'
    scope = temporal.get('output_scope', {})
    out['evidence_temporal_context']['output_scope'] = {k: scope[k] for k in ('cell','phase','constraint') if k in scope}
    # Already deduplicated by compact_inventory: retain discoverable native IDs.
    # This directory is not a freshness assertion.
    out['evidence_refs'] = refs[-8:]
    if len(refs) > 8: out['omitted_refs'] = len(refs)-8
    update = out.pop('evidence_update', {})
    if automatic:
        if update.get('recovery_evidence'):
            failure = update['recovery_evidence']
            out['action_observation'] = {k: failure[k] for k in ('failure','guidance','association') if k in failure}
        # A measured arrival or gripper aperture does not establish object/task effects.
        if scope.get('actions'):
            out['effect_status'] = {
            'task_effect': 'unverified_by_this_adapter',
            'meaning': 'Execution results alone do not establish task completion; this adapter has not verified the task outcome.',
            'next_evidence': 'Use fresh public observations and evidence_compare(ref) for before/after measurements. Different query/view results do not establish object identity.'}
    return out
