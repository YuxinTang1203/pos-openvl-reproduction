"""Model-facing temporal labels; annotations never execute or validate robot actions."""

from copy import deepcopy

TEMPORAL_GUIDANCE = (
    "Coordinates printed in earlier stdout/code and images in earlier messages are historical. "
    "Freshness labels are snapshots at the stated assessment step, not permanent approvals. "
    "After intervening actions, use evidence_current(ref); if unusable or no reference exists, "
    "re-localize before reusing coordinates as a current target. Unknown time means unverified, "
    "not fresh. A crop/Zoom preserves its source capture time and cannot verify a later action. "
    "Use a new post-action observation to assess that action's outcome."
)


def annotate_replay_history(records, images, *, cutoff_message):
    """Label archived cells/images without inventing missing acquisition steps.

    Message order proves delivery order only. It does not establish when an image
    was captured, whether a printed coordinate is a measurement, or task success.
    """
    rows, pictures = deepcopy(records), deepcopy(images)
    if any(r['message_index'] > cutoff_message for r in rows):
        raise ValueError('History includes a record after the cutoff')
    for row in rows:
        row['temporal_context'] = {
            'source_cell': row.get('cell'),
            'source_message': row['message_index'],
            'assessment_message': cutoff_message,
            'source_step': None,
            'current_target': False,
            'status': 'historical_current_validity_unverified',
            'later_cells': [r.get('cell') for r in rows
                            if r['message_index'] > row['message_index']],
            'scope': 'code and feedback, including literal coordinates; not fresh measurements',
            'next_step': 'evidence_current(ref) if linked; otherwise re-localize before targeting',
        }
    for picture in pictures:
        index = picture['message_index']
        # A user image message can immediately follow the cutoff tool response.
        if index > cutoff_message + 1:
            raise ValueError('Image delivery is beyond the checkpoint')
        preceding = [r for r in rows if r['message_index'] < index]
        picture['temporal_context'] = {
            'delivered_after_cell': preceding[-1].get('cell') if preceding else None,
            'delivery_message': index,
            'assessment_message': cutoff_message,
            'capture_step': None,
            'capture_phase': 'unknown; delivery is not acquisition',
            'later_cells': [r.get('cell') for r in rows if r['message_index'] > index],
            'current_target': False,
            'status': 'historical_current_validity_unverified',
            'constraint': 'Cannot verify later cells. Zoom does not refresh this image.',
        }
    return {'records': rows, 'images': pictures, 'temporal_guidance': TEMPORAL_GUIDANCE}


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Annotate a frozen replay INPUT.json offline')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    data.update(annotate_replay_history(data['records'], data['images'],
                                        cutoff_message=data['cutoff_message']))
    # Exclusive create preserves the original input and previous experiment artifacts.
    with args.output.open('x') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
