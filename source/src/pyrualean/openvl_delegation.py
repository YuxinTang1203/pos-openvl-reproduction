"""PyRUA's explicit evidence delegation; upstream generation/execution stay native."""
import copy
import json


def prepare(adapter, args):
    from openvlharness.memory.run_state import BindingsState
    refs = args.get('evidence_refs')
    if not isinstance(refs, list) or not 1 <= len(refs) <= 16 or any(not isinstance(r, str) or not r for r in refs):
        raise ValueError('evidence_refs must contain 1–16 stored binding ID strings')
    refs = list(dict.fromkeys(refs))
    if not isinstance(args.get('objective'), str) or not args['objective'].strip():
        raise ValueError('objective must be a nonempty analysis question')
    source = adapter.memory.state.bindings
    if any(r not in source.bindings or r not in source.descriptors for r in refs):
        raise ValueError('Unknown evidence binding ID; use openvl_bindings() or trace_find() to discover IDs')
    # Share native image output registry, but isolate the computational input bindings.
    memory = copy.copy(adapter.memory)
    memory.state = copy.copy(adapter.memory.state)
    memory.state.bindings = BindingsState()
    origins = []
    for ref in refs:
        value = copy.deepcopy(source.bindings[ref])
        context = {'current_target': False, 'status': 'historical_or_unverified'}
        if ref in adapter.bridge.refs:
            context['validity'] = adapter.bridge.validity(ref)
        elif any(r == ref for r, _ in adapter.trace.rows):
            value = adapter.trace.read(ref)
            context.update(value.get('temporal_context', {}))
        memory.state.bindings.bindings[ref] = value
        memory.state.bindings.descriptors[ref] = copy.deepcopy(source.descriptors[ref])
        origins.append({'reference': ref, 'source_tool': source.descriptors[ref].source_tool,
                        'temporal_context': context})
    scope = {'references': refs, 'retrieval': origins,
             'limits': ['Only explicitly requested structured bindings were retrieved; linked records are not automatically followed.',
                        'Image references in records are metadata, not an inspection of their pixels.',
                        'Historical evidence is not a current action target; correspondence and task effects require evidence.']}
    objective = args['objective'] + '\nDelegation scope: ' + json.dumps(scope) + '''
Compute using the supplied bindings; do not invent measurements or assume list
order establishes object identity. Do not issue robot actions. Print a final JSON
object with answer (text or structured explanation), calculations (list of computed results), evidence_refs (only
IDs in this scope), and limitations (list of uncertainty/missing evidence).
Report what you computed, not a claim that the robot task has succeeded.'''
    return memory, {'objective': objective}, scope


def assessment(data, scope):
    """Validate reporting/citations, not the truth of the generated conclusions."""
    if data.get('generation_error') or data.get('execution_error'):
        return {'status': 'unavailable', 'reason': 'Generation or execution failed'}
    try:
        output = data['printed_output'].strip()
        value = None
        # Accept pretty-printed final JSON, optionally preceded by computation logs.
        # Do not accept trailing non-JSON text or search arbitrary brace substrings.
        lines = output.splitlines()
        for start, line in enumerate(lines):
            if line.lstrip().startswith('{'):
                try:
                    value = json.loads('\n'.join(lines[start:]))
                    break
                except ValueError:
                    continue
        assert isinstance(value, dict) and isinstance(value['answer'], (str, dict)) and value['answer']
        assert isinstance(value['calculations'], list)
        assert isinstance(value['limitations'], list) and all(isinstance(x, str) for x in value['limitations'])
        refs = value['evidence_refs']
        assert isinstance(refs, list) and refs and all(isinstance(r,str) and r in scope['references'] for r in refs)
        return {'status': 'reported', 'report': value, 'conclusion_verified': False}
    except (KeyError, TypeError, ValueError, AssertionError, IndexError):
        return {'status': 'unavailable', 'reason': 'Malformed report or unsupported evidence citations'}
