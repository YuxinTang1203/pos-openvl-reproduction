"""Opt-in completion review using the upstream Coding Agent and local executor.

Robot adaptation supplies public evidence; the review never sees the policy's
success claim, reward predicates or hidden simulator object states.
"""
import copy
import json
import time
from pathlib import Path

OBJECTIVE = '''Independently assess whether the ORIGINAL TASK in public_evidence is
supported by the available public observations. Generate and execute Python checks
using public_evidence and any supplied image arrays. Do not assume execution of a
command proves its intended effect. Do not invent missing measurements or object
identity. Return unknown for requirements that cannot be established. Do not treat
missing evidence as a contradiction. No robot actions may be issued.
Print exactly one JSON object with keys verdict (satisfied/not_satisfied/unknown),
checks (nonempty list of objects with requirement, status, evidence), and explanation.
Each check.status MUST also be exactly satisfied, not_satisfied, or unknown; do
not use synonyms such as supported. Distinguish initial, intermediate, and current
observations using their temporal order. Historical positions are not final state.
A satisfied verdict requires every task requirement to be supported; any explicit
contradiction means not_satisfied; otherwise use unknown. Compute checks from the
supplied values where possible, and state limits of visual inference.'''


def parse_review(output):
    lines = [x.strip() for x in output.splitlines() if x.strip()]
    try:
        value = json.loads(lines[-1])
        assert value['verdict'] in {'satisfied', 'not_satisfied', 'unknown'}
        checks = value['checks']
        assert isinstance(checks, list) and checks
        for c in checks:
            assert isinstance(c, dict) and c['status'] in {'satisfied', 'not_satisfied', 'unknown'}
            assert isinstance(c['requirement'], str) and c['requirement'].strip()
            assert c.get('evidence')
        statuses = {c['status'] for c in checks}
        expected = 'not_satisfied' if 'not_satisfied' in statuses else 'unknown' if 'unknown' in statuses else 'satisfied'
        assert value['verdict'] == expected
        return value
    except (ValueError, KeyError, TypeError, AssertionError, IndexError):
        return {'verdict': 'unknown', 'checks': [], 'explanation': 'Invalid or inconsistent reviewer output.'}


class GatewayModel:
    """Transport adapter for native generator; optional public images add vision."""
    def __init__(self, base_url, api_key, model, directory, images=(), timeout=120):
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.directory, self.images, self.timeout = Path(directory), images, timeout

    def generate_text(self, messages, max_tokens=4096, inherit_options=False):
        from .openai_stream import streaming_transport
        from .openai_runtime import image_part
        messages = copy.deepcopy(messages)
        if self.images:
            parts = [{'type': 'text', 'text': messages[-1]['content']}]
            for label, data in self.images:
                parts += [{'type': 'text', 'text': label}, image_part(data)]
            messages[-1]['content'] = parts
        body = dict(model=self.model, messages=messages, temperature=0, top_p=1.0, max_tokens=max_tokens,
                    chat_template_kwargs={'enable_thinking': False})
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory/'request.json').write_text(json.dumps(body)+'\n')
        def event(kind, **fields):
            with (self.directory/'network.jsonl').open('a') as f:
                f.write(json.dumps(dict(type=kind, **fields))+'\n')
        code, response = streaming_transport(self.api_key, event)(
            self.base_url.rstrip('/')+'/chat/completions', body, self.timeout)
        (self.directory/'response.json').write_text(json.dumps(response, default=str)+'\n')
        if code != 200:
            raise RuntimeError(f'Reviewer HTTP status {code}')
        choice = response['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError('Reviewer generation incomplete')
        return choice['message']['content']


def review_bundle(bundle, model, directory, execution_timeout=15):
    """Real native generation and execution; injected model also enables offline tests."""
    from openvlharness.services.tools.python_coding_agent.tool import Python_Coding_Agent_Tool
    from openvlharness.services.tools.python_executor_local.executor import prepare_execution_context, execute_python_code
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {'verdict': 'unknown', 'checks': [], 'explanation': 'Review unavailable'}
    try:
        # Actual scalar evidence is in the objective, not only type descriptors.
        bindings = {'public_evidence': copy.deepcopy(bundle)}
        image_notes = []
        for i, (label, data) in enumerate(getattr(model, 'images', ())):
            import io
            import numpy as np
            from PIL import Image
            bindings[f'image_{i}'] = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
            image_notes.append(f"image_{i}: RGB uint8 array {bindings[f'image_{i}'].shape}, {label}")
        objective = OBJECTIVE + '\nArray bindings: ' + '; '.join(image_notes) + '\npublic_evidence = ' + json.dumps(bundle, default=str)
        generated = Python_Coding_Agent_Tool(model).execute(objective)
        (directory/'generation.json').write_text(json.dumps(generated, default=str)+'\n')
        if generated.get('error') or not generated.get('generated_code'):
            raise ValueError('Native coding generation failed')
        executed = execute_python_code(generated['generated_code'],
            prepare_execution_context(bindings),
            timeout_seconds=execution_timeout)
        (directory/'execution.json').write_text(json.dumps(executed, default=str)+'\n')
        if executed.get('execution_error'):
            raise ValueError('Native generated-code execution failed')
        result = parse_review(executed.get('printed_output', ''))
    except Exception as exc:
        # No exception text containing provider credentials is forwarded to policy.
        result = {'verdict': 'unknown', 'checks': [], 'explanation': f'Review failed: {type(exc).__name__}'}
    result['elapsed_s'] = round(time.monotonic()-started, 3)
    (directory/'review.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


class CompletionReviewer:
    def __init__(self, adapter, task, initial_images, base_url, api_key, model, mode):
        self.adapter, self.task, self.initial_images = adapter, task, dict(initial_images)
        self.base_url, self.api_key, self.model, self.mode = base_url, api_key, model, mode
        self.count = 0

    def review(self):
        a = self.adapter
        if self.count >= 2:
            return {'verdict':'unknown', 'checks':[], 'explanation':'Completion review limit reached', 'exhausted':True}
        left = a.remaining() if a.remaining else 120
        if left <= 2 or (a.stopped and a.stopped()):
            return {'verdict':'unknown', 'checks':[], 'explanation':'Episode budget ended', 'exhausted':True}
        self.count += 1
        directory = a.root/'completion_review'/f'{self.count:02d}'
        try:
            bundle = {'task':self.task, 'perceptions':[], 'actions':[]}
            for ref in a.bridge.refs:
                if ref in a.bridge.entries:
                    v = a.bridge.values[ref]
                    bundle['perceptions'].append({'reference':ref, 'query':a.bridge.entries[ref]['query'], 'measurement':{k:v[k] for k in ('found','center_xyz','top_z','box_px','centroid_rc','geometry_error','source') if k in v}, 'validity':a.bridge.validity(ref)})
            # Bounds are explicit: no claim that omitted history was verified.
            bundle['omitted_perceptions'] = max(0,len(bundle['perceptions'])-24)
            bundle['perceptions'] = bundle['perceptions'][-24:]
            records = a.kit.robo._ledger.records
            for record in records:
                if record.stateful:
                    bundle['actions'].append({'index':record.index,'tool':record.tool,
                        'arguments':record.kwargs,'result':record.summary,'ok':record.ok,'error':record.error})
            bundle['omitted_actions'] = max(0,len(bundle['actions'])-80)
            bundle['actions'] = bundle['actions'][-80:]
            from .arms import _to_png
            images = []
            if self.initial_images:
                name, data = next(iter(self.initial_images.items()))
                images.append((f'Initial public observation: {name}; historical', data))
            camera = a.kit.robo.CAMERAS[0]
            images.append((f'Current public observation: {camera}', _to_png(a.kit.robo.image(camera))))
            transport = GatewayModel(self.base_url,self.api_key,self.model,directory,images,timeout=min(120,max(1,left-16)))
            return review_bundle(bundle,transport,directory,execution_timeout=min(15,max(1,left-1)))
        except Exception as exc:
            return {'verdict':'unknown','checks':[],'explanation':f'Evidence collection failed: {type(exc).__name__}'}
