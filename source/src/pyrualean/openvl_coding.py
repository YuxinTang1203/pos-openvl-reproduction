"""Local transport for the upstream coding capability, invoked by the planner.

Only transport is adapted. Upstream builds the fresh generator prompt, supplies
binding descriptors, executes copied bindings, and renders code/output/errors.
"""
from .openvl_completion import GatewayModel


class LocalCodingClient:
    def __init__(self, adapter, base_url, api_key, model):
        self.adapter = adapter
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.calls = 0

    def execute(self, tool_name, args, *, tool_call_id):
        from openvlharness.services.envelope import _build_success_payload
        from openvlharness.services.tools.python_coding_agent.tool import Python_Coding_Agent_Tool
        a = self.adapter
        self.calls += 1
        def remaining():
            return a.remaining() if a.remaining else 150
        try:
            if remaining() <= 1 or (a.stopped and a.stopped()):
                raise TimeoutError('Episode ended')
            model = GatewayModel(self.base_url, self.api_key, self.model,
                                 a.root / 'coding' / f'{self.calls:03d}',
                                 timeout=min(120, remaining() - 1))
            generated = Python_Coding_Agent_Tool(model).execute(**args)
            # Generation consumed part of the episode: recheck before upstream execution.
            if remaining() <= 0 or (a.stopped and a.stopped()):
                raise TimeoutError('Episode ended during generation')
            cap = a.tools['Python_Coding_Agent_Tool']
            cap.service_config['execution_timeout_seconds'] = min(cap.service_config.get('execution_timeout_seconds', 30), remaining())
        except Exception as exc:
            generated = {'error': f'Local coding generation unavailable: {type(exc).__name__}'}
        return _build_success_payload(tool_name, tool_call_id, generated)
