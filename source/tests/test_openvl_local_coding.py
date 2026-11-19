"""Native generator/capability/executor exercised through the actual host loop."""
import json
import yaml
import pytest
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.openvl_adapter import adapt_prompt
from tests.test_openvl_native import ROOT
from tests.test_openai_runtime import ScriptedTransport, _reply


@pytest.mark.parametrize('fails', [False, True])
def test_planner_receives_native_calculation_and_errors(robo, tmp_path, monkeypatch, fails):
    from pyrualean.openvl_coding import GatewayModel
    generated_requests = []
    def generate(self, messages, **kwargs):
        generated_requests.append(messages)
        assert self.model == 'qwen3.8-flash'
        assert 'ocr_result' in json.dumps(messages)
        code = "raise ValueError('calculation failed')" if fails else "print(int(ocr_result['text']) + 2); ocr_result['text'] = '99'"
        return '```python\n' + code + '\n```'
    monkeypatch.setattr(GatewayModel, 'generate_text', generate)
    config = yaml.safe_load((ROOT/'capabilities_manipulation_coding.yml').read_text())
    # OCR fixture is a public native binding; generator and executor stay real.
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'workspace', images='on-demand', feedback='pure')
    configure_migration(kit, {'kind':'openvl','config':config,'local_coding':True,'feedback_mode':'ondemand'},
                        task='Compute a measurement',initial_images={},model='qwen3.8-flash',
                        base_url='http://unused/v1',api_key='secret-fixture')
    from tests.test_openvl_native import inject
    from openvlharness.bootstrap import build_capabilities
    from tests.test_openvl_native import settings
    # Exercise registration via upstream OCR, not private binding insertion.
    a=kit.evidence_adapter
    a.tools['OCR_Tool']=build_capabilities(settings())['OCR_Tool']
    a.native_names.add('OCR_Tool')
    inject(kit,'OCR_Tool',{'text':'3'})
    import io
    from PIL import Image
    from pyrualean.openai_runtime import image_part
    buf=io.BytesIO(); Image.new('RGB',(8,8)).save(buf,format='PNG')
    a.memory.register_image(image_part(buf.getvalue())['image_url']['url'], aliases=('fixture',))
    provider=ScriptedTransport([
        _reply([('read','OCR_Tool',{'image_id':'fixture'})]),
        _reply([('calc','Python_Coding_Agent_Tool',{'objective':'Add two to the OCR number'})]),
        _reply([('end','finish',{'status':'stuck','summary':'Scripted integration smoke only'})])])
    run_openai_agent(adapt_prompt('Compute',kit),[],kit,model='qwen3.8-flash',base_url='http://unused/v1',
                     api_key=None,max_turns=5,budget_s=60,out_dir=tmp_path/'episode',transport=provider)
    assert len(generated_requests)==1
    history=json.dumps(provider.requests[-1])
    assert 'ocr_result' in history and 'openvl_environment_update' in history
    assert ('calculation failed' if fails else 'Execution output:') in history
    archived=[json.loads(line) for line in (a.root/'full_feedback.jsonl').read_text().splitlines()]
    calculation=next(row['data'] for row in archived if 'generated_code' in row.get('data', {}))
    if not fails:
        assert calculation['printed_output'].strip()=='5'
        assert not calculation['execution_error']
    assert a.memory.state.bindings.bindings['ocr_result']['text']=='3'
    assert a.provenance['coding_model']=='qwen3.8-flash'
    assert 'secret-fixture' not in (a.root/'provenance.json').read_text()
    assert not any(r.stateful for r in robo._ledger.records)


def test_local_coding_requires_explicit_configuration(robo,tmp_path):
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace')
    with pytest.raises(ValueError,match='planner gateway credentials'):
        configure_migration(kit,{'kind':'openvl','config':str(ROOT/'capabilities_manipulation_coding.yml'),'local_coding':True},
                            task='',initial_images={},model='qwen3.8-flash')


@pytest.mark.parametrize('expired_before', [True, False])
def test_deadline_does_not_execute_late_generated_code(tmp_path, monkeypatch, expired_before):
    from types import SimpleNamespace
    from pyrualean.openvl_coding import LocalCodingClient, GatewayModel
    clock = [0 if expired_before else 5]
    called = []
    def generate(self, messages, **kwargs):
        called.append(True)
        clock[0] = 0
        return '```python\nprint(123)\n```'
    monkeypatch.setattr(GatewayModel, 'generate_text', generate)
    a=SimpleNamespace(root=tmp_path,remaining=lambda:clock[0],stopped=None,tools={})
    client=LocalCodingClient(a,'http://unused','secret','qwen3.8-flash')
    result=client.execute('Python_Coding_Agent_Tool',{'objective':'Compute'},tool_call_id='test')['result']
    assert result.get('error') and not result.get('generated_code')
    assert len(called)==(0 if expired_before else 1)
