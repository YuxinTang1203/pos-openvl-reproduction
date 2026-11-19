import json
from types import SimpleNamespace
import pytest
from pyrualean.openvl_completion import parse_review, review_bundle
from tests.test_openvl_native import setup

class Model:
    def generate_text(self, messages, **kwargs):
        assert len(messages)==2
        return '''```python
import json
ok = public_evidence['measured'] == public_evidence['required']
s = 'satisfied' if ok else 'not_satisfied'
print(json.dumps({'verdict':s,'checks':[{'requirement':'measured matches required','status':s,'evidence':public_evidence}],'explanation':'Computed equality'}))
```'''

@pytest.mark.parametrize('measured,verdict', [(1,'satisfied'),(2,'not_satisfied')])
def test_actual_native_generation_and_execution(tmp_path,measured,verdict):
    result=review_bundle({'measured':measured,'required':1},Model(),tmp_path)
    assert result['verdict']==verdict
    assert (tmp_path/'execution.json').exists()

def test_inconsistent_or_empty_output_never_passes():
    for text in ['garbage','{"verdict":"satisfied","checks":[]}',json.dumps({'verdict':'satisfied','checks':[{'requirement':'x','status':'unknown','evidence':'missing'}]})]:
        assert parse_review(text)['verdict']=='unknown'

@pytest.mark.parametrize('mode,verdict,finished',[('gate','not_satisfied',False),('gate','unknown',False),('gate','satisfied',True),('observe','not_satisfied',True)])
def test_finish_review_modes(robo,tmp_path,mode,verdict,finished):
    kit,_=setup(robo,tmp_path)
    kit.evidence_adapter.completion_reviewer=SimpleNamespace(mode=mode,review=lambda:{'verdict':verdict,'checks':[]})
    out=kit.execute_tool('finish',{'status':'success','summary':'claim'})
    assert bool(kit.finish)==finished
    assert out.result['completion_review']['verdict']==verdict

def test_stuck_does_not_call_reviewer_and_exhaustion_is_not_success(robo,tmp_path):
    kit,_=setup(robo,tmp_path)
    def fail():raise AssertionError('unexpected review')
    kit.evidence_adapter.completion_reviewer=SimpleNamespace(mode='gate',review=fail)
    kit.execute_tool('finish',{'status':'stuck','summary':'cannot continue'})
    assert kit.finish['status']=='stuck'
    kit.finish=None
    kit.evidence_adapter.completion_reviewer.review=lambda:{'verdict':'unknown','exhausted':True}
    kit.execute_tool('finish',{'status':'success','summary':'claim'})
    assert kit.finish['status']=='stuck'

def test_generator_and_executor_errors_are_unknown(tmp_path):
    class Broken:
        def generate_text(self,*a,**k):raise RuntimeError('provider error')
    assert review_bundle({},Broken(),tmp_path/'generation')['verdict']=='unknown'
    class BadCode:
        def generate_text(self,*a,**k):return 'raise ValueError("bad check")'
    assert review_bundle({},BadCode(),tmp_path/'execution')['verdict']=='unknown'

def test_timeout_is_unknown(tmp_path):
    class Loop:
        def generate_text(self,*a,**k):return 'while True: pass'
    assert review_bundle({},Loop(),tmp_path,execution_timeout=.2)['verdict']=='unknown'


def test_online_collection_excludes_claims_and_oracle(robo,tmp_path,monkeypatch):
    from pyrualean.openvl_completion import CompletionReviewer
    import pyrualean.openvl_completion as module
    kit,_=setup(robo,tmp_path)
    a=kit.evidence_adapter
    reviewer=CompletionReviewer(a,'Original instruction',{},'https://example.invalid/v1','secret','qwen3.8-flash','gate')
    captured=[]
    def check(bundle, model, directory, **kwargs):
        captured.append(bundle)
        assert bundle['task']=='Original instruction'
        assert 'environment_success' not in json.dumps(bundle)
        assert 'policy claims success' not in json.dumps(bundle)
        assert model.images
        return {'verdict':'unknown','checks':[]}
    monkeypatch.setattr(module,'review_bundle',check)
    a.completion_reviewer=reviewer
    kit.execute_tool('finish',{'status':'success','summary':'policy claims success'})
    kit.execute_tool('finish',{'status':'success','summary':'policy claims success'})
    kit.execute_tool('finish',{'status':'success','summary':'policy claims success'})
    assert len(captured)==2 and kit.finish['status']=='stuck'


def test_zero_budget_never_calls_generator(robo,tmp_path,monkeypatch):
    from pyrualean.openvl_completion import CompletionReviewer
    kit,_=setup(robo,tmp_path);a=kit.evidence_adapter;a.remaining=lambda:0
    r=CompletionReviewer(a,'task',{},'http://invalid','x','qwen3.8-flash','gate')
    assert r.review()['exhausted']
    assert r.count==0


def test_gateway_uses_fresh_native_messages_and_visual_inputs(tmp_path, monkeypatch):
    import io
    from PIL import Image
    from pyrualean.openvl_completion import GatewayModel
    import pyrualean.openai_stream as stream
    received=[]
    def transport(key,event):
        def post(url,body,timeout):
            received.append(body)
            return 200, {'choices':[{'finish_reason':'stop','message':{'content':'print(1)'}}]}
        return post
    monkeypatch.setattr(stream,'streaming_transport',transport)
    image=io.BytesIO();Image.new('RGB',(3,3)).save(image,format='PNG')
    model=GatewayModel('http://example.invalid/v1','not-in-files','qwen3.8-flash',tmp_path,[('initial',image.getvalue())])
    model.generate_text([{'role':'system','content':'code'},{'role':'user','content':'objective'}])
    assert len(received[0]['messages'])==2
    assert received[0]['messages'][1]['content'][2]['type']=='image_url'
    assert received[0]['chat_template_kwargs']['enable_thinking'] is False
    assert 'not-in-files' not in (tmp_path/'request.json').read_text()


def test_configured_gate_uses_planner_model_and_keeps_secret_out_of_provenance(robo,tmp_path):
    from pyrualean.arms import ArmToolkit
    from pyrualean.migrations import configure_migration
    from tests.test_openvl_native import ROOT
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images='on-demand',feedback='pure')
    record=configure_migration(kit,{'kind':'openvl','config':str(ROOT/'capabilities_local.yml'),'completion_review':'gate'},
        task='original task',initial_images={},base_url='http://example.invalid/v1',api_key='test-secret',model='qwen3.8-flash')
    reviewer=kit.evidence_adapter.completion_reviewer
    assert reviewer.mode=='gate' and reviewer.model=='qwen3.8-flash'
    assert record['native']['completion_review']=='gate'
    assert 'test-secret' not in (kit.evidence_adapter.root/'provenance.json').read_text()
    assert 'independent native Coding Agent' in kit.evidence_adapter.prompt()
