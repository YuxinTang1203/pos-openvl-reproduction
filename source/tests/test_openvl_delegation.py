import json
import pytest
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration
from tests.test_openvl_native import ROOT


def setup(robo,tmp_path):
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images='on-demand',feedback='pure')
    configure_migration(kit,{'kind':'openvl','config':str(ROOT/'capabilities_manipulation_coding.yml'),'local_coding':True,'feedback_mode':'ondemand'},task='Analysis',initial_images={},model='qwen3.8-flash',base_url='http://unused',api_key='secret')
    return kit


def test_selected_records_only_and_report_returned(robo,tmp_path,monkeypatch):
    from pyrualean.openvl_coding import GatewayModel
    kit=setup(robo,tmp_path);a=kit.evidence_adapter
    wanted=a.trace.store('measurement',value=3)
    other=a.trace.store('private_unselected',value='DO_NOT_PASS_THIS_RECORD')
    def generate(self,messages,**kwargs):
        text=json.dumps(messages)
        assert wanted in text and other not in text and 'DO_NOT_PASS_THIS_RECORD' not in text
        return '```python\nimport json\nassert '+repr(other)+' not in globals()\nv='+wanted+"['value'] * 2\nprint(json.dumps({'answer':str(v),'calculations':[v],'evidence_refs':["+repr(wanted)+"],'limitations':['Historical measurement only']}))\n"+wanted+"['value']=100\n```"
    monkeypatch.setattr(GatewayModel,'generate_text',generate)
    result=kit.execute_tool('Python_Coding_Agent_Tool',{'objective':'Double the recorded value','evidence_refs':[wanted]}).result
    assert result['delegated_analysis']['report']['calculations']==[6]
    assert result['delegated_analysis']['conclusion_verified'] is False
    assert result['delegation']['retrieval'][0]['temporal_context']['current_target'] is False
    assert a.memory.state.bindings.bindings[wanted]['value']==3
    assert other in a.memory.state.bindings.bindings


@pytest.mark.parametrize('refs', [[],['missing'],'bad',[1]])
def test_invalid_refs_never_call_generator(robo,tmp_path,monkeypatch,refs):
    from pyrualean.openvl_coding import GatewayModel
    monkeypatch.setattr(GatewayModel,'generate_text',lambda *a,**k:pytest.fail('Unexpected model call'))
    kit=setup(robo,tmp_path)
    result=kit.execute_tool('Python_Coding_Agent_Tool',{'objective':'Analyze','evidence_refs':refs}).result
    assert result['error']['code']=='invalid_delegation'


def test_invalid_citations_and_execution_errors_are_not_reports():
    from pyrualean.openvl_delegation import assessment
    data={'printed_output':json.dumps({'answer':'x','calculations':[1],'limitations':[],'evidence_refs':['invented']})}
    assert assessment(data,{'references':['real']})['status']=='unavailable'
    data['execution_error']='boom'
    assert assessment(data,{'references':['invented']})['status']=='unavailable'


def test_pretty_printed_structured_answer_with_computation_log():
    from pyrualean.openvl_delegation import assessment
    report={'answer':{'description':'Computed distance'},'calculations':[0.5],
            'evidence_refs':['record_a'],'limitations':['Historical']}
    data={'printed_output':'Loaded one record\n'+json.dumps(report,indent=2)}
    assert assessment(data,{'references':['record_a']})['report']==report
    data['printed_output']+='\nnot a final report'
    assert assessment(data,{'references':['record_a']})['status']=='unavailable'
