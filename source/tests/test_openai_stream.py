import json
import time

import httpx
import pytest

from pyrualean.openai_stream import assemble, streaming_transport


def wire(items, done=True):
    lines=[]
    for item in items:
        lines.extend(['data: '+json.dumps(item), ''])
    if done:lines.extend(['data: [DONE]', ''])
    return lines


def test_interleaved_tools_reasoning_and_final_usage():
    items=[{'choices':[{'delta':{'reasoning_content':'thinking'}}]},
           {'choices':[{'delta':{'tool_calls':[
               {'index':1,'id':'b','function':{'name':'finish','arguments':'{"sta'}},
               {'index':0,'id':'a','function':{'name':'python','arguments':'{"code":"'}}]}}]},
           {'choices':[{'delta':{'tool_calls':[
               {'index':0,'function':{'arguments':'print(1)"}'}},
               {'index':1,'function':{'arguments':'tus":"stuck"}'}}]},'finish_reason':'tool_calls'}]},
           {'choices':[],'usage':{'completion_tokens':20}}]
    events=[]
    out=assemble(wire(items),lambda *a,**kw:events.append((a,kw)),time.monotonic()+5)
    calls=out['choices'][0]['message']['tool_calls']
    assert [c['id'] for c in calls]==['a','b']
    assert json.loads(calls[0]['function']['arguments'])=={'code':'print(1)'}
    assert out['usage']['completion_tokens']==20
    assert {e[1]['channel'] for e in events}=={'reasoning_content','tool_calls'}
    assert 'thinking' not in str(events)


def test_partial_stream_is_never_accepted():
    items=[{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'a','function':{'name':'python','arguments':'{"code":'}}]}}]}]
    with pytest.raises(ValueError,match='Incomplete'):
        assemble(wire(items,done=False),lambda *a,**k:None,time.monotonic()+5)
    with pytest.raises(ValueError,match='finish_reason'):
        assemble(wire(items),lambda *a,**k:None,time.monotonic()+5)


def test_deadline_and_stop():
    for deadline,stop in [(time.monotonic()-1,False),(time.monotonic()+5,True)]:
        with pytest.raises(TimeoutError):
            assemble(['data: {}'],lambda *a,**k:None,deadline,lambda:stop)


def test_http_stream_and_json_fallback(monkeypatch):
    client_type=httpx.Client
    events=[]
    def serve(request):
        body=json.loads(request.content)
        assert body['stream'] and body['stream_options']['include_usage']
        if body.get('fallback'):
            return httpx.Response(200,json={'choices':[]})
        text='\n'.join(wire([{'choices':[{'delta':{'content':'OK'},'finish_reason':'stop'}]}]))+'\n'
        return httpx.Response(200,text=text,headers={'content-type':'text/event-stream'})
    monkeypatch.setattr(httpx,'Client',lambda **kw:client_type(**kw,transport=httpx.MockTransport(serve)))
    post=streaming_transport('test',lambda *a,**kw:events.append((a,kw)))
    code,out=post('https://example.test',{},5)
    assert code==200 and out['choices'][0]['message']['content']=='OK'
    assert post('https://example.test',{'fallback':True},5)==(200,{'choices':[]})
    assert any(a==('stream_first_delta',) for a,kw in events)


def test_incomplete_reply_does_not_execute_tools(robo, tmp_path):
    from pyrualean.arms import ArmToolkit
    from pyrualean.openai_runtime import run_openai_agent
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images='none',feedback='pure')
    def broken(url,body,timeout):
        return 200,assemble(wire([{'choices':[{'delta':{'tool_calls':[
            {'index':0,'id':'a','function':{'name':'python','arguments':'{"code":"marker=1"}'}}]},
            'finish_reason':'tool_calls'}]}],done=False),lambda *a,**k:None,time.monotonic()+5)
    result=run_openai_agent('test',[],kit,model='test',base_url='https://example.test',api_key=None,
        max_turns=1,budget_s=5,out_dir=tmp_path,transport=broken,retries=0,max_requests=1)
    assert result['status']=='error' and result['tool_calls']=={}
    assert 'marker' not in kit._globals
