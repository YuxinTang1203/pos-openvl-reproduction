import base64,io,json
from PIL import Image
from pyrualean.image_transport import lossless_alternate,image_manifest,is_image_fetch_timeout
from tests.test_openai_runtime import _toolkit,_run,_reply,ScriptedTransport,card_images
ERROR={'error':{'message':'Download multimodal file timed out'}}

def test_text_error_and_invalid400_are_distinguished():
    assert is_image_fetch_timeout(400,json.dumps(ERROR))
    assert not is_image_fetch_timeout(400,'not-json')
    assert not is_image_fetch_timeout(400,{'error':{'message':'invalid image'}})

def test_lossless_fallback_preserves_pixels_text_tools_and_original():
    im=Image.new('RGB',(640,480),(27,144,201));b=io.BytesIO();im.save(b,format='PNG');url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
    original={'messages':[{'role':'user','content':[{'type':'text','text':'same source row,col'},{'type':'image_url','image_url':{'url':url}}]}],'tools':[{'name':'unchanged'}]}
    changed,audit=lossless_alternate(original)
    assert original['messages'][0]['content'][1]['image_url']['url']==url
    assert changed['tools']==original['tools'] and changed['messages'][0]['content'][0]==original['messages'][0]['content'][0]
    data=base64.b64decode(changed['messages'][0]['content'][1]['image_url']['url'].split(',',1)[1])
    decoded=Image.open(io.BytesIO(data));assert decoded.size==im.size and decoded.convert('RGB').tobytes()==im.tobytes()
    assert audit['same_pixels_and_dimensions'] and audit['converted_images']==1
    assert image_manifest(changed)['image_formats']=={'image/webp':1}

def test_second_timeout_changes_only_wire_images_and_never_replays_python(robo,tmp_path,card_images):
    kit=_toolkit(robo,tmp_path)
    transport=ScriptedTransport([_reply([('one','python',{'code':"executions=globals().get('executions',0)+1"})]),(400,ERROR),(400,json.dumps(ERROR)),_reply([('done','finish',{'status':'failure','summary':'fixture'})])])
    result=_run(kit,transport,tmp_path,card_images,retries=3)
    assert result['status']=='completed' and kit._globals['executions']==1
    assert transport.requests[1]['body']==transport.requests[2]['body']
    last=transport.requests[3]['body'];prior=transport.requests[2]['body']
    assert last['tools']==prior['tools'] and last['model']==prior['model']
    urls=[p['image_url']['url'] for m in last['messages'] if isinstance(m.get('content'),list) for p in m['content'] if p.get('type')=='image_url']
    assert urls and all(u.startswith('data:image/webp;base64,') for u in urls)


def test_encoding_failure_keeps_original_body_and_bounded_retry(robo,tmp_path,card_images,monkeypatch):
    from pyrualean import openai_runtime as rt
    def unavailable(body):raise RuntimeError('webp unavailable')
    monkeypatch.setattr(rt,'lossless_alternate',unavailable)
    t=ScriptedTransport([(400,ERROR),(400,ERROR),_reply([('end','finish',{'status':'failure','summary':'fixture'})])])
    r=_run(_toolkit(robo,tmp_path),t,tmp_path,card_images,retries=2)
    assert r['status']=='completed' and t.requests[0]['body']==t.requests[2]['body']
