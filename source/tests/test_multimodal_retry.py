from tests.test_openai_runtime import ScriptedTransport,_toolkit,_run,_reply,card_images
from pyrualean.openai_runtime import _multimodal_download_timeout

TIMEOUT={'error':{'code':'invalid_parameter_error','message':'Download multimodal file timed out'}}

def test_only_specific_provider_image_timeout_is_retryable():
    assert _multimodal_download_timeout(400,TIMEOUT)
    assert not _multimodal_download_timeout(400,{'error':{'message':'Invalid image format'}})
    assert not _multimodal_download_timeout(401,TIMEOUT)
    assert not _multimodal_download_timeout(400,'timeout')

def test_image_timeout_retries_same_request_without_replaying_python(robo,tmp_path,card_images):
    kit=_toolkit(robo,tmp_path)
    t=ScriptedTransport([_reply([('one','python',{'code':"executions=globals().get('executions',0)+1"})]),
        (400,TIMEOUT),_reply([('done','finish',{'status':'failure','summary':'fixture'})])])
    result=_run(kit,t,tmp_path,card_images,retries=2)
    assert result['status']=='completed' and not result['errors']
    assert kit._globals['executions']==1
    assert t.requests[1]['body']==t.requests[2]['body']

def test_image_timeout_is_still_bounded(robo,tmp_path,card_images):
    t=ScriptedTransport([(400,TIMEOUT)]*3)
    result=_run(_toolkit(robo,tmp_path),t,tmp_path,card_images,retries=2)
    assert result['status']=='error' and result['requests']==3
