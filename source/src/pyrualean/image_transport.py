"""Lossless alternate wire encoding for a provider-side PNG fetch failure."""
import base64,copy,hashlib,io,json
from PIL import Image

def is_image_fetch_timeout(code,payload):
    if code!=400:return False
    if isinstance(payload,(str,bytes)):
        try:payload=json.loads(payload)
        except (ValueError,TypeError):return False
    error=payload.get('error') if isinstance(payload,dict) else None
    return isinstance(error,dict) and isinstance(error.get('message'),str) and 'download multimodal file timed out' in error['message'].lower()

def image_manifest(body):
    count=0;total=0;formats={}
    for message in body.get('messages',[]):
        content=message.get('content')
        if not isinstance(content,list):continue
        for part in content:
            if part.get('type')!='image_url':continue
            count+=1;url=part.get('image_url',{}).get('url','')
            if url.startswith('data:') and ';base64,' in url:
                head,data=url.split(';base64,',1);mime=head[5:];formats[mime]=formats.get(mime,0)+1;total+=len(data)*3//4-len(data)+len(data.rstrip('='))
            else:formats['external_or_other']=formats.get('external_or_other',0)+1
    return {'image_count':count,'image_payload_bytes':total,'image_formats':formats}

def lossless_alternate(body):
    changed=copy.deepcopy(body);converted=0;saved=0;hashes=[];cache={}
    for message in changed.get('messages',[]):
        content=message.get('content')
        if not isinstance(content,list):continue
        for part in content:
            if part.get('type')!='image_url':continue
            item=part.get('image_url',{});url=item.get('url','')
            if not url.startswith('data:image/png;base64,'):continue
            if url not in cache:
                raw=base64.b64decode(url.split(',',1)[1],validate=True)
                im=Image.open(io.BytesIO(raw));im.load()
                if im.mode not in ('RGB','RGBA'):cache[url]=None;continue
                b=io.BytesIO();im.save(b,format='WEBP',lossless=True,exact=True,method=6);encoded=b.getvalue()
                decoded=Image.open(io.BytesIO(encoded)).convert(im.mode)
                if decoded.size!=im.size or decoded.tobytes()!=im.tobytes():raise ValueError('Lossless transport RGB validation failed')
                cache[url]=('data:image/webp;base64,'+base64.b64encode(encoded).decode(),len(raw)-len(encoded),hashlib.sha256(im.tobytes()).hexdigest())
            result=cache[url]
            if result:
                item['url']=result[0];converted+=1;saved+=result[1];hashes.append(result[2])
    return changed,{'converted_images':converted,'bytes_saved':saved,'decoded_pixel_hashes':hashes,'same_pixels_and_dimensions':True,'robot_replay':False}
