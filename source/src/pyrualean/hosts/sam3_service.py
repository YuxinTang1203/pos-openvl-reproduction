"""Loopback SAM3 service: lossless RGB/masks, serialized inference, no episode state."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, build_opener, ProxyHandler
import numpy as np
from .sam3_local import Segmented


def pixels_payload(image):
    pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError('image must be HxWx3 RGB')
    raw = pixels.tobytes()
    return pixels, dict(shape=list(pixels.shape), rgb=base64.b64encode(raw).decode(),
                       image_sha256=hashlib.sha256(raw).hexdigest())


class RemoteSam3:
    def __init__(self, url, timeout_s=300):
        self.url = url.rstrip('/')
        self.timeout_s = timeout_s
        self.opener = build_opener(ProxyHandler({}))

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        return self._request('/segment', image, text_prompt=text_prompt,
                             point=point, min_score=min_score)

    def segment_all(self, image, *, text_prompt, min_score=0.2):
        return self._request('/segment_all', image, text_prompt=text_prompt,
                             min_score=min_score)

    def _request(self, endpoint, image, **options):
        _, payload = pixels_payload(image)
        payload.update(request_id=uuid.uuid4().hex, **options)
        payload['min_score'] = float(payload['min_score'])
        request = Request(self.url+endpoint, data=json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
        with self.opener.open(request, timeout=self.timeout_s) as response:
            data = json.load(response)
        if data.get('request_id') != payload['request_id'] or data.get('image_sha256') != payload['image_sha256']:
            raise RuntimeError('SAM3 response/request identity mismatch')
        if data.get('error'):
            raise RuntimeError(data['error'])
        if endpoint == '/segment_all':
            return [self._decode(value, payload) for value in data['candidates']]
        return self._decode(data, payload)

    @staticmethod
    def _decode(data, payload):
        mask = None
        if data.get('mask') is not None:
            shape = tuple(data['mask_shape'])
            if shape != tuple(payload['shape'][:2]):
                raise RuntimeError('SAM3 mask shape mismatch')
            packed = np.frombuffer(base64.b64decode(data['mask'], validate=True), dtype=np.uint8)
            mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
        return Segmented(data['found'], data.get('score'), data.get('box'), mask, data.get('reason'))


def encode_segment(value):
    mask = None if value.mask is None else np.asarray(value.mask, dtype=bool).copy()
    return dict(found=bool(value.found), score=value.score, box=value.box, reason=value.reason,
                mask=None if mask is None else base64.b64encode(np.packbits(mask).tobytes()).decode(),
                mask_shape=None if mask is None else list(mask.shape))


def make_server(model, host='127.0.0.1', port=0, audit=None):
    inference_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path=='/health' else 404, {'ready':self.path=='/health'})

        def do_POST(self):
            if self.path not in {'/segment', '/segment_all'}:
                self.reply(404, {'error':'unknown endpoint'}); return
            data = {}
            start = time.monotonic()
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 32*1024*1024:
                    raise ValueError('invalid payload size')
                data = json.loads(self.rfile.read(length))
                shape = data['shape']
                if len(shape)!=3 or shape[2]!=3 or any(type(n)!=int or n<=0 for n in shape):
                    raise ValueError('invalid RGB shape')
                raw = base64.b64decode(data['rgb'], validate=True)
                if len(raw)!=int(np.prod(shape)) or hashlib.sha256(raw).hexdigest()!=data['image_sha256']:
                    raise ValueError('invalid RGB payload')
                pixels = np.frombuffer(raw,dtype=np.uint8).reshape(shape).copy()
                # Own the whole request: neither image features nor prompts can
                # be changed by another client while inference is in progress.
                with inference_lock:
                    entered = time.monotonic()
                    response = dict(request_id=data['request_id'], image_sha256=data['image_sha256'],
                                    endpoint=self.path)
                    if self.path == '/segment_all':
                        values = model.segment_all(pixels, text_prompt=data.get('text_prompt'),
                                                   min_score=data.get('min_score', 0.2))
                        response['candidates'] = [encode_segment(value) for value in values]
                    else:
                        value = model.segment(pixels, text_prompt=data.get('text_prompt'),
                                              point=data.get('point'), min_score=data.get('min_score',0.2))
                        response.update(encode_segment(value))
                    if audit:
                        with open(audit,'a') as f:
                            f.write(json.dumps(dict(request_id=data['request_id'], image_sha256=data['image_sha256'],
                                                    endpoint=self.path,
                                                    queue_s=entered-start,inference_s=time.monotonic()-entered))+'\n')
                self.reply(200,response)
            except Exception as exc:
                self.reply(400,dict(request_id=data.get('request_id'),image_sha256=data.get('image_sha256'),
                                    error=f'{type(exc).__name__}: {exc}'))
    server = ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads = True
    return server


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--audit')
    args=parser.parse_args()
    from .sam3_local import LocalSam3
    model=LocalSam3(cuda_device=args.gpu)
    server=make_server(model,port=args.port,audit=args.audit)
    print(f'SAM3 ready on 127.0.0.1:{server.server_port}',flush=True)
    server.serve_forever()

if __name__=='__main__':
    main()
