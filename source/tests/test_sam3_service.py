import concurrent.futures
import threading
import time
import numpy as np
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.hosts.sam3_service import make_server, RemoteSam3


def test_clients_keep_images_points_and_masks_isolated():
    class StatefulModel:
        def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
            self.image=image
            time.sleep(0.005)
            assert np.array_equal(image,self.image)
            mask = self.image[...,0] == (255 if text_prompt else 0)
            if point is not None:
                assert point == [2,3]
            return Segmented(True,0.9,[0.,0.,7.,5.],mask)
    server=make_server(StatefulModel())
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        clients=[RemoteSam3(f'http://127.0.0.1:{server.server_port}') for _ in range(2)]
        images=[np.zeros((5,7,3),dtype=np.uint8),np.full((5,7,3),255,dtype=np.uint8)]
        def run(i):
            result=clients[i%2].segment(images[i%2],**({'text_prompt':'object'} if i%2 else {'point':[2,3]}))
            assert result.found and result.mask.shape==(5,7) and result.mask.all()
            assert result.mask.dtype==bool
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run,range(24)))
    finally:
        server.shutdown();server.server_close();thread.join()


def test_remote_selection_does_not_load_local_weights(monkeypatch):
    from pyrualean.hosts import sam3_local
    monkeypatch.setenv('PYRUALEAN_SAM3_URL','http://127.0.0.1:1234')
    monkeypatch.setattr(sam3_local,'LocalSam3',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('must not load')))
    assert isinstance(sam3_local.shared(),RemoteSam3)


def test_all_candidates_preserve_masks_order_scores_and_empty_results():
    masks = [np.eye(5, 7, dtype=bool), np.ones((5, 7), dtype=bool)]
    class Model:
        def segment_all(self, image, *, text_prompt, min_score):
            assert min_score == 0.35
            if text_prompt == 'missing':
                return []
            assert text_prompt == 'tubes' and image[0, 0, 0] == 123
            return [Segmented(True, score, [1., 0., 7., 5.], mask)
                    for score, mask in zip([.6, .9], masks)]
    server = make_server(Model())
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        client = RemoteSam3(f'http://127.0.0.1:{server.server_port}')
        image = np.full((5, 7, 3), 123, dtype=np.uint8)
        values = client.segment_all(image, text_prompt='tubes', min_score=.35)
        assert [value.score for value in values] == [.6, .9]
        assert all(value.box == [1., 0., 7., 5.] for value in values)
        assert all(np.array_equal(value.mask, mask) for value, mask in zip(values, masks))
        assert client.segment_all(image, text_prompt='missing', min_score=.35) == []
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_segment_and_segment_all_serialize_shared_image_features():
    class Model:
        def segment(self, image, **options):
            self.image = image
            time.sleep(.005)
            assert np.array_equal(image, self.image)
            return Segmented(True, .9, [0., 0., 7., 5.], image[..., 0] > 0)
        def segment_all(self, image, **options):
            return [self.segment(image, **options)]
    server = make_server(Model())
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        client = RemoteSam3(f'http://127.0.0.1:{server.server_port}')
        def run(index):
            image = np.full((5, 7, 3), 255 * (index % 2), dtype=np.uint8)
            value = (client.segment_all(image, text_prompt='tubes')[0] if index % 3
                     else client.segment(image, text_prompt='tubes'))
            assert np.array_equal(value.mask, image[..., 0] > 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, range(24)))
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_candidate_decoder_rejects_wrong_image_shape():
    import base64
    import pytest
    with pytest.raises(RuntimeError, match='mask shape mismatch'):
        RemoteSam3._decode(dict(found=True, mask=base64.b64encode(b'\xff').decode(),
                                mask_shape=[1, 8]), dict(shape=[5, 7, 3]))
