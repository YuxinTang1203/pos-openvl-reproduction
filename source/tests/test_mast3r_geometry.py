import numpy as np
import pytest
from pyrualean.mast3r_geometry import triangulate, summarize_cloud, Mast3rMatcher
from pyrualean.openvl_geometry import calibrated_view


def meta(x=0., shape=(480,640), focal=500.):
    e=np.eye(4);e[:3,3]=[x,0,1]
    return {'shape':list(shape),'intrinsic_matrix':[[focal,0,shape[1]/2],[0,focal,shape[0]/2],[0,0,1]],'extrinsic_matrix':e.tolist()}


def pixels(m,points):
    r,c,_=calibrated_view(m).project(points)
    return np.c_[r,c]


def test_known_calibrated_metric_geometry():
    a,b=meta(-.12),meta(.12)
    rng=np.random.default_rng(2);p=rng.uniform([-.1,-.1,.2],[.1,.1,.5],(80,3))
    out=triangulate(a,b,pixels(a,p),pixels(b,p))
    assert out['accepted'].all()
    np.testing.assert_allclose(out['xyz_m'],p,atol=1e-10)
    assert out['reprojection_px'].max()<1e-9


def test_wrong_correspondence_rejected_by_epipolar_geometry():
    a,b=meta(-.12),meta(.12);p=np.array([[0,0,.3]])
    pb=pixels(b,p);pb[:,0]+=12
    out=triangulate(a,b,pixels(a,p),pb)
    assert not out['accepted'].any()


def test_small_parallax_and_duplicate_camera_not_measurements():
    a,b=meta(0),meta(.0001);p=np.array([[0,0,.3]])
    assert not triangulate(a,b,pixels(a,p),pixels(b,p))['accepted'].any()
    with pytest.raises(ValueError,match='baseline'):triangulate(a,a,pixels(a,p),pixels(a,p))


def test_behind_camera_rejected():
    a,b=meta(-.1),meta(.1);p=np.array([[0,0,2.]])
    out=triangulate(a,b,pixels(a,p),pixels(b,p))
    assert not out['accepted'].any()


@pytest.mark.parametrize('bad', [np.array([[np.nan,2.]]),np.array([[-1,2.]]),np.array([[481.,2.]])])
def test_invalid_original_pixels_rejected(bad):
    with pytest.raises(ValueError):triangulate(meta(-.1),meta(.1),bad,np.array([[200.,300.]]))


def test_explicit_plane_fit_is_not_hidden_height_prior():
    rng=np.random.default_rng(3)
    p=np.c_[rng.uniform(-.1,.1,(100,2)),np.full(100,.823)]
    out=summarize_cloud(p)
    assert out['plane'] is not None
    assert abs(out['plane']['origin_m'][2]-.823)<1e-10
    assert abs(out['plane']['normal'][2])>.999
    assert 'not a measured physical tip' in out['interpretation']


def test_collinear_points_do_not_claim_a_surface_plane():
    p=np.c_[np.linspace(0,.1,100),np.zeros((100,2))]
    assert summarize_cloud(p)['plane'] is None
    assert summarize_cloud(p)['axis_dominance']>10


def test_no_accidental_gpu_use_or_eager_vendor_import(tmp_path,monkeypatch):
    m=Mast3rMatcher(tmp_path,tmp_path/'weights',device='cpu')
    assert m.model is None
    with pytest.raises(RuntimeError,match='CPU model inference disabled'):m.load()
    monkeypatch.delenv('PYRUA_ADMITTED_GPU_UUID',raising=False)
    g=Mast3rMatcher(tmp_path,tmp_path/'weights',device='cuda:0')
    with pytest.raises(RuntimeError,match='admitted physical UUID'):g.load()


def test_pil_resize_coordinates_preserve_pixel_centres(tmp_path):
    m=Mast3rMatcher(tmp_path,tmp_path/'weights')
    image=np.zeros((480,640,3),np.uint8)
    p,s=m._prepare(image,0)
    assert tuple(p['img'].shape)==(1,3,384,512)
    rc=np.array([[40.,70.],[211.,305.]])
    reduced=(rc+.5)*s-.5
    np.testing.assert_allclose((reduced+.5)/s-.5,rc)
