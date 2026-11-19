"""Exercise actual native APIs and PoS delivery, with synthetic matcher only."""
import copy,json,hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from pyrualean import mast3r_auto_v36 as auto
from pyrualean.hosts import robodojo_tools as tools,robodojo_native as native
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration,validate_migration
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_robodojo_colour import Scene,Segmenter,CUBE,CAMERAS
from tests.test_openvl_routed import CONFIG

class SyntheticMatcher:
    def __init__(self,scene,empty=False):self.scene=scene;self.empty=empty;self.calls=0
    def match(self,a,b):
        self.calls+=1
        x,y=CUBE['center'];top=.765+CUBE['size'][2]
        xx,yy=np.meshgrid(np.linspace(x-.015,x+.015,20),np.linspace(y-.015,y+.015,20))
        xyz=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,top)]
        rc=[]
        for rgb in [a,b]:
            view=self.scene.view(CAMERAS[int(rgb[0,0,0])]);r,c,_=view.project(xyz);rc.append(np.c_[r,c])
        if self.empty:rc=[r[:0] for r in rc]
        return {'rc_a':rc[0],'rc_b':rc[1],'match_wall_s':.001}

@pytest.fixture
def world(tmp_path):
    scene=Scene([CUBE]);sam=Segmenter(scene);tk=native.NativeToolkit(scene,sam,tmp_path/'native',colour_only=True)
    robo=RobodojoOfficialColourRobot(native.NativeBackend(tk));matcher=SyntheticMatcher(scene)
    provider=auto.AutomaticGeometry(matcher);tk._primitives.multiview_provider=provider
    return scene,sam,tk,robo,matcher,provider

def test_existing_locate_runs_model_without_new_helper_and_returns_surface(world,monkeypatch):
    scene,sam,tk,robo,matcher,provider=world
    monkeypatch.setattr(tools,'_measure_in_colour',lambda *a,**k:pytest.fail('legacy geometry must not run'))
    sight=robo.locate('the cube')
    assert matcher.calls==2 and scene.step_count==0
    g=sight.multiview_geometry
    assert g['status']=='measured_visible_surface' and g['matcher_inference_calls']==2 and g['accepted_matches']>=6
    assert sight.center_xyz is None and sight.top_z is None and sight.grasp_yaw is None
    assert g['fallback_used'] is False and g['semantic_identity_verified'] is False
    np.testing.assert_allclose(g['geometry']['median_xyz_m'][2],.8,atol=1e-8)
    # Exact current-mask query can reuse measurements without hiding the fact.
    again=robo.locate('the cube').multiview_geometry
    assert matcher.calls==2 and again['measurement_cache_reused'] and again['matcher_inference_calls']==0

def test_selected_mask_uses_same_pipeline_without_text_redetection(world):
    scene,sam,tk,robo,matcher,provider=world
    from pycocotools import mask as mu
    obs=scene.get_obs();rgb=obs['vision']['cam_head']['color'];mask=sam.segment(rgb,text_prompt='the cube').mask
    encoded=mu.encode(np.asfortranarray(mask.astype(np.uint8)));encoded['counts']=encoded['counts'].decode()
    sam.calls.clear()
    value=robo._readonly('locate_mask',{'camera':'cam_head','mask_rle':encoded,'score':.9,'source_step':0,'source_rgb_sha256':auto.digest(rgb)})
    assert matcher.calls==2 and value['multiview_geometry']['status']=='measured_visible_surface'
    assert value['geometry_source']=='mast3r_calibrated_visible_surface'
    assert all(camera!='cam_head' for camera,_ in sam.calls),'source selected instance must not be text-redetected'
    bad=copy.deepcopy(encoded);bad['size']=[1,1]
    with pytest.raises(Exception):robo._readonly('locate_mask',{'camera':'cam_head','mask_rle':bad,'score':.9,'source_step':0,'source_rgb_sha256':auto.digest(rgb)})
    assert matcher.calls==2

def test_empty_matches_are_unknown_never_guessed_or_table_fallback(world,monkeypatch):
    scene,sam,tk,robo,matcher,provider=world;matcher.empty=True
    monkeypatch.setattr(tools,'_measure_in_colour',lambda *a,**k:pytest.fail('no fallback'))
    sight=robo.locate('the cube')
    assert matcher.calls==2 and sight.center_xyz is None and sight.multiview_geometry['status']=='unknown'
    assert sight.multiview_geometry['fallback_used'] is False
    rgb=scene.get_obs()['vision']['cam_head']['color'];mask=sam.segment(rgb,text_prompt='the cube').mask
    r,c=np.argwhere(mask)[len(np.argwhere(mask))//2]
    with pytest.raises(Exception):robo.back_project(int(r),int(c))
    assert scene.step_count==0

def test_back_project_returns_real_nearby_match_not_assumed_plane(world):
    scene,sam,tk,robo,matcher,provider=world
    r,c,_=scene.view('cam_head').project(np.array([[*CUBE['center'],.8]]))
    point=robo.back_project(int(round(r[0])),int(round(c[0])))
    assert matcher.calls==2 and point.surface=='matched_visible_surface'
    assert abs(point.xyz[2]-.8)<1e-8 and point.multiview_geometry['fallback_used'] is False

def test_motion_during_matching_rejects_stale_cloud(world):
    scene,sam,tk,robo,matcher,provider=world
    original=matcher.match
    def moves(a,b):
        result=original(a,b);scene.step_count+=1;return result
    matcher.match=moves
    sight=robo.locate('the cube');g=sight.multiview_geometry
    assert g['status']=='unknown' and g['reason']=='observation_changed_during_perception' and 'geometry' not in g
    assert not provider.cache

def test_matcher_failure_remains_visible_and_no_cpu_fallback(world):
    scene,sam,tk,robo,matcher,provider=world
    def fails(*args):raise RuntimeError('GPU ownership refused')
    matcher.match=fails
    g=robo.locate('the cube').multiview_geometry
    assert g['status']=='unknown' and g['matcher_inference_calls']==0
    assert len(g['pairs'])==2 and all('GPU ownership refused' in p['reason'] for p in g['pairs'])

def test_legacy_mode_is_unchanged_when_provider_absent(world):
    scene,sam,tk,robo,matcher,provider=world;del tk._primitives.multiview_provider
    s=robo.locate('the cube')
    assert s.found and matcher.calls==0 and s.multiview_geometry is None

def test_real_migration_python_and_pos_receive_unprinted_automatic_measurement(tmp_path,monkeypatch):
    scene=Scene([CUBE]);sam=Segmenter(scene);tk=native.NativeToolkit(scene,sam,tmp_path/'native',colour_only=True)
    robo=RobodojoOfficialColourRobot(native.NativeBackend(tk));kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images='on-demand',feedback='pure')
    import pyrualean.openvl_routed as routed
    original=routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed,'OpenVLRoutedAdapter',lambda *a,**k:original(*a,**k,segmenter=sam))
    def segment_all(image,**kw):
        from pyrualean.hosts.sam3_local import Segmented
        r=sam.segment(image,**kw);rows,cols=np.nonzero(r.mask)
        return [Segmented(True,r.score,[float(cols.min()),float(rows.min()),float(cols.max()+1),float(rows.max()+1)],r.mask)]
    sam.segment_all=segment_all
    matcher=SyntheticMatcher(scene);monkeypatch.setattr(auto,'Mast3rMatcher',lambda *a,**k:matcher)
    cfg={'kind':auto.KIND,'condition':'M','pos':{},'openvl':{'enabled':True,'config':str(CONFIG)},'mast3r':{'repo':'/unused','checkpoint':'/unused','device':'cuda:0'}}
    validate_migration(cfg);configure_migration(kit,cfg,task='Insert tubes',initial_images={})
    out=kit.execute_tool('python',{'code':"unprinted_result=robo.locate('the cube')"})
    assert not out.result.get('exception')
    assert matcher.calls==2 and out.result['automatic_multiview_geometry'][0]['matcher_inference_calls']==2
    h=kit.general_harness;entry=h.evidence['obs-'+str(h.sequence)]
    assert 'automatic_multiview_geometry' in entry['text'] and 'measured_visible_surface' in entry['text']
    assert h.delivered_visual_packets[-1]['automatic_geometry'][0]['status']=='measured_visible_surface'
    assert scene.step_count==0

    assert h.delivered_visual_packets[-1]['automatic_geometry'][0]['geometry']['median_xyz_m'][2]==pytest.approx(.8)
    binding=next(iter(kit.evidence_adapter.groundings))
    selected=kit.execute_tool('python',{'code':f"selected_result=robo.locate_candidate({binding!r},1)"})
    assert not selected.result.get('exception')
    assert kit._globals['selected_result'].multiview_geometry['status']=='measured_visible_surface'
    assert selected.result['automatic_multiview_geometry'][0]['current_eligible']
    # Currentness is rechecked when the same Python cell performs a later action.
    stale=kit.execute_tool('python',{'code':"measurement=robo.locate('the cube')\nrobo.set_gripper('right',robo.CLOSE)"})
    assert not stale.result.get('exception')
    g=stale.result['automatic_multiview_geometry'][0]
    assert not g['current_eligible'] and g['delivery_status']=='historical_or_unverifiable_at_delivery'

def test_competing_target_instances_are_rejected_not_bound_by_top_score():
    from pyrualean.hosts.sam3_local import Segmented
    a=np.zeros((30,30),bool);a[5,0:12]=True
    b=np.zeros_like(a);b[5,8:16]=True
    def segment(rgb,point,min_score):return Segmented(True,.99,None,a if point[1]<12 else b)
    primitives=SimpleNamespace(sam3_client=SimpleNamespace(segment=segment))
    rc=np.c_[np.full(16,5),np.arange(16)]
    provider=auto.AutomaticGeometry(None)
    mask,reason=provider._target_mask(primitives,np.zeros((30,30,3),np.uint8),rc,np.ones(16,bool))
    assert mask is None and reason=='ambiguous_target_instance'
