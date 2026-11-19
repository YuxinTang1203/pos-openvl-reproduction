from types import SimpleNamespace
import copy
import numpy as np
import pytest
from pyrualean.mast3r_adapter_v35 import GeometryBridge,PROMPT
from pyrualean.openvl_geometry import calibrated_view
from pyrualean.arms import ArmToolkit
from pyrualean.robodojo import RobodojoRobot
from pyrualean.migrations import configure_migration
from tests.test_robodojo import FakeRobodojoBackend
from tests.test_openvl_routed import Candidates,CONFIG


def metadata(x):
    e=np.eye(4);e[:3,3]=[x,0,1.3]
    return {'shape':[480,640],'intrinsic_matrix':[[500,0,320],[0,500,240],[0,0,1]],'extrinsic_matrix':e.tolist()}


def test_calibrated_bridge_and_freshness_without_model_or_robot_motion():
    a,b=metadata(-.12),metadata(.12);rng=np.random.default_rng(0)
    xyz=np.c_[rng.uniform(-.08,.08,(120,2)),np.full(120,.823)]
    def pix(m):
        r,c,_=calibrated_view(m).project(xyz);return np.c_[r,c]
    class Matcher:
        calls=0
        def match(self,*args):
            self.calls+=1
            return {'rc_a':pix(a),'rc_b':pix(b),'match_wall_s':0}
    state={'revision':0};events=[]
    visual=SimpleNamespace(action_revision=lambda:state['revision'],
        _current_candidate=lambda binding,n:({'camera':'head' if binding=='binding' else 'wrist','image_id':binding}, {},np.ones((480,640),bool),np.zeros((480,640,3),np.uint8)),
        _original_camera_meta=lambda name:a if name=='head' else b,
        register_frame=lambda *args,**kw:'other',frame_context=lambda *args:{'current_camera_pixels':True},
        event=lambda *args,**kwargs:events.append((args,kwargs)))
    robo=SimpleNamespace(CAMERAS=['head','wrist'],_backend=SimpleNamespace(env_steps=lambda:0),image=lambda name:np.zeros((480,640,3),np.uint8))
    matcher=Matcher();g=GeometryBridge(SimpleNamespace(robo=robo,evidence_adapter=visual),matcher)
    result=g.measure('binding',1,'target',1)
    assert result['accepted_matches']==120 and matcher.calls==1
    assert result['semantic_identity_verified'] is False and result['contact_verified'] is False
    p=g.point(result['geometry_id'],*pix(a)[17]);np.testing.assert_allclose(p['xyz_m'],xyz[17],atol=1e-8)
    q=g.plane_point(result['geometry_id'],240,320)
    assert q['status']=='conditional_on_fitted_plane' and abs(q['xyz_m'][2]-.823)<1e-8
    assert len(g.pending)==1 and len(events)==1
    state['revision']=1
    with pytest.raises(ValueError,match='historical'):g.point(result['geometry_id'],240,320)
    with pytest.raises(ValueError,match='historical'):g.plane_point(result['geometry_id'],240,320)
    assert matcher.calls==1
    # A target mask excluding all supported pixels must not return a plane/point.
    state['revision']=0
    original=visual._current_candidate
    def different_target(binding,n):
        src,data,mask,pixels=original(binding,n)
        if binding=='target':mask=np.zeros_like(mask)
        return src,data,mask,pixels
    visual._current_candidate=different_target
    result=g.measure('binding',1,'target',1)
    assert result['accepted_matches']==0
    assert g.point(result['geometry_id'],240,320)['status']=='unknown'
    assert g.plane_point(result['geometry_id'],240,320)['status']=='unknown'


def test_actual_original_harness_control_unchanged(tmp_path,monkeypatch):
    import pyrualean.openvl_routed as routed
    original=routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed,'OpenVLRoutedAdapter',lambda *a,**kw:original(*a,**kw,segmenter=Candidates()))
    def make(condition=None):
        k=ArmToolkit(RobodojoRobot(FakeRobodojoBackend()),arm='cells',workspace=tmp_path/str(condition),images='on-demand',feedback='pure')
        cfg={'kind':'v35_mast3r','condition':condition,'pos':{},'openvl':{'enabled':True,'config':str(CONFIG)}}
        if condition is None:
            cfg.pop('condition');cfg.update(kind='v17_ablation',profile='fusion',static_rules=False,action_focus=False,reliability=False)
        if condition=='M':cfg['mast3r']={'repo':'/unused/vendor','checkpoint':'/unused/checkpoint','device':'cuda:0'}
        configure_migration(k,cfg,task='Insert tubes',initial_images={});return k
    a,m,d=make('A'),make('M'),make()
    assert a.general_harness.prompt()==d.general_harness.prompt()
    assert m.general_harness.prompt()==a.general_harness.prompt()+PROMPT
    assert a.get_tools_spec()==m.get_tools_spec()==d.get_tools_spec()
    assert m.general_harness.mast3r_bridge.matcher.model is None
    assert hasattr(m.robo,'measure_geometry') and not hasattr(a.robo,'measure_geometry')
    # Normal Python still executes on its first call, with no V34 draft stage.
    m.execute_tool('python',{'code':'execution_counter=1'})
    assert m._globals['execution_counter']==1
    assert m.robo._action_feedback is None
