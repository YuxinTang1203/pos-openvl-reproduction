"""Opt-in calibrated MASt3R measurements inside original PoS + OpenVL.

No V34 draft stage, automatic robot motion or task-success certification.
"""
import copy
import hashlib
import json
import numpy as np
from .ablation_v17 import build_harness as build_base, validate_options as validate_base
from .mast3r_geometry import Mast3rMatcher, triangulate, summarize_cloud
from .openvl_geometry import calibrated_view

KIND='v35_mast3r'
PROMPT='''
Calibrated multi-view geometry is available inside Python:
g = robo.measure_geometry(binding_a, candidate_a, binding_b, candidate_b)
Select current SAM3 masks for the SAME physical object in two different cameras.
Use Visual_Grounding_Tool on each camera first; candidate numbers do not establish
cross-camera identity. Matches must fall inside BOTH selected masks.
This reads the selected current RGB images and public calibration, runs MASt3R
matching, and triangulates matches; it does not move the robot. It returns
visible surface geometry, uncertainty diagnostics and a geometry_id. A sparse
or inconsistent result is unknown, not a reason to invent a coordinate. Obtain
a more informative view when needed. No object CAD, simulator depth or hidden
object pose is read. Calls consume the existing episode wall-time budget.
p = robo.geometry_point(geometry_id, row, col)
This retrieves the nearest supported matched surface point, reporting its pixel
offset. It is not automatically the tube tip, hole centre or object centre.
q = robo.geometry_plane_point(geometry_id, row, col)
This intersects the ray with the fitted visible plane, if one exists; it is
conditional on that plane and the selected semantic feature. Inspect these
assumptions. No fit, score or API status certifies contact or task success.
All geometry expires after motion; remeasure to check each correction and the
state after release/retreat. Keep source/frame IDs with physical claims.
The host supplies measurements, not an insertion policy or an action gate.
'''


class GeometryBridge:
    def __init__(self, kit, matcher):
        self.kit=kit;self.visual=kit.evidence_adapter;self.matcher=matcher
        self.entries={};self.pending=[]

    def measure(self,binding_id,candidate_number,target_binding_id,target_candidate_number):
        v=self.visual;robo=self.kit.robo
        source,data,mask,pixels=v._current_candidate(binding_id,candidate_number)
        other,other_data,target_mask,target=v._current_candidate(target_binding_id,target_candidate_number)
        if other['camera']==source['camera']:
            raise ValueError('Choose masks from two different public cameras')
        revision=v.action_revision();step=robo._backend.env_steps()
        meta_a=v._original_camera_meta(source['camera']);meta_b=v._original_camera_meta(other['camera'])
        target_id=other['image_id']
        if v.action_revision()!=revision or robo._backend.env_steps()!=step or not all(v.frame_context(i)['current_camera_pixels'] for i in [source['image_id'],target_id]):
            raise ValueError('Scene changed while acquiring synchronized calibration/images')
        m=self.matcher.match(pixels,target)
        geo=triangulate(meta_a,meta_b,m['rc_a'],m['rc_b'])
        rc=np.rint(m['rc_a']).astype(int)
        inside=mask[rc[:,0],rc[:,1]]
        target_rc=np.rint(m['rc_b']).astype(int)
        target_inside=target_mask[target_rc[:,0],target_rc[:,1]]
        accepted=geo['accepted']&inside&target_inside
        if v.action_revision()!=revision or robo._backend.env_steps()!=step:
            raise ValueError('Measurement became historical during matching')
        key='mv3d_'+str(len(self.entries)+1)
        result={'geometry_id':key,'source_frame_ids':[source['image_id'],target_id],
            'source_step':step,'action_revision':revision,'source_binding_id':binding_id,
            'candidate_number':candidate_number,'target_binding_id':target_binding_id,
            'target_candidate_number':target_candidate_number,'cameras':[source['camera'],other['camera']],
            'raw_matches':len(inside),'source_region_matches':int(inside.sum()),'accepted_matches':int(accepted.sum()),
            'both_mask_matches':int((inside&target_inside).sum()),
            'matching_wall_s':m['match_wall_s'],'geometry':summarize_cloud(geo['xyz_m'][accepted]),
            'median_reprojection_px':float(np.median(geo['reprojection_px'][accepted])) if accepted.any() else None,
            'median_conditioning_sigma_m':float(np.median(geo['sigma_m'][accepted])) if accepted.any() else None,
            'uncertainty_scope':'Pixel noise and ray conditioning only; wrong identity, calibration bias and unseen geometry excluded.',
            'semantic_identity_verified':False,'rigid_grasp_verified':False,'contact_verified':False,
            'learned_metric_points_used':False}
        self.entries[key]={'result':result,'rc':m['rc_a'][accepted],
            'xyz':geo['xyz_m'][accepted],'sigma':geo['sigma_m'][accepted],'meta':meta_a}
        v.event('v35_calibrated_measurement',**copy.deepcopy(result))
        self.pending.append(copy.deepcopy(result))
        return result

    def _entry(self,key):
        if key not in self.entries:raise ValueError('Unknown measured geometry_id')
        e=self.entries[key];r=e['result']
        if self.visual.action_revision()!=r['action_revision'] or self.kit.robo._backend.env_steps()!=r['source_step']:
            raise ValueError('Geometry is historical after motion; obtain a fresh measurement')
        return e

    def point(self,key,row,col,max_pixel_radius=3.):
        e=self._entry(key)
        if not np.isfinite([row,col,max_pixel_radius]).all() or max_pixel_radius<=0:
            raise ValueError('Finite pixel query and positive radius required')
        if not len(e['rc']):return {'status':'unknown','reason':'no_valid_calibrated_matches'}
        distance=np.linalg.norm(e['rc']-[row,col],axis=1);i=int(np.argmin(distance))
        if distance[i]>max_pixel_radius:
            return {'status':'unknown','reason':'no_match_near_requested_pixel','nearest_offset_px':float(distance[i])}
        return {'status':'matched_visible_surface_point','geometry_id':key,'xyz_m':e['xyz'][i].tolist(),
            'matched_pixel_rc':e['rc'][i].tolist(),'query_offset_px':float(distance[i]),'conditioning_sigma_m':float(e['sigma'][i]),
            'interpretation':'Nearest matched visible surface; semantic feature and hidden endpoint unverified.'}

    def plane_point(self,key,row,col):
        e=self._entry(key);plane=e['result']['geometry'].get('plane')
        if plane is None:return {'status':'unknown','reason':'no_supported_plane_fit'}
        view=calibrated_view(e['meta'])
        if not np.isfinite([row,col]).all() or not (0<=row<view.shape[0] and 0<=col<view.shape[1]):
            raise ValueError('Pixel outside calibrated original image')
        ray=view.rays(np.asarray(row),np.asarray(col));normal=np.asarray(plane['normal']);origin=np.asarray(plane['origin_m'])
        denominator=float(ray@normal)
        if abs(denominator)<1e-6:return {'status':'unknown','reason':'ray_parallel_to_plane'}
        t=float((origin-view.pos)@normal)/denominator
        if t<=0:return {'status':'unknown','reason':'intersection_behind_camera'}
        return {'status':'conditional_on_fitted_plane','geometry_id':key,'xyz_m':(view.pos+t*ray).tolist(),
                'fitted_plane':copy.deepcopy(plane),'interpretation':'Plane-ray intersection; hole identity, hole axis and contact unverified.'}


def base_config(config):
    if not isinstance(config,dict) or config.get('kind')!=KIND or set(config)-{'kind','condition','pos','openvl','mast3r'} or config.get('condition') not in {'A','M'}:
        raise ValueError('v35 requires A/M and explicit original PoS/OpenVL settings')
    return {'kind':'v17_ablation','profile':'fusion','static_rules':False,'action_focus':False,'reliability':False,
            'pos':copy.deepcopy(config.get('pos',{})),'openvl':copy.deepcopy(config.get('openvl'))}


def validate_options(config):
    validate_base(base_config(config))
    if config['condition']=='M':
        m=config.get('mast3r')
        if not isinstance(m,dict) or set(m)-{'repo','checkpoint','device'} or not all(isinstance(m.get(k),str) and m[k] for k in ['repo','checkpoint']):
            raise ValueError('Explicit local MASt3R repo/checkpoint required')
        if m.get('device','cuda:0')!='cuda:0':raise ValueError('GPU inference required; physical UUID must pass ownership admission')


def build_harness(kit,config,task,initial_images=()):
    validate_options(config)
    h=build_base(kit,base_config(config),task,initial_images)
    enabled=config['condition']=='M'
    if enabled:
        cfg=config['mast3r'];bridge=GeometryBridge(kit,Mast3rMatcher(cfg['repo'],cfg['checkpoint'],device=cfg.get('device','cuda:0')))
        for name,fn in [('measure_geometry',bridge.measure),('geometry_point',bridge.point),('geometry_plane_point',bridge.plane_point)]:
            kit._globals[name]=fn;setattr(kit.robo,name,fn)
        prompt=h.prompt;h.prompt=lambda:prompt()+PROMPT
        after=h.after_tool
        def after_tool(name,result,records):
            if bridge.pending:
                result.result['multiview_geometry']=copy.deepcopy(bridge.pending);bridge.pending.clear()
            return after(name,result,records)
        h.after_tool=after_tool;h.mast3r_bridge=bridge
    h.kind=KIND
    h.provenance[KIND]={'condition':config['condition'],'model_added':'MASt3R' if enabled else None,
        'original_policy':'v17_ablation/fusion A','v34_draft_stage':False,'robot_runtime_modified':False,
        'automatic_robot_actions_added':False,'new_model_calls_are_local_perception':enabled,
        'new_llm_roles':False,'learned_point_maps_used_as_metric_measurement':False,
        'calibrated_point_type':'visible surface hypothesis, not certified semantic tip/contact',
        'new_perception_wall_time_in_original_budget':True,'device':config.get('mast3r',{}).get('device','cuda:0') if enabled else None}
    h._log('v35_mast3r_configured',config=h.provenance[KIND])
    return h
