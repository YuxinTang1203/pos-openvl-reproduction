"""Automatic public-RGB perception; native locate calls cannot bypass MASt3R.

No object truth, depth, remembered height or automatic robot action is used.
Sparse clouds describe visible surface support, never a complete object pose.
"""
import copy
import hashlib
import time
import numpy as np
from .mast3r_geometry import Mast3rMatcher,triangulate,summarize_cloud
from .hosts import robodojo_tools as tools

KIND='v36_mast3r_auto'
PROMPT='''
Native RGB geometry now uses SAM3 -> MASt3R -> calibrated triangulation automatically.
Use the existing locate / locate_candidate / back_project interfaces; no separate
measure_geometry call or manual cross-camera candidate binding is needed.
ObjectSight.multiview_geometry contains status, source IDs, supported visible
surface geometry, uncertainty and model-call accounting. It is also delivered
in the PoS tool observation even if Python does not print the return value.
Legacy guessed center_xyz/top_z/height_m/grasp_yaw are not supplied by this mode.
Use the explicit visible-surface measurements and current RGB to reason about
object/target relations; a surface median is not an object centre or tube tip.
Sparse, ambiguous, failed or stale measurements return unknown/error; there is
no one-view height guess or table-plane fallback. Seek a better informative
view or state the unresolved geometry. back_project returns only a matched
surface point within 3 original pixels, not an interpolated or assumed depth.
SAM3 mask membership and geometric consistency do not certify same physical
point, semantic tip/hole, rigid attachment, contact, stability or task success.
Measurements expire with motion/native time, RGB or public calibration changes.
Perception time uses the original episode budget; robot actions are unchanged.
'''


def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def meta(view):
    return {'shape':list(view.shape),'intrinsic_matrix':[[view.fx,0,view.cx],[0,view.fy,view.cy],[0,0,1]],'extrinsic_matrix':np.asarray(view.pose).tolist()}

def snapshot(primitives,obs):
    # Only colour arrays, public image-state calibration and the native clock.
    state=tools._shown(obs);frames={}
    for name in sorted(obs.get('vision') or {}):
        color=(obs['vision'][name] or {}).get('color');view=tools._view(name,state)
        if color is None or view is None:continue
        rgb=np.array(np.asarray(color)[...,:3],dtype=np.uint8,copy=True)
        calibration=meta(view)
        frames[name]={'rgb':rgb,'meta':calibration,'rgb_sha256':digest(rgb)}
    step=primitives.env.get_status().get('step')
    if type(step) is not int:raise ValueError('Native observation clock unavailable')
    identity=(step,tuple((n,f['rgb_sha256'],str(f['meta'])) for n,f in frames.items()))
    return {'step':step,'frames':frames,'identity':identity,
            'calibration_state':'public image_state' if obs.get('image_state') else 'public state'}


def inside(mask,rc):
    rc=np.asarray(rc,float);ok=np.isfinite(rc).all(axis=1)
    ij=np.zeros(rc.shape,dtype=int);ij[ok]=np.rint(rc[ok]).astype(int)
    ok &= (ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<mask.shape[0])&(ij[:,1]<mask.shape[1])
    out=np.zeros(len(rc),bool);out[ok]=mask[ij[ok,0],ij[ok,1]]
    return out


class AutomaticGeometry:
    MIN_POINTS=6
    MIN_MASK_SHARE=.6
    MAX_SEEDS=3
    def __init__(self,matcher,event=lambda *a,**k:None):
        self.matcher=matcher;self.event=event;self.pending=[];self.cache={};self.sequence=0

    def _target_mask(self,primitives,rgb,rc,good):
        indices=np.flatnonzero(good);candidates=[]
        if not len(indices):return None,'no_source_supported_matches'
        # Deterministic spread across supported correspondences, never a plane
        # or assumed-height seed. At most three additional point segmentations.
        selected=indices[np.linspace(0,len(indices)-1,min(len(indices),self.MAX_SEEDS)).astype(int)]
        for i in selected:
            point=np.rint(rc[i]).astype(int)
            if any(m[point[0],point[1]] for m in candidates):continue
            res=primitives.sam3_client.segment(rgb,point=point.tolist(),min_score=.2)
            if not res.found or res.mask is None or res.score is None or not np.isfinite(res.score) or res.score<.2:continue
            mask=np.array(res.mask,dtype=bool,copy=True)
            if mask.shape!=rgb.shape[:2] or not mask[point[0],point[1]]:continue
            # A frame-filling point mask cannot bind a specific target instance.
            if mask.sum()>.8*mask.size:continue
            if any((mask&m).sum()/max(1,(mask|m).sum())>=.8 for m in candidates):continue
            candidates.append(mask)
        ranked=sorted([(int((good&inside(m,rc)).sum()),m) for m in candidates],key=lambda x:x[0],reverse=True)
        if not ranked or ranked[0][0]<self.MIN_POINTS:return None,'insufficient_dual_mask_support'
        if ranked[0][0]/len(indices)<self.MIN_MASK_SHARE:return None,'source_matches_split_across_instances'
        if len(ranked)>1 and ranked[1][0]>=max(self.MIN_POINTS,ranked[0][0]*.5):return None,'ambiguous_target_instance'
        return ranked[0][1],None

    def measure(self,primitives,obs,camera,mask):
        self.sequence+=1;started=time.monotonic();before_calls=getattr(self.matcher,'calls',0)
        result={'measurement_id':f'auto_mv_{self.sequence}','model':'MASt3R','mode':'automatic_native_perception',
          'status':'unknown','geometry_source':'mast3r_calibrated_visible_surface','fallback_used':False,
          'semantic_identity_verified':False,'correspondence_verified':False,'contact_verified':False,
          'learned_metric_points_used':False,'source_camera':camera,'views':1,'pairs':[],
          'uncertainty_scope':'Pixel noise/ray conditioning only; excludes wrong correspondence, identity and calibration bias.'}
        points=np.empty((0,3));rc_source=np.empty((0,2));sigmas=np.empty(0)
        cached=False
        try:
            snap=snapshot(primitives,obs);frames=snap['frames']
            if camera not in frames:raise ValueError('Source RGB/calibration unavailable')
            mask=np.array(mask,dtype=bool,copy=True)
            if mask.shape!=frames[camera]['rgb'].shape[:2] or not mask.any():raise ValueError('Source instance mask invalid')
            result.update(source_step=snap['step'],source_mask_sha256=digest(mask),source_mask_area_px=int(mask.sum()),calibration_state=snap['calibration_state'],
                source_frames=[{'camera':n,'rgb_sha256':f['rgb_sha256'],'calibration':f['meta']} for n,f in frames.items()],
                sensor_synchronization_verified=False,observation_bundle_verified=True)
            key=(snap['identity'],camera,digest(mask))
            if key in self.cache:
                old=self.cache[key];cached=True
                result.update(copy.deepcopy(old['result']),measurement_id=result['measurement_id'])
                points,rc_source,sigmas=old['points'],old['rc'],old['sigma']
            else:
                pp=[];rr=[];ss=[];source=frames[camera]
                for name,target in frames.items():
                    if name==camera:continue
                    pair={'target_camera':name,'accepted_matches':0};result['pairs'].append(pair)
                    try:
                        m=self.matcher.match(source['rgb'],target['rgb'])
                        geo=triangulate(source['meta'],target['meta'],m['rc_a'],m['rc_b'])
                        good=geo['accepted']&inside(mask,m['rc_a'])
                        pair.update(raw_matches=len(good),source_geometric_matches=int(good.sum()),matching_wall_s=m['match_wall_s'])
                        target_mask,reason=self._target_mask(primitives,target['rgb'],m['rc_b'],good)
                        if target_mask is None:pair['reason']=reason;continue
                        accepted=good&inside(target_mask,m['rc_b']);pair['accepted_matches']=int(accepted.sum())
                        pair['target_mask_sha256']=digest(target_mask)
                        if accepted.sum()<self.MIN_POINTS:pair['reason']='insufficient_calibrated_matches';continue
                        pp.append(geo['xyz_m'][accepted]);rr.append(np.asarray(m['rc_a'])[accepted]);ss.append(geo['sigma_m'][accepted]);result['views']+=1
                    except Exception as exc:
                        pair.update(status='error',reason=type(exc).__name__+': '+str(exc))
                if pp:
                    points=np.concatenate(pp);rc_source=np.concatenate(rr);sigmas=np.concatenate(ss)
                    # Count distinct original source pixels, not copies of the
                    # same feature matched against both alternate cameras.
                    _,ids=np.unique(np.rint(rc_source).astype(int),axis=0,return_index=True)
                    points,rc_source,sigmas=points[ids],rc_source[ids],sigmas[ids]
                    if len(points)>=self.MIN_POINTS:
                        result.update(status='measured_visible_surface',accepted_matches=len(points),geometry=summarize_cloud(points),
                            median_conditioning_sigma_m=float(np.median(sigmas)))
                    else:result['reason']='insufficient_unique_source_points'
                else:result['reason']='no_supported_cross_view_surface'
                self.cache={key:{'result':copy.deepcopy(result),'points':points,'rc':rc_source,'sigma':sigmas}}
            # Never publish historical pixels under a current native clock.
            if snapshot(primitives,tools._obs(primitives))['identity']!=snap['identity']:
                result.update(status='unknown',reason='observation_changed_during_perception');result.pop('geometry',None)
                points=np.empty((0,3));rc_source=np.empty((0,2));sigmas=np.empty(0);self.cache.clear()
        except Exception as exc:
            result.update(status='error',reason=type(exc).__name__+': '+str(exc));result.pop('geometry',None)
            points=np.empty((0,3));rc_source=np.empty((0,2));sigmas=np.empty(0)
        result.update(matcher_inference_calls=getattr(self.matcher,'calls',0)-before_calls,
                      measurement_cache_reused=cached,wall_s=round(time.monotonic()-started,6))
        self.event('automatic_mast3r_measurement',**copy.deepcopy(result));self.pending.append(copy.deepcopy(result))
        return result,points,rc_source,sigmas

    def locate(self,primitives,obs,camera,mask,base):
        result,_,_,_=self.measure(primitives,obs,camera,mask)
        return {**base,'views':result['views'],'geometry_source':result['geometry_source'],'multiview_geometry':result,
                'geometry_error':'Use multiview_geometry: visible surface support only; complete-object centre/height/grasp pose unmeasured.'}

    def back_project(self,primitives,obs,row,col,camera):
        frame=(obs.get('vision') or {}).get(camera) or {};rgb=frame.get('color')
        if rgb is None:return {'error':'Current source RGB unavailable'}
        if not (0<=row<rgb.shape[0] and 0<=col<rgb.shape[1]):return {'error':'Source pixel outside original image'}
        res=primitives.sam3_client.segment(rgb,point=[row,col],min_score=.2)
        if not res.found or res.mask is None or not np.asarray(res.mask,bool)[row,col]:
            result={'status':'unknown','reason':'source_instance_not_found','fallback_used':False}
            self.pending.append(copy.deepcopy(result));return {'error':result['reason'],'multiview_geometry':result}
        result,xyz,rc,sigma=self.measure(primitives,obs,camera,np.asarray(res.mask,bool))
        if result['status']!='measured_visible_surface' or not len(rc):return {'error':result.get('reason','No supported surface'),'multiview_geometry':result}
        distance=np.linalg.norm(rc-[row,col],axis=1);i=int(np.argmin(distance))
        if distance[i]>3:return {'error':'No matched surface point within 3 original pixels','multiview_geometry':result,'nearest_offset_px':float(distance[i])}
        view=tools._view(camera,tools._shown(obs));depth=float(-((xyz[i]-view.pos)@view.rot)[2])
        return {'camera':camera,'pixel':[row,col],'world_xyz':xyz[i].tolist(),'depth_m':depth,'surface':'matched_visible_surface',
                'matched_pixel_rc':rc[i].tolist(),'query_offset_px':float(distance[i]),'conditioning_sigma_m':float(sigma[i]),'multiview_geometry':result}


def validate_options(config):
    from .mast3r_adapter_v35 import validate_options as validate
    if config.get('kind')!=KIND:raise ValueError('Expected automatic MASt3R migration')
    cfg=copy.deepcopy(config);cfg['kind']='v35_mast3r';cfg['condition']='M'
    if config.get('condition')!='M':raise ValueError('Automatic mode requires condition M')
    validate(cfg)


def build_harness(kit,config,task,initial_images=()):
    from .ablation_v17 import build_harness as base
    from .mast3r_adapter_v35 import base_config
    from .hosts.robodojo_native import NativeBackend
    from .robodojo_xpl_rgb import RobodojoOfficialColourRobot
    validate_options(config)
    if not isinstance(kit.robo,RobodojoOfficialColourRobot) or not isinstance(kit.robo._backend,NativeBackend) or not kit.robo._backend.colour_only:
        raise ValueError('Automatic MASt3R requires the native RGB-only backend')
    cfg=copy.deepcopy(config);cfg['kind']='v35_mast3r'
    h=base(kit,base_config(cfg),task,initial_images)
    mc=config['mast3r'];provider=AutomaticGeometry(Mast3rMatcher(mc['repo'],mc['checkpoint'],device='cuda:0'),event=h._log)
    return install_provider(kit, h, provider, kind=KIND, prompt=PROMPT, model='MASt3R',
                            supported_status='measured_visible_surface')


def install_provider(kit, h, provider, *, kind, prompt, model, supported_status):
    """Bind an automatic geometry provider to native perception and PoS delivery."""
    from .hosts.robodojo_native import NativeBackend
    from .robodojo_xpl_rgb import RobodojoOfficialColourRobot
    if not isinstance(kit.robo, RobodojoOfficialColourRobot) or not isinstance(kit.robo._backend, NativeBackend) or not kit.robo._backend.colour_only:
        raise ValueError('Automatic geometry requires the native RGB-only backend')
    primitives=kit.robo._backend.toolkit._primitives
    if getattr(primitives,'multiview_provider',None) is not None:raise ValueError('Perception provider already installed')
    primitives.multiview_provider=provider
    previous_prompt=h.prompt;h.prompt=lambda:previous_prompt()+prompt
    previous_after=h.after_tool
    def after_tool(name,result,records):
        if provider.pending:
            delivered=copy.deepcopy(provider.pending);provider.pending.clear()
            try:fresh=snapshot(primitives,tools._obs(primitives))
            except Exception:fresh=None
            for g in delivered:
                same=fresh is not None and g.get('source_step')==fresh['step'] and all(
                    f['camera'] in fresh['frames'] and f['rgb_sha256']==fresh['frames'][f['camera']]['rgb_sha256']
                    and f['calibration']==fresh['frames'][f['camera']]['meta'] for f in g.get('source_frames',[]))
                g['current_eligible']=same and g.get('status')==supported_status
                if not same:g['delivery_status']='historical_or_unverifiable_at_delivery'
            result.result['automatic_multiview_geometry']=delivered
        return previous_after(name,result,records)
    h.after_tool=after_tool;h.kind=kind;h.geometry_provider=provider
    if model == 'MASt3R':
        h.mast3r_provider=provider
    h.provenance[kind]={'automatic_perception':True,'entry_points':['locate','locate_candidate','back_project'],
        'model':model,'device':'cuda:0','legacy_geometry_fallback':False,'automatic_robot_actions_added':False,
        'extra_llm_roles':False,'source':'native public RGB+image_state calibration','semantic_correspondence_verified':False}
    h._log('automatic_'+model.lower()+'_configured',config=h.provenance[kind])
    return h
