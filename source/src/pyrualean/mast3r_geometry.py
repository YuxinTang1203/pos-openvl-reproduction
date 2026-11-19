"""MASt3R image correspondences plus calibrated metric geometry.

Learned point-map coordinates are deliberately not used as metric observations.
Correspondence identity, calibration and object rigidity remain assumptions.
"""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
from .openvl_geometry import calibrated_view


def triangulate(meta_a, meta_b, rc_a, rc_b, *, max_reprojection_px=2.0,
                min_parallax_deg=1.5, pixel_sigma_px=1.0):
    a, b = calibrated_view(meta_a), calibrated_view(meta_b)
    pa, pb = np.asarray(rc_a, float), np.asarray(rc_b, float)
    if pa.ndim != 2 or pa.shape[1] != 2 or pb.shape != pa.shape:
        raise ValueError('Expected aligned Nx2 original-image row,col coordinates')
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError('Nonfinite pixels')
    if max_reprojection_px <= 0 or not 0 < min_parallax_deg < 90 or pixel_sigma_px < 0:
        raise ValueError('Invalid geometric thresholds')
    for p, v in [(pa,a),(pb,b)]:
        if np.any(p<0) or np.any(p[:,0]>=v.shape[0]) or np.any(p[:,1]>=v.shape[1]):
            raise ValueError('Pixels outside original calibrated image')
    if np.linalg.norm(a.pos-b.pos) < 1e-5:
        raise ValueError('Camera baseline is zero or too small')
    n = len(pa)
    if not n:
        return {'accepted':np.zeros(0,bool),'xyz_m':np.empty((0,3)),
                'reprojection_px':np.empty((0,2)),'parallax_deg':np.empty(0),
                'ray_gap_m':np.empty(0),'sigma_m':np.empty(0)}
    ra, rb = a.rays(pa[:,0],pa[:,1]), b.rays(pb[:,0],pb[:,1])
    ra /= np.linalg.norm(ra,axis=1)[:,None]
    rb /= np.linalg.norm(rb,axis=1)[:,None]
    cosine = np.sum(ra*rb,axis=1)
    det = 1-cosine*cosine
    delta = b.pos-a.pos
    da, db = ra@delta, rb@delta
    ta = (da-cosine*db)/np.maximum(det,1e-15)
    tb = (cosine*da-db)/np.maximum(det,1e-15)
    xa, xb = a.pos+ta[:,None]*ra, b.pos+tb[:,None]*rb
    xyz=(xa+xb)/2
    gap=np.linalg.norm(xa-xb,axis=1)
    ea,eb=[],[]
    depths=[]
    for view,p,errs in [(a,pa,ea),(b,pb,eb)]:
        row,col,depth=view.project(xyz)
        errs.extend(np.linalg.norm(np.c_[row,col]-p,axis=1))
        depths.append(depth)
    error=np.c_[ea,eb]
    # The supplementary angle is also poorly conditioned for opposed rays.
    angle=np.degrees(np.arcsin(np.sqrt(np.clip(det,0,1))))
    accepted=(ta>0)&(tb>0)&(np.asarray(depths)>0).all(axis=0)
    accepted &= (angle>=min_parallax_deg)&(error.max(axis=1)<=max_reprojection_px)
    accepted &= np.isfinite(xyz).all(axis=1)
    # A first-order conditioning diagnostic, not a calibrated error bound.
    sigma = pixel_sigma_px*np.sqrt((ta/min(a.fx,a.fy))**2+(tb/min(b.fx,b.fy))**2)/np.sqrt(np.maximum(det,1e-15))
    return {'accepted':accepted,'xyz_m':xyz,'reprojection_px':error,
            'parallax_deg':angle,'ray_gap_m':gap,'sigma_m':sigma}


def summarize_cloud(xyz, *, plane_residual_m=0.002, min_plane_points=24):
    """Visible-surface hypotheses only; no semantic tip/axis/contact certification."""
    p=np.asarray(xyz,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 cloud')
    if len(p)<3:
        return {'status':'unknown','reason':'insufficient_calibrated_points'}
    center=np.median(p,axis=0)
    _,s,vt=np.linalg.svd(p-center,full_matrices=False)
    out={'status':'visible_surface_geometry','points':len(p),'median_xyz_m':center.tolist(),
         'principal_direction':vt[0].tolist(),'axis_dominance':float(s[0]/max(s[1],1e-12)),
         'extent_along_direction_m':np.quantile((p-center)@vt[0],[.02,.98]).tolist(),
         'interpretation':'Surface-cloud principal direction and visible extent; not a measured physical tip or full 6D pose.',
         'plane':None}
    rng=np.random.default_rng(0);best=None
    for _ in range(192):
        q=p[rng.choice(len(p),3,replace=False)];normal=np.cross(q[1]-q[0],q[2]-q[0]);norm=np.linalg.norm(normal)
        if norm<1e-10:continue
        normal/=norm;mask=np.abs((p-q[0])@normal)<=plane_residual_m
        if best is None or mask.sum()>best.sum():best=mask
    if best is not None and best.sum()>=min_plane_points and best.mean()>=.45:
        q=p[best];origin=q.mean(axis=0);_,splane,vtplane=np.linalg.svd(q-origin,full_matrices=False)
        # A nearly collinear tube must not become a reliable surface plane.
        if splane[1]/max(splane[0],1e-12)>.1:
            normal=vtplane[-1];res=np.abs((q-origin)@normal)
            out['plane']={'origin_m':origin.tolist(),'normal':normal.tolist(),'inliers':int(best.sum()),
                          'support_fraction':float(best.mean()),'median_residual_m':float(np.median(res)),
                          'interpretation':'Dominant visible-plane fit; semantic identity and unseen geometry unverified.'}
    return out


class Mast3rMatcher:
    """Lazy local model, unchanged vendor code, explicit resize-to-original mapping."""
    def __init__(self, repo, checkpoint, *, device='cuda:0', image_size=512, threads=4):
        if device not in {'cpu','cuda:0'}:
            raise ValueError('Use logical cuda:0 on a separately admitted physical GPU UUID')
        self.repo=Path(repo).resolve();self.checkpoint=Path(checkpoint).resolve()
        self.device=device;self.image_size=image_size;self.threads=threads
        self.model=None;self.cache={};self.calls=0

    def load(self):
        if self.model is not None:return
        if self.device=='cpu':
            raise RuntimeError('CPU model inference disabled by user preference; wait for an admitted GPU')
        from .gpu_ownership_v35 import assert_visible_gpu_owned
        assert_visible_gpu_owned()
        import torch
        torch.set_num_threads(self.threads)
        sys.path.insert(0,str(self.repo))
        from mast3r.model import AsymmetricMASt3R
        self.model=AsymmetricMASt3R.from_pretrained(str(self.checkpoint)).to(device=self.device,dtype=torch.float32).eval()

    def _prepare(self, rgb, index):
        import torch
        from PIL import Image
        im=Image.fromarray(np.asarray(rgb,dtype=np.uint8),'RGB');w,h=im.size
        scale=self.image_size/max(w,h)
        nw=max(16,round(w*scale/16)*16);nh=max(16,round(h*scale/16)*16)
        arr=np.asarray(im.resize((nw,nh),Image.Resampling.LANCZOS),dtype=np.float32).copy()
        tensor=torch.from_numpy(arr).permute(2,0,1)[None]/127.5-1
        item={'img':tensor,'true_shape':np.int32([[nh,nw]]),'idx':index,'instance':str(index)}
        return item,np.array([nh/h,nw/w])

    def match(self, rgb_a, rgb_b):
        if self.device=='cpu':
            raise RuntimeError('CPU model inference disabled; no fallback')
        from .gpu_ownership_v35 import assert_visible_gpu_owned
        assert_visible_gpu_owned()
        key=tuple(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in [rgb_a,rgb_b])
        if key in self.cache:return self.cache[key]
        self.load()
        import torch
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        t=time.monotonic();a,sa=self._prepare(rgb_a,0);b,sb=self._prepare(rgb_b,1)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            # SAM3 may select BF16 in this process. Keep MASt3R model, decoder
            # indexed writes and reciprocal-NN buffers consistently FP32.
            out=inference([(a,b)],self.model,self.device,batch_size=1,verbose=False)
            d0=out['pred1']['desc'].squeeze(0).detach().float();d1=out['pred2']['desc'].squeeze(0).detach().float()
            # Bounded GPU blocks avoid a dense all-pairs descriptor allocation.
            xy0,xy1=fast_reciprocal_NNs(d0,d1,subsample_or_initxy1=8,device=self.device,dist='dot',block_size=4096)
            ix0=np.asarray(xy0,int);ix1=np.asarray(xy1,int)
            dot=(d0[ix0[:,1],ix0[:,0]]*d1[ix1[:,1],ix1[:,0]]).sum(-1).cpu().numpy()
        # PIL resize maps pixel centres, independently in each dimension.
        rc0=(ix0[:,::-1]+.5)/sa-.5;rc1=(ix1[:,::-1]+.5)/sb-.5
        # Ignore resize-edge samples; calibration remains in original pixels.
        valid=np.ones(len(rc0),bool)
        for rc,img in [(rc0,rgb_a),(rc1,rgb_b)]:
            valid &= (rc[:,0]>=1)&(rc[:,1]>=1)&(rc[:,0]<img.shape[0]-1)&(rc[:,1]<img.shape[1]-1)
        result={'rc_a':rc0[valid],'rc_b':rc1[valid],'descriptor_cosine':dot[valid],
                'match_wall_s':time.monotonic()-t,'learned_metric_points_used':False,
                'original_shapes':[list(rgb_a.shape[:2]),list(rgb_b.shape[:2])]}
        self.calls+=1;self.cache[key]=result
        if len(self.cache)>2:self.cache.pop(next(iter(self.cache)))
        return result
