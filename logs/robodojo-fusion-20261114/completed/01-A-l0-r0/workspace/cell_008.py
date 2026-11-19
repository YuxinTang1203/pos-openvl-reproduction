import numpy as np
meta = robo.camera_meta('cam_head')
K = np.array(meta['intrinsic_matrix']); E = np.array(meta['extrinsic_matrix'])
R = E[:3,:3]; O = E[:3,3]
def proj(row, col, plane_z):
    p = np.array([col, row, 1.0]); x = np.linalg.inv(K) @ p
    d = R @ np.array([x[0], x[1], -1.0])
    t = (plane_z - O[2]) / d[2]
    return O + t*d
# sanity: plate pixel earlier gave (0.0236,0.0459,0.7871)
print("check", proj(183,330,0.7871).round(4))

img0 = robo.image('cam_head', step=0).astype(int)
blue = (img0[:,:,2] > img0[:,:,0]+25) & (img0[:,:,2] > 110)
r0,r1,c0,c1 = 155,220,260,385
sub = blue[r0:r1,c0:c1]
# fill holes: for each column find min/max blue row -> interior mask
inter = np.zeros_like(sub)
for j in range(sub.shape[1]):
    idx = np.nonzero(sub[:,j])[0]
    if len(idx)>3: inter[idx[0]:idx[-1]+1, j] = True
hole = inter & ~sub
from collections import deque
lab = np.zeros(hole.shape,int); cur=0; comps=[]
for i in range(hole.shape[0]):
    for j in range(hole.shape[1]):
        if hole[i,j] and lab[i,j]==0:
            cur+=1; q=deque([(i,j)]); lab[i,j]=cur; pts=[]
            while q:
                a,b=q.popleft(); pts.append((a,b))
                for da,db in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    na,nb=a+da,b+db
                    if 0<=na<hole.shape[0] and 0<=nb<hole.shape[1] and hole[na,nb] and lab[na,nb]==0:
                        lab[na,nb]=cur; q.append((na,nb))
            if 30<=len(pts)<=300:
                P=np.array(pts,float)
                cy,cx=P[:,0].mean(),P[:,1].mean()
                h=cy+r0; w=cx+c0
                # circularity: extent
                ex = (P[:,0].max()-P[:,0].min()+1); ey=(P[:,1].max()-P[:,1].min()+1)
                if 0.5 < len(pts)/(ex*ey) < 1.0 and 0.6<ex/ey<1.6:
                    comps.append((round(h,1), round(w,1), len(pts), round(ex,1), round(ey,1)))
comps.sort()
print("n holes", len(comps))
for cc in comps:
    wxyz = proj(cc[0], cc[1], 0.7871)
    print(cc, "world", wxyz.round(4))
