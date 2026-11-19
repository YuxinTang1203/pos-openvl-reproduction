import numpy as np
cur = robo.image('cam_head').astype(int)
init = robo.image('cam_head', step=0).astype(int)

# 1) sample blue plate pixels in CURRENT frame (unoccluded) for a pixel->world plane fit
blue = (cur[:,:,2] > cur[:,:,0]+25) & (cur[:,:,2] > 110)
r0,r1,c0,c1 = 150,215,260,385
ys,xs = np.nonzero(blue[r0:r1,c0:c1])
pts_px = list(zip(ys+r0, xs+c0))
# pick 6 spread points: extremes of the visible blue set
def spread(pts, k=6):
    P=np.array(pts,float)
    sel=[]
    order=[(P[:,1],1),(P[:,1],-1),(P[:,0],1),(P[:,0],-1),(P[:,0]+P[:,1],1),(P[:,0]+P[:,1],-1)]
    for key,s in order:
        i=int(np.argmax(key*s)); sel.append((int(P[i,0]),int(P[i,1])))
    return sel
samp = spread(pts_px)
print("sample plate px:", samp)
W=[]
for (r,c) in samp:
    p=robo.back_project(r,c,'cam_head')
    W.append((r,c,p.xyz,p.surface))
    print(r,c,p.surface,[round(v,4) for v in p.xyz])
