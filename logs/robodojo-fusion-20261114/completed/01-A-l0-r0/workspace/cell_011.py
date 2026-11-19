import numpy as np
from collections import deque
img0 = robo.image('cam_head', step=0).astype(int)
B=img0[:,:,2]; Rr=img0[:,:,0]; G=img0[:,:,1]
blue = (B > Rr+25) & (B > 110)
r0,r1,c0,c1 = 150,225,255,395
sub = blue[r0:r1, c0:c1]
# fill plate column-wise
filled = np.zeros_like(sub)
for j in range(sub.shape[1]):
    idx=np.nonzero(sub[:,j])[0]
    if len(idx)>3: filled[idx[0]:idx[-1]+1, j]=True
for i in range(sub.shape[0]):
    idx=np.nonzero(sub[i,:])[0]
    if len(idx)>3: filled[i, idx[0]:idx[-1]+1]=True
holes = filled & ~sub
lab=np.zeros(holes.shape,int); cur=0; res=[]
for i in range(holes.shape[0]):
    for j in range(holes.shape[1]):
        if holes[i,j] and lab[i,j]==0:
            cur+=1; q=deque([(i,j)]); lab[i,j]=cur; pts=[]
            while q:
                a,b=q.popleft(); pts.append((a,b))
                for da in (-1,0,1):
                    for db in (-1,0,1):
                        na,nb=a+da,b+db
                        if 0<=na<holes.shape[0] and 0<=nb<holes.shape[1] and holes[na,nb] and lab[na,nb]==0:
                            lab[na,nb]=cur; q.append((na,nb))
            P=np.array(pts,float)
            ex=P[:,0].max()-P[:,0].min()+1; ey=P[:,1].max()-P[:,1].min()+1
            if len(pts)>=40 and 4<=ex<=25 and 4<=ey<=25:
                res.append((round(P[:,0].mean()+r0,1), round(P[:,1].mean()+c0,1), len(pts), int(ex), int(ey)))
res.sort()
print("n", len(res))
for t in res: print(t)
