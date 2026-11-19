import numpy as np
img = robo.image('cam_head').astype(int)
# plate region (rows 150-220, cols 265-380) from the rack mask box
r0,r1,c0,c1 = 150,222,262,382
sub = img[r0:r1, c0:c1]
lum = sub.mean(2)
print("lum percentiles", np.percentile(lum,[5,20,40,60,80,95]).round(1))
# holes are the dark openings; plate is bright blue
thr = np.percentile(lum,32)
m = lum < thr
# simple BFS labelling
lab = np.zeros(m.shape,int); cur=0; comps=[]
from collections import deque
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        if m[i,j] and lab[i,j]==0:
            cur+=1; q=deque([(i,j)]); lab[i,j]=cur; pts=[]
            while q:
                a,b=q.popleft(); pts.append((a,b))
                for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
                    na,nb=a+da,b+db
                    if 0<=na<m.shape[0] and 0<=nb<m.shape[1] and m[na,nb] and lab[na,nb]==0:
                        lab[na,nb]=cur; q.append((na,nb))
            if 25<=len(pts)<=400:
                aa=np.mean([p[0] for p in pts]); bb=np.mean([p[1] for p in pts])
                comps.append((round(aa+r0,1), round(bb+c0,1), len(pts)))
print("n comps", len(comps))
for cc in comps: print(cc)
