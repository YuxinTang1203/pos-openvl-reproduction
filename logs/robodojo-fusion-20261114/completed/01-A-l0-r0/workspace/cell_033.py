w = robo.move_to([-0.30,-0.15,0.95], arm='left', gripper=robo.OPEN, pitch=0, yaw=0, path='line')
print("withdraw:", w.reached, w.steps_used)
import numpy as np
from collections import deque
img = robo.image('cam_head').astype(int)
# find white tube bodies (bright, low saturation) outside the arms
Rr,Gg,Bb = img[:,:,0],img[:,:,1],img[:,:,2]
white = (Rr>200)&(Gg>200)&(Bb>200)
ys,xs=np.nonzero(white)
print("white px", len(ys))
lab=np.zeros(white.shape,int); cur=0; res=[]
for i in range(white.shape[0]):
    for j in range(white.shape[1]):
        if white[i,j] and lab[i,j]==0:
            cur+=1; q=deque([(i,j)]); lab[i,j]=cur; pts=[]
            while q:
                a,b=q.popleft(); pts.append((a,b))
                for da in (-1,0,1):
                    for db in (-1,0,1):
                        na,nb=a+da,b+db
                        if 0<=na<480 and 0<=nb<640 and white[na,nb] and lab[na,nb]==0:
                            lab[na,nb]=cur; q.append((na,nb))
            if len(pts)>250:
                P=np.array(pts,float)
                res.append((len(pts), round(P[:,0].mean(),1), round(P[:,1].mean(),1), int(P[:,0].min()), int(P[:,0].max()), int(P[:,1].min()), int(P[:,1].max())))
for r in sorted(res, reverse=True): print(r)
