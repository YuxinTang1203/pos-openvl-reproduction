import numpy as np
im = robo.image('cam_head').astype(int)
R,G,B = im[...,0],im[...,1],im[...,2]
blue = (B > R + 15) & (B > 90)
# hole blobs: non-blue inside plate bbox
r0,r1,c0,c1 = 168,221,269,370
hole = ~blue
lab = np.zeros(im.shape[:2], int); nxt=1; comps=[]
for i in range(r0,r1+1):
    for j in range(c0,c1+1):
        if hole[i,j] and lab[i,j]==0:
            stack=[(i,j)]; lab[i,j]=nxt; pix=[]
            while stack:
                y,x=stack.pop(); pix.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=y+dy,x+dx
                    if r0<=yy<=r1 and c0<=xx<=c1 and hole[yy,xx] and lab[yy,xx]==0:
                        lab[yy,xx]=nxt; stack.append((yy,xx))
            ys=np.array([p[0] for p in pix]); xs=np.array([p[1] for p in pix])
            h=ys.max()-ys.min()+1; w=xs.max()-xs.min()+1
            if len(pix)>=20 and 4<=h<=14 and 4<=w<=16:
                comps.append((round(ys.mean(),1), round(xs.mean(),1), len(pix), h, w))
            nxt+=1
print("hole blobs (row,col,area,h,w):")
for cc in comps: print(cc)
pts=[]
for rr,cc_,a,h,w in comps:
    p = robo.visual_project('cam_head' if False else None or 'x', 0,0,0.815) if False else None
print("use visual_project next")
