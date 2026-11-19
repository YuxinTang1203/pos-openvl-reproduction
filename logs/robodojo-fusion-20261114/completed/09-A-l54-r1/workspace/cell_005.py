import numpy as np
im = robo.image('cam_head').astype(int)
R,G,B = im[...,0], im[...,1], im[...,2]
blue = (B > R + 15) & (B > 90)
win = np.zeros_like(blue); win[160:220, 260:375] = True
bluep = blue & win
inner = np.zeros_like(blue); inner[168:214, 269:367] = True
hole = inner & (~bluep)
lab = np.zeros(hole.shape, int); nxt=1; comps=[]
for i in range(168,214):
    for j in range(269,367):
        if hole[i,j] and lab[i,j]==0:
            stack=[(i,j)]; lab[i,j]=nxt; pix=[]
            while stack:
                y,x=stack.pop(); pix.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=y+dy,x+dx
                    if 168<=yy<214 and 269<=xx<367 and hole[yy,xx] and lab[yy,xx]==0:
                        lab[yy,xx]=nxt; stack.append((yy,xx))
            ys=np.array([p[0] for p in pix]); xs=np.array([p[1] for p in pix])
            touches = ys.min()<=168 or ys.max()>=213 or xs.min()<=269 or xs.max()>=366
            if len(pix)>=15 and not touches:
                comps.append((len(pix), ys.mean(), xs.mean(), ys.max()-ys.min()+1, xs.max()-xs.min()+1))
            nxt+=1
for c in sorted(comps, reverse=True): print("area,row,col,h,w:", c)
