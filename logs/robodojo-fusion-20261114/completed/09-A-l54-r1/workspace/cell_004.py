import numpy as np
im = robo.image('cam_head').astype(int)
R,G,B = im[...,0], im[...,1], im[...,2]
blue = (B > R + 15) & (B > 90)
# restrict to rack area
mask = np.zeros_like(blue); mask[140:240, 230:420] = True
blue = blue & mask
rows = np.any(blue, axis=1); cols = np.any(blue, axis=0)
r0,r1 = np.where(rows)[0].min(), np.where(rows)[0].max()
c0,c1 = np.where(cols)[0].min(), np.where(cols)[0].max()
print("plate bbox rows", r0, r1, "cols", c0, c1, "blue px", blue.sum())

# holes = non-blue pixels inside the plate bbox region
inner = np.zeros_like(blue)
inner[r0:r1+1, c0:c1+1] = True
dark = (R+G+B) < 300
hole = inner & (~blue) & dark & mask
# label components (4-connectivity BFS)
lab = np.zeros(hole.shape, int); nxt=1
H,W = hole.shape
for i in range(r0, r1+1):
    for j in range(c0, c1+1):
        if hole[i,j] and lab[i,j]==0:
            stack=[(i,j)]; lab[i,j]=nxt; pix=[]
            while stack:
                y,x = stack.pop(); pix.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W and hole[yy,xx] and lab[yy,xx]==0:
                        lab[yy,xx]=nxt; stack.append((yy,xx))
            if len(pix) >= 12:
                ys=np.array([p[0] for p in pix]); xs=np.array([p[1] for p in pix])
                print("comp",nxt,"area",len(pix),"rc",(ys.mean(), xs.mean()),
                      "rowspan",(ys.min(),ys.max()),"colspan",(xs.min(),xs.max()))
            nxt+=1
