import numpy as np
# 1) find hole openings: pixels over the plate whose ray reaches BELOW the plate top
im = robo.image('cam_head').astype(int)
R,G,B = im[...,0],im[...,1],im[...,2]
blue = (B > R + 15) & (B > 90)
ys,xs = np.where(blue[140:240,240:420]); ys=ys+140; xs=xs+240
print("plate rows",ys.min(),ys.max(),"cols",xs.min(),xs.max())
through=[]
for r in range(ys.min()+4, ys.max()-3, 4):
    for c in range(xs.min()+4, xs.max()-3, 4):
        if blue[r,c]:   continue  # solid plate
        p = robo.back_project(r,c,'cam_head')
        z = p.xyz[2]
        if z < 0.80:
            through.append((r,c,round(z,3),p.surface))
print("n through", len(through))
for t in through: print(t)
