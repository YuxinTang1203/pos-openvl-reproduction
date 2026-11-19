import numpy as np
img = robo.image('cam_head').astype(int)
Rr,Gg,Bb = img[:,:,0],img[:,:,1],img[:,:,2]
white = (Rr>200)&(Gg>200)&(Bb>200)
orange = (Rr>200)&(Gg>120)&(Gg<200)&(Bb<140)
# tube1 region rows 135-185, cols 255-305
sub_w = np.zeros_like(white); sub_w[135:190,250:310]=white[135:190,250:310]
sub_o = np.zeros_like(orange); sub_o[135:190,250:310]=orange[135:190,250:310]
ys,xs=np.nonzero(sub_w); yo,xo=np.nonzero(sub_o)
print("white n",len(ys),"centroid",(ys.mean(),xs.mean()) if len(ys) else None)
print("orange n",len(yo),"centroid",(yo.mean(),xo.mean()) if len(yo) else None)
if len(ys)>0 and len(yo)>0:
    c=np.array([xs.mean(),ys.mean()]); o=np.array([xo.mean(),yo.mean()])
    d=o-c; L=np.hypot(*d)
    print("cap dir (col,row)", d/L, "dist px", L)
    # axis from cap end through centre; cap end point = o + unit*L2
    # project pixels to world on table plane z=0.777 for the lying tube
    for nm,(r,cx) in [("centre",(ys.mean(),xs.mean())),("cap",(yo.mean(),xo.mean()))]:
        p=robo.visual_project('frame_0033' if False else None, 0,0,0.777) if False else None
    # use back_project on the two pixels
    for nm,(r,cx) in [("centre",(int(ys.mean()),int(xs.mean()))),("cap",(int(yo.mean()),int(xo.mean())))]:
        p=robo.back_project(r,cx,'cam_head'); print(nm,p.surface,np.round(p.xyz,4))
