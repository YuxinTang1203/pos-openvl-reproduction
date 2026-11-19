import numpy as np
im = robo.image('cam_head')
# find blue plate bbox again in current frame
a = im.astype(int); R,G,B = a[...,0],a[...,1],a[...,2]
blue = (B > R + 15) & (B > 90)
win = np.zeros_like(blue); win[140:240, 240:420] = True
blue = blue & win
ys, xs = np.where(blue)
print("plate rows", ys.min(), ys.max(), "cols", xs.min(), xs.max(), "n", len(ys))
r_mid = int((ys.min()+ys.max())/2); c_mid = int((xs.min()+xs.max())/2)
p = robo.back_project(r_mid, c_mid, 'cam_head')
print("plate centre pixel", r_mid, c_mid, p.surface, tuple(round(v,4) for v in p.xyz))
# right-edge hole area pixel (near-row right hole was around row 199 col 352 in step0)
for (rr,cc) in [(ys.max()-16, xs.max()-16), (ys.max()-16, xs.max()-40), (ys.min()+16, xs.max()-16)]:
    q = robo.back_project(rr, cc, 'cam_head')
    print((rr,cc), q.surface, tuple(round(v,4) for v in q.xyz))
