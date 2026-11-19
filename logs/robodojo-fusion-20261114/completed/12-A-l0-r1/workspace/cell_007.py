import numpy as np
from scipy import ndimage
img = robo.image('cam_head').astype(int)
R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]
blue = (B > R + 20) & (B > 80)
blue_d = ndimage.binary_dilation(blue, iterations=3)
ys, xs = np.nonzero(blue_d)
r0,r1,c0,c1 = ys.min(), ys.max(), xs.min(), xs.max()
print('rack bbox rows', r0, r1, 'cols', c0, c1)
lum = img.mean(axis=2)
dark = (lum < 110) & blue_d
lab, n = ndimage.label(dark)
print('n comp', n)
holes = []
for i in range(1, n+1):
    sz = int((lab==i).sum())
    if sz >= 25:
        rr, cc = np.nonzero(lab==i)
        holes.append((int(rr.mean()), int(cc.mean()), sz))
holes.sort()
pts = []
for (rr, cc, sz) in holes:
    p = robo.back_project(rr, cc, 'cam_head')
    print('hole px', rr, cc, 'sz', sz, 'surface', p.surface, 'xyz', [round(v,4) for v in p.xyz])
    pts.append(((rr,cc), p.xyz, p.surface))
np.save('holes.npy', np.array([[p[1][0], p[1][1], p[1][2]] for p in pts if p[2]=='object']))
