import numpy as np
img = robo.image('cam_head').astype(int)
R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]
blue = (B > R + 25) & (B > 90)
ys, xs = np.nonzero(blue)
print('blue px', len(xs), 'col range', xs.min(), xs.max(), 'row range', ys.min(), ys.max())

# dark holes inside the blue plate area
sub_r0, sub_r1 = ys.min(), ys.max()
sub_c0, sub_c1 = xs.min(), xs.max()
lum = img.mean(axis=2)
dark = (lum < 90)
region = np.zeros_like(dark); region[sub_r0:sub_r1+1, sub_c0:sub_c1+1] = True
holes = dark & region
print('dark px in rack bbox', holes.sum())

try:
    from scipy import ndimage
    lab, n = ndimage.label(holes)
    print('components', n)
    sizes = ndimage.sum(holes, lab, range(1, n+1))
    coms = ndimage.center_of_mass(holes, lab, range(1, n+1))
    big = [(int(s), (int(round(r)), int(round(c)))) for s, (r, c) in zip(sizes, coms) if s >= 30]
    big.sort(key=lambda t: (t[1][0], t[1][1]))
    for s, rc in big: print('hole', s, rc)
except Exception as e:
    print('scipy fail', e)
