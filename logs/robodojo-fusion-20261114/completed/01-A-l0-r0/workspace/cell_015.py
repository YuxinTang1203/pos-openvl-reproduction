import numpy as np
# Plate top is actually ~0.799 (wrist back_project of plate gave 0.7991), not 0.7871.
# Re-project the visible hole openings from the wrist frame onto z=0.799.
for r,c,t in [(452,596,"holeA"),(466,540,"holeB"),(476,470,"holeC"),(430,322,"tip")]:
    p = robo.visual_project('frame_0017', r, c, 0.799)
    print(t, p["status"], np.round(p["xyz_m"],4) if p["status"]=="ok" else p["reason"])
# also head-camera lattice at 0.799
for r,c,t in [(177.0,286.5,"H_far_left"),(175.5,302.5,"H_far_2"),(189.7,287.1,"H_near_left"),(186.7,320.4,"H_near_3")]:
    p = robo.visual_project('frame_0010', int(r), int(c), 0.799)
    print(t, p["status"], np.round(p["xyz_m"],4) if p["status"]=="ok" else p["reason"])
