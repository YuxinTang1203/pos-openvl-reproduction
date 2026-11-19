import numpy as np
# project wrist-image pixels onto the plate plane z=0.7871: hole openings and the tube tip contact
tests = [(452,596,"holeA"),(466,540,"holeB"),(476,470,"holeC"),(430,322,"tip"),(415,300,"tube_mid"),(470,610,"holeA2")]
for r,c,t in tests:
    p = robo.visual_project('frame_0017', r, c, 0.7871)
    print(t, p["status"], np.round(p["xyz_m"],4) if p["status"]=="ok" else p.get("reason"))
