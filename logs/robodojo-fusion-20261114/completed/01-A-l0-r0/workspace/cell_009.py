import numpy as np
holes_px = [(171.4,350.5),(172.9,334.5),(174.3,318.6),(175.5,302.5),(177.0,286.5),
            (183.8,353.1),(185.2,337.0),(186.7,320.4),(188.0,304.0),(189.7,287.1)]
out=[]
for (r,c) in holes_px:
    p = robo.visual_project('frame_0010', int(round(r)), int(round(c)), 0.7871)
    if p["status"]=="ok":
        out.append(((r,c), tuple(np.round(p["xyz_m"],4))))
        print((r,c), np.round(p["xyz_m"],4))
    else:
        print((r,c), "fail", p["reason"])
