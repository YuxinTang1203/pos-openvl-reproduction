import numpy as np
holes_px = [(171.4,350.5),(172.9,334.5),(174.3,318.6),(175.5,302.5),(177.0,286.5),
            (183.8,353.1),(185.2,337.0),(186.7,320.4),(188.0,304.0),(189.7,287.1)]
HL=[]
for (r,c) in holes_px:
    p = robo.visual_project('frame_0014', int(r), int(c), 0.799)
    HL.append(np.array(p["xyz_m"][:2]) if p["status"]=="ok" else None)
    print((r,c), p["status"], None if p["status"]!="ok" else np.round(p["xyz_m"],4))
tip = np.array([-0.0366, 0.0877])
print("tip", tip)
d = [(round(float(np.linalg.norm(h-tip)),4), np.round(h-tip,4)) for h in HL if h is not None]
for i,(dd,v) in enumerate(d): print(i, dd, v)
