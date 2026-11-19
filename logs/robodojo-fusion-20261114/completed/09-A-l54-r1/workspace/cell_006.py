import numpy as np
# 1) zoom tube-1 (candidate 1 of earlier tube grounding) to see cap end
# 2) project big-hole pixel centres onto plate plane z=0.8231
holes_px = [(199.6,351.8),(197.5,334.9),(195.1,318.1),(193.0,301.6),(191.0,285.1),
            (186.4,353.5),(184.4,336.9),(182.5,320.9),(180.5,304.6),(178.6,288.5)]
out=[]
for r,c in holes_px:
    p = robo.visual_project('frame_0013', int(r), int(c), 0.8231)
    if p['status']=='ok':
        out.append(((r,c), tuple(round(v,4) for v in p['xyz_m'])))
    else:
        out.append(((r,c), p['reason']))
for o in out: print(o)
