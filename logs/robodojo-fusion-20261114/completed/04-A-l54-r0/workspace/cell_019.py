
r = robo.locate("blue and white test tube rack", camera='cam_head')
print("RACK:", r.center_xyz, "top", r.top_z, "L", r.length_m, "W", r.width_m, "axis", r.long_axis_deg, "views", r.views, "note", r.note, "score", r.score)
try:
    o = robo.segment("test tube", camera='cam_head')
except Exception as e:
    print("segerr", str(e)[:200])
for n in range(1,6):
    try:
        t = robo.locate_candidate("segment_test_tube_2", n) if n<=2 else None
    except Exception as e:
        t=None
# simpler: grounding again fresh
import numpy as np
