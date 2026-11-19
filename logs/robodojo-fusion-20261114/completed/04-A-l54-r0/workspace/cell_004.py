
import numpy as np
def try_locate(text, cam):
    try:
        o = robo.locate(text, camera=cam)
        return o
    except Exception as e:
        print("LOCERR", type(e).__name__, str(e)[:300])
        return None

for txt in ["blue rack", "the rack"]:
    o = try_locate(txt, 'cam_right_wrist')
    if o is not None:
        print(txt, "found", o.found, o.center_xyz, "top", o.top_z, "h", o.height_m, "L", o.length_m, "W", o.width_m,
              "axis", o.long_axis_deg, "views", o.views, "note", o.note, "score", o.score, "cent", o.centroid)
    if o is not None and o.found and o.views >= 2:
        break
