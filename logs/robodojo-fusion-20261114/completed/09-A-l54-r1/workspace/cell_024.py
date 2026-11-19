import numpy as np
b="segment_plastic_centrifuge_tube_with_o"
for n in (1,2,3):
    o = robo.locate_candidate(b,n)
    print(n, o.found, None if o.center_xyz is None else tuple(round(v,4) for v in o.center_xyz),
          None if o.top_z is None else round(o.top_z,4), o.length_m, o.width_m, o.long_axis_deg, o.grasp_yaw, o.views, round(o.score or 0,3), o.note)
r = robo.locate("blue plastic rack with round holes")
print("rack", r.found, tuple(round(v,4) for v in r.center_xyz) if r.center_xyz else None, r.top_z, r.length_m, r.width_m, r.long_axis_deg, r.views, r.note)
st=robo.state(); print("step", st.step, "trunc", st.truncated)
