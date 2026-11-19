
try:
    t1 = robo.locate("test tube standing upright near the left gripper", camera='cam_head')
    print("T1:", t1.center_xyz, "top", t1.top_z, "h", t1.height_m, "L", t1.length_m, "axis", t1.long_axis_deg, "views", t1.views, "note", t1.note, "score", t1.score)
except Exception as e:
    print("LOCERR", str(e)[:300])
r = robo.locate("blue and white test tube rack", camera='cam_head')
print("RACK:", r.center_xyz, "top", r.top_z, "L", r.length_m, "W", r.width_m, "axis", r.long_axis_deg, "views", r.views, "note", r.note, "score", r.score)
