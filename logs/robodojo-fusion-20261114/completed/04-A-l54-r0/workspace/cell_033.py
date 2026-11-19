
r = robo.locate("blue and white test tube rack", camera='cam_head')
print("RACK:", r.center_xyz, "top", r.top_z, "L", r.length_m, "W", r.width_m, "axis", r.long_axis_deg, "views", r.views, "note", r.note, "score", r.score)
try:
    t = robo.locate_candidate("segment_test_tube_lying_on_table", 2)
    print("T:", t.center_xyz, "top", t.top_z, "L", t.length_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
except Exception as e:
    print("ERR", str(e)[:200])
