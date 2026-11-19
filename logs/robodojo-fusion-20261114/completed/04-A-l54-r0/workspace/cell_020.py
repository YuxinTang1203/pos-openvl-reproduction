
for n in (1,2,3):
    t = robo.locate_candidate("segment_test_tube_3", n)
    print(n, t.center_xyz, "top", t.top_z, "L", t.length_m, "W", t.width_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
