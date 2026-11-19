
t = robo.locate_candidate("segment_test_tube_2", 2)
print("rack-tube:", t.center_xyz, "top", t.top_z, "h", t.height_m, "L", t.length_m, "W", t.width_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
t1 = robo.locate_candidate("segment_test_tube_2", 1)
print("left-tube:", t1.center_xyz, "top", t1.top_z, "h", t1.height_m, "L", t1.length_m, "axis", t1.long_axis_deg, "gy", t1.grasp_yaw, "views", t1.views, "note", t1.note)
