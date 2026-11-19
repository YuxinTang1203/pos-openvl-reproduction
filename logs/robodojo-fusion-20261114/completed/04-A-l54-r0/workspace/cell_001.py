
rack = robo.locate_candidate("segment_test_tube_rack", 1)
print("RACK", rack.center_xyz, rack.top_z, rack.height_m, rack.length_m, rack.width_m, rack.long_axis_deg, rack.grasp_yaw, rack.views, rack.note)
for n in (1,2,3):
    t = robo.locate_candidate("segment_test_tube", n)
    print("TUBE", n, t.center_xyz, "top", t.top_z, "h", t.height_m, "L", t.length_m, "W", t.width_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score, "cent", t.centroid)
