print(robo.task)
for n in (1,2,3):
    o = robo.locate_candidate("segment_test_tube_with_orange_cap", n)
    print(n, o.center_xyz, "top", o.top_z, "h", o.height_m, "L", o.length_m, "W", o.width_m,
          "long", o.long_axis_deg, "gy", o.grasp_yaw, "views", o.views, "note", o.note, "score", o.score)
r = robo.locate_candidate("segment_test_tube_rack", 1)
print("rack", r.center_xyz, "top", r.top_z, "h", r.height_m, "L", r.length_m, "W", r.width_m,
      "long", r.long_axis_deg, "views", r.views, "note", r.note, "score", r.score)
