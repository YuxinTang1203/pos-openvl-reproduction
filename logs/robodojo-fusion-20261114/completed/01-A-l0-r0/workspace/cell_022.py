b='segment_test_tube_with_orange_cap_lyin'
for n in (2,3,4,5):
    o = robo.locate_candidate(b, n)
    print(n, o.center_xyz, "top", o.top_z, "L", o.length_m, "W", o.width_m, "long", o.long_axis_deg,
          "gy", o.grasp_yaw, "views", o.views, "score", round(o.score,3), "note", o.note)
