
for n in (2,3):
    t = robo.locate_candidate("segment_test_tube_4", n)
    print(n, t.center_xyz, "top", t.top_z, "h", t.height_m, "L", t.length_m, "W", t.width_m, "axis", t.long_axis_deg, "views", t.views, "note", t.note, "score", t.score)
