b = "segment_plastic_centrifuge_tube_with_o"
for n in (1,2,3):
    o = robo.locate_candidate(b, n)
    print(n, "found", o.found, "center", o.center_xyz, "top_z", o.top_z, "h", o.height_m,
          "L", o.length_m, "W", o.width_m, "long_axis", o.long_axis_deg, "grasp_yaw", o.grasp_yaw,
          "views", o.views, "score", o.score, "note", o.note)
for p in ["the blue test tube rack", "blue plastic rack with round holes", "the rack", "laboratory tube rack"]:
    r = robo.locate(p)
    print(p, "->", r.found, r.center_xyz, r.top_z, r.height_m, r.length_m, r.width_m, r.long_axis_deg, r.views, r.score, r.note)
