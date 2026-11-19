b="segment_plastic_centrifuge_tube_with_o_2"
for n in (1,2):
    o = robo.locate_candidate(b,n)
    print(n, o.found, None if o.center_xyz is None else tuple(round(v,4) for v in o.center_xyz),
          "top", None if o.top_z is None else round(o.top_z,4), "L", o.length_m, "W", o.width_m,
          "long", o.long_axis_deg, "gy", o.grasp_yaw, "views", o.views, "sc", round(o.score or 0,3), o.note)
r = robo.locate("blue plastic rack with round holes")
print("rack", r.found, tuple(round(v,4) for v in r.center_xyz) if r.center_xyz else None, "top", r.top_z, "h", r.height_m, r.length_m, r.width_m, r.long_axis_deg, r.views, r.note)
im = robo.image('cam_head'); display(im[140:220, 270:380])
