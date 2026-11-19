s = robo.state(); print("step", s.step, "truncated", s.truncated)
b="segment_white_tube_lying_on_table_with"
for n in (1,2,3):
    o = robo.locate_candidate(b,n)
    print(n, o.found, None if o.center_xyz is None else tuple(round(v,4) for v in o.center_xyz),
          "top", None if o.top_z is None else round(o.top_z,4), "L", o.length_m, "W", o.width_m,
          "long", o.long_axis_deg, "gy", o.grasp_yaw, "v", o.views, round(o.score or 0,3), o.note)
holes = [(178.5,288.5),(180.5,304.6),(182.5,320.8),(184.3,337.1),(186.5,353.5),
         (191.1,285.1),(193.2,301.4),(195.1,318.1),(197.4,334.8),(199.6,351.9)]
for r,c in holes:
    p = robo.visual_project('frame_0051', int(r), int(c), 0.815)
    print((r,c), p['status'], tuple(round(v,4) for v in p['xyz_m']) if p['status']=='ok' else p['reason'])
