o = robo.locate("blue test tube rack with holes")
print('rack:', o.center_xyz, 'top', o.top_z, 'h', o.height_m, 'L', o.length_m, 'W', o.width_m,
      'long', o.long_axis_deg, 'views', o.views, 'score', o.score, 'note', o.note)
print('state right', robo.state().right.tcp_pos, robo.state().right.gripper)
