o = robo.locate_candidate("segment_test_tube_lying_flat_on_the_ta", 1)
print('tube1:', o.center_xyz, 'top', o.top_z, 'L', o.length_m, 'W', o.width_m, 'gyaw', o.grasp_yaw,
      'views', o.views, 'score', o.score, 'note', o.note)
