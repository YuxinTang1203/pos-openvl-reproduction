b = "segment_the_white_test_tube_lying_on_t_2"
for n in (1,2,3):
    o = robo.locate_candidate(b, n)
    print(n, 'found', o.found, 'center', o.center_xyz, 'top', o.top_z, 'h', o.height_m,
          'L', o.length_m, 'W', o.width_m, 'long', o.long_axis_deg, 'gyaw', o.grasp_yaw,
          'views', o.views, 'score', o.score, 'centroid', o.centroid, 'note', o.note)
