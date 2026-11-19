try:
    o = robo.locate("test tube lying flat on the table")
    print('lying tube:', o.center_xyz, 'top', o.top_z, 'gyaw', o.grasp_yaw, 'L', o.length_m, 'W', o.width_m,
          'views', o.views, 'score', o.score, 'note', o.note)
except Exception as e:
    print('EXC', str(e)[:400])
