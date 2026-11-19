def safe_locate(txt, cam='cam_head'):
    try:
        o = robo.locate(txt, camera=cam)
        print(txt, '->', 'found', o.found, 'center', o.center_xyz, 'top', o.top_z, 'h', o.height_m,
              'L', o.length_m, 'W', o.width_m, 'long', o.long_axis_deg, 'gyaw', o.grasp_yaw,
              'views', o.views, 'score', o.score, 'note', o.note)
        return o
    except Exception as e:
        print(txt, 'EXC', type(e).__name__, str(e)[:300])
        return None

t1 = safe_locate("the white test tube lying on the left")
t2 = safe_locate("the white test tube lying in the middle")
t3 = safe_locate("the white test tube lying on the right")
