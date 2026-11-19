
m = robo.move_to([0.25, -0.20, 0.97], arm='right', gripper=robo.OPEN, pitch=60, yaw=-30, path='auto')
print("clear:", m.reached, m.final_xyz, m.steps_used)
r = robo.locate("blue and white test tube rack", camera='cam_head')
print("RACK:", r.center_xyz, "top", r.top_z, "L", r.length_m, "W", r.width_m, "axis", r.long_axis_deg, "views", r.views, "note", r.note, "score", r.score)
try:
    t = robo.locate("test tube lying on the table", camera='cam_head')
    print("T:", t.center_xyz, "top", t.top_z, "L", t.length_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
except Exception as e:
    print("ERR", str(e)[:300])
