
m = robo.move_to([0.25, -0.15, 0.98], arm='right', gripper=robo.OPEN, pitch=90, yaw=-30, path='auto')
print("away:", m.reached, m.final_xyz, m.steps_used)
try:
    t = robo.locate("orange capped test tube lying on table", camera='cam_head')
    print("T2:", t.center_xyz, "top", t.top_z, "L", t.length_m, "W", t.width_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
except Exception as e:
    print("LOCERR", str(e)[:250])
robo.show('cam_head')
