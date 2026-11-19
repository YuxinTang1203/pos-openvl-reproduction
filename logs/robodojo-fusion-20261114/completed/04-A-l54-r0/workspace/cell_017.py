
# Re-grasp tube-2 (lying on rack plate) at its MIDDLE, approach pitch=60 yaw=53.8 (along its axis)
tx, ty = 0.0009, 0.0519
m = robo.move_to([tx, ty, 0.90], arm='right', gripper=robo.OPEN, pitch=60, yaw=53.8, path='auto')
print("above:", m.reached, m.final_xyz, m.pitch, m.yaw, m.ik_error, m.note, m.steps_used)
try:
    t = robo.locate("test tube lying on rack", camera='cam_right_wrist')
    print("T2:", t.center_xyz, t.top_z, t.length_m, t.long_axis_deg, t.views, t.note, t.score)
except Exception as e:
    print("LOCERR", str(e)[:200])
m2 = robo.move_to([tx, ty, 0.80], arm='right', gripper=robo.OPEN, pitch=60, yaw=53.8, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.blocked, m2.ik_error, m2.note, m2.steps_used)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.opening, g.width_m)
