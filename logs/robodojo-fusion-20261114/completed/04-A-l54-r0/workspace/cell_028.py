
import numpy as np
cx, cy = 0.0534, -0.0809
m = robo.move_to([cx, cy, 0.87], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='auto')
print("above:", m.reached, m.final_xyz, m.ik_error, m.steps_used)
try:
    t = robo.locate("white test tube with orange cap", camera='cam_right_wrist')
    print("refine:", t.center_xyz, "L", t.length_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
    if t.center_xyz and t.views>=2: cx, cy = t.center_xyz[0], t.center_xyz[1]
except Exception as e:
    print("LOCERR", str(e)[:200])
m2 = robo.move_to([cx, cy, 0.788], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.blocked, m2.ik_error, m2.steps_used)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.opening, g.width_m)
m3 = robo.move_to([cx, cy, 0.87], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-58.6, path='line')
print("lift:", m3.reached, m3.final_xyz, m3.steps_used)
m4 = robo.move_to([cx, cy, 0.87], arm='right', gripper=robo.CLOSE, pitch=0, yaw=-58.6, roll=0, path='auto')
print("rotate0:", m4.reached, m4.final_xyz, m4.pitch, m4.yaw, m4.roll, m4.ik_error, m4.note, m4.steps_used)
robo.show('cam_head')
