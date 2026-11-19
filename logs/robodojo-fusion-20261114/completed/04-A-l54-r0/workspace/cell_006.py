
import numpy as np
# Tube-2 last known: center (0.1955,-0.1118), axis 51.5deg, cap at far end (+u)
u = np.array([np.cos(np.radians(51.5)), np.sin(np.radians(51.5))])
gxy = np.array([0.1955,-0.1118]) + 0.04*u   # grasp 4cm toward cap end
print("grasp xy", gxy)

m = robo.move_to([float(gxy[0]), float(gxy[1]), 0.87], arm='right', gripper=robo.OPEN, pitch=90, yaw=-38.5, path='auto')
print("above:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.note, m.ik_error, m.steps_used)

try:
    t = robo.locate("white test tube with orange cap", camera='cam_right_wrist')
    print("T2 above:", t.center_xyz, "top", t.top_z, "h", t.height_m, "L", t.length_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score)
except Exception as e:
    print("LOCERR", str(e)[:300])
