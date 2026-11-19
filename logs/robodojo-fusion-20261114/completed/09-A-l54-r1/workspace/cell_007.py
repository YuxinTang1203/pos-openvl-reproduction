import numpy as np
# Grasp point: 2 cm inboard from tube-1's cap end, along its axis
C = np.array([-0.305,-0.074]); B = np.array([-0.289,-0.200])
u = (B-C)/np.linalg.norm(B-C)
gp = C + 0.02*u
print("grasp xy", gp)
try:
    m1 = robo.move_to([gp[0], gp[1], 0.87], arm='left', gripper=robo.OPEN, pitch=90, yaw=7.5, path='line')
    print("above:", m1.reached, m1.final_xyz, m1.ik_error, m1.note, m1.steps_used)
except Exception as e:
    print("ERR", e)
try:
    o = robo.locate("white tube with orange cap", camera='cam_left_wrist')
    print("wrist locate:", o.found, o.center_xyz, o.top_z, o.height_m, o.length_m, o.width_m, o.long_axis_deg, o.grasp_yaw, o.views, o.score, o.note)
except Exception as e:
    print("LOCERR", type(e).__name__, e)
