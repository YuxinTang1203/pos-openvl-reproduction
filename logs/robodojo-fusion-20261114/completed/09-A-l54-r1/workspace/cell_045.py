import numpy as np
o = robo.locate_candidate('segment_white_tube_with_orange_cap_5', 1)
print("tube1", o.found, o.center_xyz, "top", o.top_z, "L", o.length_m, "W", o.width_m, "long", o.long_axis_deg, "gy", o.grasp_yaw, o.views, o.note)
C = np.array(o.center_xyz)
ax = np.deg2rad(o.long_axis_deg); u = np.array([np.cos(ax), np.sin(ax)])
gp = C[:2] - 0.02*u   # 2 cm from cap end (cap is at -u end per image: cap left)
gz = o.top_z - 0.012
print("grasp", gp.round(4), "gz", round(gz,4))
m=robo.move_to([gp[0],gp[1],0.86],arm='right',gripper=robo.OPEN,pitch=60,yaw=o.grasp_yaw,path='line')
print("above:",m.reached,m.final_xyz,m.ik_error)
m2=robo.move_to([gp[0],gp[1],gz],arm='right',gripper=robo.OPEN,pitch=60,yaw=o.grasp_yaw,path='line')
print("descend:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
g=robo.set_gripper('right',robo.CLOSE)
print("close:",g.holding,g.width_m,g.opening)
