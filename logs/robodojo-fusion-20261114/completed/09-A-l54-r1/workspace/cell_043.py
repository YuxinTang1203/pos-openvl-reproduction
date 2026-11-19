import numpy as np
pc = robo.visual_project('frame_0054', 162, 352, 0.83)   # cap end of tube on plate
pt = robo.visual_project('frame_0054', 168, 388, 0.83)   # tip end
print("cap", pc['status'], pc['xyz_m'] if pc['status']=='ok' else pc['reason'])
print("tip", pt['status'], pt['xyz_m'] if pt['status']=='ok' else pt['reason'])
C=np.array(pc['xyz_m'][:2]); T=np.array(pt['xyz_m'][:2])
u=(T-C)/np.linalg.norm(T-C); L=np.linalg.norm(T-C)
gp=C+0.02*u
print("axis len", round(L,3), "grasp xy", gp.round(4))
m=robo.move_to([gp[0],gp[1],0.87],arm='right',gripper=robo.OPEN,pitch=60,yaw=81.5,path='line')
print("above:",m.reached,m.final_xyz,m.ik_error)
m2=robo.move_to([gp[0],gp[1],0.845],arm='right',gripper=robo.OPEN,pitch=60,yaw=81.5,path='line')
print("descend:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
g=robo.set_gripper('right',robo.CLOSE)
print("close:",g.holding,g.width_m,g.opening)
m3=robo.move_to([gp[0],gp[1],0.95],arm='right',gripper=robo.CLOSE,pitch=60,yaw=81.5,path='line')
print("lift:",m3.reached,m3.final_xyz,m3.blocked)
