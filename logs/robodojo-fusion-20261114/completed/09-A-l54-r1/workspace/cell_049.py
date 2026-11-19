g=robo.set_gripper('right', robo.OPEN)
print("open:", g.opening, g.steps_used)
s=robo.state(); x,y,z=s.right.tcp_pos
m=robo.move_to([x,y,0.99],arm='right',gripper=robo.OPEN,pitch=0,yaw=-38.4,roll=0,path='line')
print("withdraw:",m.reached,m.final_xyz,m.steps_used,m.blocked)
robo.show('cam_head')
