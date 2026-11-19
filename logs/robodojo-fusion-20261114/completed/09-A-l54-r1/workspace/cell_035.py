# Lower so the tip goes ~2 cm into the hole (tip ~9 cm below the grasp point), then release.
m=robo.move_to([0.0697,0.0044,0.888],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='line')
print("lower:",m.reached,m.final_xyz,m.blocked,m.ik_error)
robo.show('cam_head')
