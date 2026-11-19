
def safe_locate(text, cam='cam_head'):
    try:
        o = robo.locate(text, camera=cam)
        return o
    except Exception as e:
        print("LOCERR", type(e).__name__, str(e)[:400])
        return None

st = robo.state()
print("state", st.left.tcp_pos, st.right.tcp_pos, st.left.gripper, st.right.gripper)

# put right gripper up-and-tilted above the rack so the wrist camera looks down on it
m = robo.move_to([-0.004, 0.057, 0.95], arm='right', gripper=robo.OPEN, pitch=60, yaw=0, path='auto')
print("move", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.steps_used)

r = safe_locate("blue and white test tube rack", 'cam_right_wrist')
if r: print("RACK wrist", r.center_xyz, r.top_z, r.height_m, r.length_m, r.width_m, r.long_axis_deg, r.views, r.note, r.score)
r2 = safe_locate("blue and white test tube rack", 'cam_head')
if r2: print("RACK head", r2.center_xyz, r2.top_z, r2.height_m, r2.length_m, r2.width_m, r2.long_axis_deg, r2.views, r2.note, r2.score)
