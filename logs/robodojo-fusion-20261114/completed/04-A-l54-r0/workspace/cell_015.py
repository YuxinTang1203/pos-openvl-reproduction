
# The released tube lies tilted on the rack plate. Re-grasp it: park above it, look from wrist cam.
m = robo.move_to([0.02, 0.01, 0.93], arm='right', gripper=robo.OPEN, pitch=90, yaw=-40, path='auto')
print("above tube:", m.reached, m.final_xyz, m.steps_used)
try:
    t = robo.locate("white test tube with orange cap", camera='cam_right_wrist')
    print("T:", t.center_xyz, "top", t.top_z, "h", t.height_m, "L", t.length_m, "axis", t.long_axis_deg, "gy", t.grasp_yaw, "views", t.views, "note", t.note, "score", t.score, "cent", t.centroid)
except Exception as e:
    print("LOCERR", str(e)[:300])
robo.show('cam_right_wrist')
