
r = robo.locate("test tube lying on the table", camera='cam_head')
print("T:", r.center_xyz, "top", r.top_z, "L", r.length_m, "axis", r.long_axis_deg, "gy", r.grasp_yaw, "views", r.views, "note", r.note, "score", r.score, "cent", r.centroid)
