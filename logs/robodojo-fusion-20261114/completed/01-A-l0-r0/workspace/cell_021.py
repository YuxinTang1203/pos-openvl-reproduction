import numpy as np
# Tube 1 fell over; it now lies behind-left of the rack. Ground all tubes in the CURRENT head frame.
segs = []
for n in range(1,4):
    pass
# use locate with generic prompt to get all three candidates via routing
try:
    o = robo.locate("test tube with orange cap lying on table", 'cam_head')
    print("loc1", o.center_xyz, o.top_z, o.length_m, o.width_m, o.long_axis_deg, o.grasp_yaw, o.views, o.score, o.note)
except Exception as e:
    print("err", e)
