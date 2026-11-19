# Wrist view shows tube tip resting ON the blue plate with a hole opening just to the lower-right.
# Measure the hole centre and the tip contact point from the wrist camera.
for (r,c,tag) in [(455,600,"hole_right"),(470,545,"hole_right2"),(430,320,"tip_contact"),(470,640,"hole_far_corner")]:
    try:
        p=robo.back_project(r,c,'cam_left_wrist')
        print(tag, p.surface, np.round(p.xyz,4))
    except Exception as e:
        print(tag,"err",e)
