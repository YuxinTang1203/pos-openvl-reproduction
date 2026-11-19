
# Probe the rack from the right wrist camera (currently parked above it, tilted 55 deg).
pts = {
 "blue_plate_a": (300, 340),
 "blue_plate_b": (355, 360),
 "hole15": (349, 321),
 "hole20": (364, 396),
 "hole25": (379, 471),
 "hole10": (335, 249),
}
for k,(r,c) in pts.items():
    try:
        p = robo.back_project(r, c, camera='cam_right_wrist')
        print(k, p.surface, [round(v,4) for v in p.xyz], round(p.depth_m,3))
    except Exception as e:
        print(k, "ERR", str(e)[:120])
