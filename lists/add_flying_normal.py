
with open("FlyingThings3D_release_TEST.list") as f:
    content = f.readlines()

with open("FlyingThings3D_release_TEST_norm.list", "w") as f:
    for line in content:
        items = line.split()
        norm = items[-1].replace("disparity", "normal").replace("pfm", "exr")
        items.append(norm)
        new_line = "\t".join(items)
        f.write("%s\n" % new_line)

