
with open("SceneFlow.list") as f:
    content = f.readlines()

with open("SceneFlow_norm.list", "w") as f:
    for line in content:
        items = line.split()
        norm = items[-1].replace("disparity", "normal").replace("pfm", "exr")
        if "15mm" in norm:
            continue
        items.append(norm)
        new_line = "\t".join(items)
        f.write("%s\n" % new_line)

