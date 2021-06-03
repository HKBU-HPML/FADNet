with open("kitti2015_test.list", "w") as f:
    for i in range(200):
        leftn = "kitti2015/testing/image_2/000%03d_10.png" % i
        rightn = "kitti2015/testing/image_3/000%03d_10.png" % i
        leftd = "kitti2015/testing/disp_occ_0/000%03d_10.png" % i
        f.write("%s %s %s\n" % (leftn, rightn, leftd))
