no=0125
girl=02
#girl05_camera8_R_XNCG_ep0005_cam01_rd_lgt.0053.png.pfm
#girl02_R_camera8_R_XNCG_ep0001_cam01_rd_lgt.0125.png.pfm
dist=/media/sf_Shared_Data/dispnet/FusionPortal/data
left_dis_root=./detect_result/girl-crop-noscale-2nd
k=0
for ((cam=1;cam <= 8;cam++)) 
do
    #left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl$girl/camera${cam}_R # >= girl05
    left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl$girl/R/camera${cam}_R
    #img=XNCG_ep00${girl}_cam01_rd_lgt.${no}.png # >= girl 05
    img=XNCG_ep0001_cam01_rd_lgt.${no}.png # <= girl 04
    target=$dist/${k}
    #disp=girl${girl}_camera${cam}_R_XNCG_ep00${girl}_cam01_rd_lgt.${no}.png.pfm # >= girl 05
    disp=girl${girl}_R_camera${cam}_R_XNCG_ep0001_cam01_rd_lgt.${no}.png.pfm # <= girl 04
    mkdir $target
    cp $left_rgb_root/$img $target/0.png
    #./tools/pfm_viewer ${left_dis_root}/${disp} test.exr
    cp ${left_dis_root}/${disp} $target/0.pfm
    k=$(expr $k + 1)
done
