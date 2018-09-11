no=0063
girl=02
#girl02_R_camera8_R_XNCG_ep0001_cam01_rd_lgt.0125.png.pfm
#girl02_R_camera7_R_XNCG_ep0001_cam01_rd_lgt.0095.png.pfm
#girl02_R_camera5_R_XNCG_ep0001_cam01_rd_lgt.0001.png.pfm
dist=/media/sf_Shared_Data/dispnet/FusionPortal/data
#left_dis_root=./detect_result/girl-crop-noscale-3nd-epe0.152
#left_dis_root=./detect_result/girl-crop-noscale-2nd-epe0.182
left_dis_root=./detect_results/virtual
k=0
for ((cam=1;cam <= 8;cam++)) 
do
    #left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl$girl/camera${cam}_R # >= girl05
    #left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl$girl/R/camera${cam}_R
    left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/ep0010/camera${cam}_R
    #img=XNCG_ep00${girl}_cam01_rd_lgt.${no}.png # >= girl 05
    #img=XNCG_ep0001_cam01_rd_lgt.${no}.png # <= girl 04
    img=XNCG_ep0010_cam01_rd_lgt.${no}.png 
    target=$dist/${k}
    #disp=girl${girl}_camera${cam}_R_XNCG_ep00${girl}_cam01_rd_lgt.${no}.png.pfm # >= girl 05
    #disp=girl${girl}_R_camera${cam}_R_XNCG_ep0001_cam01_rd_lgt.${no}.png.pfm # <= girl 04
    disp=camera${cam}_R/${img}.pfm
    mkdir $target
    cp $left_rgb_root/$img $target/0.png
    #./tools/pfm_viewer ${left_dis_root}/${disp} test.exr
    cp ${left_dis_root}/${disp} $target/0.pfm
    k=$(expr $k + 1)
done
python tools/process_cam_data.py
