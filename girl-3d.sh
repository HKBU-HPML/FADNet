# girl
directory=ep0010
no=0015
cam=camera2_R
left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/ep0010
left_dis_root=./detect_results/virtual
img=XNCG_${directory}_cam01_rd_lgt.${no}.png
disp=${cam}/XNCG_${directory}_cam01_rd_lgt.${no}.png.pfm
inv_baseline=0.05
focal=1050
maxdisp=480
mindisp=0
jview ${left_rgb_root}/${cam}/${img}
jview ${left_dis_root}/${disp}
./tools/pfm_viewer ${left_dis_root}/${disp} test.exr
./tools/DisparityTo3D test.exr test.obj ${left_rgb_root}/${cam}/${img} ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
cp test.obj /media/sf_Shared_Data/dispnet/${img}.obj

