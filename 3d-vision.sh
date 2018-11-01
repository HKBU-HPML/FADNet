img=img00077
inv_baseline=0.0038
focal=1800
maxdisp=200
mindisp=1
#real_release_frames_cleanpass_left_img00008.bmp.pfm
disp_root=./detect_results/real-monodepth
#left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/real_release/frames_cleanpass/left
left_rgb_root=/media/external/data/real_release/frames_cleanpass/left
# nn 3d model
rgb=${left_rgb_root}/${img}.bmp
disp=${disp_root}/real_release_frames_cleanpass_left_${img}.bmp.pfm
#jview $rgb
#jview $disp
#disp_root=./detect_result/domain_adaptation
#disp_root=./detect_result/flyingthings_model
#disp_root=./detect_results/flyingthings_model/disp3
#disp_root=./detect_result/cropfrom1024model_real_detect_cleandata_scale1024
#left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/real_release/frames_cleanpass/left
#left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/custom/left
rgb=${left_rgb_root}/${img}.png
#disp=${disp_root}/custom_left_${img}.png.pfm
# nn 3d model
#rgb=${left_rgb_root}/${img}.bmp
#disp=${disp_root}/real_release_frames_cleanpass_left_${img}.bmp.pfm
disp=${disp_root}/real_release_frames_cleanpass_left_${img}.pfm
jview $rgb
jview $disp
./tools/pfm_viewer $disp  predict_${img}.exr
./tools/DisparityTo3D predict_${img}.exr ${img}.obj $rgb ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
cp ${img}.obj /media/sf_Shared_Data/dispnet/

# # sgm 3d model
# pfm_viewer ./data_local/dispnet/real_release/sgm_disp/left/${img}.pfm ./real_detect_result_sgm/predict_${img}.exr
# DisparityTo3D ./real_detect_result_sgm/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
# cd ./real_obj
# meshlab ${img}.obj
# cd ..


# # flyingthing3D
# f1=A
# f2=0027
# f3=0015
# #left_rgb_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data/FlyingThings3D_release/frames_cleanpass/
# left_rgb_root=/media/gpuhomedataset/FlyingThings3D_release/frames_cleanpass/
# #left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data_local/clean_dispnet/FlyingThings3D_release/clean_disparity/
# #left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data_local/FlyingThings3D_release/disparity/
# left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/detect_results/flying_detect_cleandata
# #left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/flying_detect_result
# img=TEST/${f1}/${f2}/left/${f3}
# inv_baseline=0.0038
# focal=500
# maxdisp=800
# mindisp=10
# jview ${left_rgb_root}/${img}.png
# #jview ${left_dis_root}/${img}.pfm
# jview ${left_dis_root}/predict_${f1}_${f2}_${f3}.pfm
# #pfm_viewer ${left_dis_root}/${img}.pfm test.exr
# ./tools/pfm_viewer ${left_dis_root}/predict_${f1}_${f2}_${f3}.pfm test.exr
# ./tools/DisparityTo3D test.exr test.obj ${left_rgb_root}/${img}.png ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
# meshlab test.obj

# girl
#no=0226
#left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl
##left_dis_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl
#left_dis_root=./girl_detect_cleandata
#img=girl_camera1_Rcamera1_R.${no}.png
##disp=girl_camera1_Rcamera1_R.Z.${no}.pfm
#disp=predict_girl_camera1_Rcamera1_R.${no}.png.pfm
#inv_baseline=0.05
#focal=1050
#maxdisp=480
#mindisp=0
#jview ${left_rgb_root}/${img}
#jview ${left_dis_root}/${disp}
#./tools/pfm_viewer ${left_dis_root}/${disp} test.exr
#./tools/DisparityTo3D test.exr test.obj ${left_rgb_root}/${img} ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
#cp test.obj /media/sf_Shared_Data/dispnet/
#meshlab test.obj


# girl02
#girl05_camera8_R_XNCG_ep0005_cam01_rd_lgt.0053.png.pfm
#no=0053
#cam=8
#girl=05
#left_rgb_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl$girl/camera${cam}_R
#img=XNCG_ep00${girl}_cam01_rd_lgt.${no}.png
##left_dis_root=/media/sf_Shared_Data/gpuhomedataset/dispnet/virtual/girl02/R_Z/camera${cam}_R_Z/Z_color
##disp=XNCG_ep0001_cam01_rd_lgt_Z.Z.${no}.pfm
#
##girl05_camera2_R_XNCG_ep0005_cam01_rd_lgt.0009.png.pfm
#left_dis_root=./detect_result/girl-crop-noscale-2nd
##disp=girl05_R_camera${cam}_R_XNCG_ep0005_cam01_rd_lgt.${no}.png.pfm
#disp=girl${girl}_camera${cam}_R_XNCG_ep00${girl}_cam01_rd_lgt.${no}.png.pfm
#inv_baseline=0.05
#focal=1050
#maxdisp=480
#mindisp=0
#jview ${left_rgb_root}/${img}
#jview ${left_dis_root}/${disp}
#./tools/pfm_viewer ${left_dis_root}/${disp} test.exr
#./tools/DisparityTo3D test.exr test.obj ${left_rgb_root}/${img} ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
##cp test.obj /media/sf_Shared_Data/dispnet/test2.obj
#cp test.obj /media/sf_Shared_Data/dispnet/

