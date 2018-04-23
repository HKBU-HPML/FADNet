#img=img00055
#
## nn 3d model
#pfm_viewer ./real_detect_result_s1.5/predict_real_release_frames_cleanpass_${img}.pfm 
#pfm_viewer ./real_detect_result_s1.5/predict_real_release_frames_cleanpass_${img}.pfm ./real_detect_result_s1.5/predict_${img}.exr
## pfm_viewer ./real_detect_result_s1.5/predict_${img}.pfm 
## pfm_viewer ./real_detect_result_s1.5/predict_${img}.pfm ./real_detect_result_s1.5/predict_${img}.exr
#DisparityTo3D ./real_detect_result_s1.5/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
#cd ./real_obj
#meshlab ${img}.obj
#cd ..

# # sgm 3d model
# pfm_viewer ./data_local/dispnet/real_release/sgm_disp/left/${img}.pfm ./real_detect_result_sgm/predict_${img}.exr
# DisparityTo3D ./real_detect_result_sgm/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
# cd ./real_obj
# meshlab ${img}.obj
# cd ..


# flyingthing3D
f1=A
f2=0015
f3=0007
left_rgb_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data_local/FlyingThings3D_release/frames_cleanpass/
#left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data_local/clean_dispnet/FlyingThings3D_release/clean_disparity/
#left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/data_local/FlyingThings3D_release/disparity/
left_dis_root=/home/gpgpu-sim/gpuhome/pytorch-dispnet/flying_detect_cleandata
img=TEST/${f1}/${f2}/left/${f3}
inv_baseline=0.01
focal=1050
maxdisp=480
mindisp=10
jview ${left_rgb_root}/${img}.png
#jview ${left_dis_root}/${img}.pfm
jview ${left_dis_root}/predict_${f1}_${f2}_${f3}.pfm
#pfm_viewer ${left_dis_root}/${img}.pfm test.exr
pfm_viewer ${left_dis_root}/predict_${f1}_${f2}_${f3}.pfm test.exr
DisparityTo3D test.exr test.obj ${left_rgb_root}/${img}.png ${inv_baseline} ${focal} ${maxdisp} ${mindisp}
meshlab test.obj
