img=img00003

python fusion.py ${img}
jview ./test.pfm 
pfm_viewer ./test.pfm ./real_detect_result_s1.5/predict_${img}.exr
DisparityTo3D ./real_detect_result_s1.5/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
cd ./real_obj
meshlab ${img}.obj
cd ..

# img=img00000
# 
# # pfm_viewer ./real_detect_result_s1.4/predict_${img}.pfm ./real_detect_result_s1.4/predict_${img}.exr
# pfm_viewer /media/sf_Shared_Data/gpuhome/pytorch-dispnet/data/dispnet/real_release/sgm_disp/left/${img}.pfm ./real_detect_result_s1.4/predict_${img}.exr
# DisparityTo3D ./real_detect_result_s1.4/predict_${img}.exr ./${img}.obj ./data/dispnet/real_release/frames_cleanpass/left/${img}.bmp
# meshlab ${img}.obj
