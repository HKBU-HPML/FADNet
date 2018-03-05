img=img00065

pfm_viewer ./real_detect_result_s1.4/predict_${img}.pfm ./real_detect_result_s1.4/predict_${img}.exr
DisparityTo3D ./real_detect_result_s1.4/predict_${img}.exr ./${img}.obj ./data/dispnet/real_release/frames_cleanpass/left/${img}.bmp
meshlab ${img}.obj

# img=img00000
# 
# # pfm_viewer ./real_detect_result_s1.4/predict_${img}.pfm ./real_detect_result_s1.4/predict_${img}.exr
# pfm_viewer /media/sf_Shared_Data/gpuhome/pytorch-dispnet/data/dispnet/real_release/sgm_disp/left/${img}.pfm ./real_detect_result_s1.4/predict_${img}.exr
# DisparityTo3D ./real_detect_result_s1.4/predict_${img}.exr ./${img}.obj ./data/dispnet/real_release/frames_cleanpass/left/${img}.bmp
# meshlab ${img}.obj
