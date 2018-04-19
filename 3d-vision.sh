img=img00061

# nn 3d model
pfm_viewer ./real_detect_result_s1.5/predict_real_release_frames_cleanpass_${img}.pfm 
pfm_viewer ./real_detect_result_s1.5/predict_real_release_frames_cleanpass_${img}.pfm ./real_detect_result_s1.5/predict_${img}.exr
# pfm_viewer ./real_detect_result_s1.5/predict_${img}.pfm 
# pfm_viewer ./real_detect_result_s1.5/predict_${img}.pfm ./real_detect_result_s1.5/predict_${img}.exr
DisparityTo3D ./real_detect_result_s1.5/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
cd ./real_obj
meshlab ${img}.obj
cd ..

# # sgm 3d model
# pfm_viewer ./data_local/dispnet/real_release/sgm_disp/left/${img}.pfm ./real_detect_result_sgm/predict_${img}.exr
# DisparityTo3D ./real_detect_result_sgm/predict_${img}.exr ./real_obj/${img}.obj ./data_local/dispnet/real_release/frames_cleanpass/left/${img}.bmp
# cd ./real_obj
# meshlab ${img}.obj
# cd ..
