

#disp_path=/datasets/driving_release/
#norm_path=/datasets/qiangw/driving_release/normal/
#focus_length=1050.0
#dataset=FlyingThings

#disp_path=./monkaa_release/
#norm_path=/mnt/ssd/monkaa_release_norm/
#focus_length=1050
#dataset=FlyingThings

disp_path=/datasets/qiangw/MPI-Sintel/training/disparities/
norm_path=/datasets/qiangw/MPI-Sintel/training/normal/
focus_length=1120
dataset=Sintel

python generate_norm_from_disp.py --path $disp_path --norm_path $norm_path \
	--focus_length $focus_length \
	--dataset $dataset
