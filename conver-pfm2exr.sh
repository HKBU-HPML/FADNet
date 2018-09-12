dist=/media/sf_Shared_Data/dispnet/FusionPortal/data
for ((cam=0;cam <= 7;cam++)) 
do
    ./tools/pfm_viewer ${dist}/${cam}/0.pfm ${dist}/${cam}/p0.exr
done
