# pfm_viewer img00000.pfm img00000.exr
# webp_viewer img00000.png img00000.exr

# ./build/DisparityTo3D ../data/0000009-imgL-s.exr ../data/0000009-imgL-s.obj ../data/0000009-imgL-s.bmp
# ./build/DisparityTo3D ../libSGM/test.exr ../data/0000009-imgL.obj ../data/0000009-imgL.bmp
pfm_viewer ../data/img00000.pfm ../data/img00000.exr
pfm_viewer ../data/img00000-sgm.pfm ../data/img00000-sgm.exr
./build/DisparityTo3D ../data/img00000.exr ../data/img00000.obj ../data/img00000-L.bmp 0.0038 1800.0
./build/DisparityTo3D ../data/img00000-sgm.exr ../data/img00000-sgm.obj ../data/img00000-L.bmp 0.0038 1800.0
# ./build/DisparityTo3D ../data/0000000-imgL.exr ../data/0000000-imgL.obj ../data/0000000-imgL.ppm

# meshlab test.obj
