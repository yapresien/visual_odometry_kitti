mkdir build && cd build
cmake .. -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
make -j$(nproc)
./kitti_vo_ocl