mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DOPENCL_SGBM_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
ctest --output-on-failure
./opencl_sgbm_profile --json
cpack -G "TGZ;DEB"
