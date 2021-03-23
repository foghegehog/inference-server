 cd $TRT_OSSPATH
 mkdir -p build && cd build
 cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DGPU_ARCHS="75"
 make -j$(nproc)
