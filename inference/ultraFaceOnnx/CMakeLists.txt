SET(INFERENCE_SOURCES 
    src/main.cpp
    src/inference/ultraFaceOnnx.cpp
    src/inference/inferenceContext.cpp
    src/http/lib.cpp
    src/http/listener.cpp
    src/http/session.cpp
    src/http/query.cpp
    src/http/routing.cpp
    src/statistics.cpp
    src/frames/files_iterator.cpp
    src/frames/filesystem_frame_reader.cpp)

set(INFERENCE_PARSERS "onnx")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread system filesystem regex REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

SET(CUSTOM_LIBS ${OpenCV_LIBS} ${Boost_LIBRARIES})
SET(CUSTOM_SOURCES ${BOOST_BEAST_FILES})

configure_file(config.ini ${TRT_OUT_DIR}/config.ini)

include(../CMakeInferenceTemplate.txt)
