SET(INFERENCE_SOURCES 
    src/main.cpp
    src/inference/ultraFaceOnnx.cpp
    src/inference/inferenceContext.cpp
    src/inference/http/lib.cpp
    src/inference/http/listener.cpp
    src/inference/http/session.cpp)

set(INFERENCE_PARSERS "onnx")

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS thread system REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

SET(CUSTOM_LIBS ${OpenCV_LIBS} ${Boost_LIBRARIES})
SET(CUSTOM_SOURCES ${BOOST_BEAST_FILES})

include(../CMakeInferenceTemplate.txt)
