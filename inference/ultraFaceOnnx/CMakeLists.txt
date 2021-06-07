SET(INFERENCE_SOURCES main.cpp ultraFaceOnnx.cpp inferenceContext.cpp)

set(INFERENCE_PARSERS "onnx")

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS thread system REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

SET(CUSTOM_LIBS ${OpenCV_LIBS} ${Boost_LIBRARIES})
SET(CUSTOM_SOURCES ${BOOST_BEAST_FILES})

include(../CMakeInferenceTemplate.txt)
