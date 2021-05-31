SET(INFERENCE_SOURCES main.cpp ultraFaceOnnx.cpp)

set(INFERENCE_PARSERS "onnx")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(CUSTOM_LIBS ${OpenCV_LIBS})

include(../CMakeInferenceTemplate.txt)
