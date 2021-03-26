SET(INFERENCE_SOURCES
    main.cpp
    ultraFaceOnnx.cpp
)

set(INFERENCE_PARSERS "onnx")

include(../CMakeInferenceTemplate.txt)
