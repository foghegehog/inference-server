#ifndef ULTRAFACE_INFERENCE_PARAMS_H
#define ULTRAFACE_INFERENCE_PARAMS_H

#include "NvInfer.h"
#include "argsParser.h"

#include <array>

struct UltraFaceInferenceParams : public inferenceCommon::OnnxInferenceParams
{
    std::array<float, 3> mPreprocessingMeans;
    nvinfer1::Dims mInputDims;
    float mPreprocessingNorm;
    size_t mDetectionsCount;
    int mNumClasses;
    int mDetectionClassIndex;
    float mDetectionThreshold;
};

#endif