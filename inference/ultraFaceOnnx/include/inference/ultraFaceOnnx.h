#ifndef ULTRA_FACE_ONNX_H
#define ULTRA_FACE_ONNX_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "detection.h"
#include "inferenceContext.h"
#include "parserOnnxConfig.h"
#include "ultraFaceInferenceParams.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

//! \brief  The UltraFaceOnnxSample class implements the ONNX UltraFace inference
//!
//! \details It creates the network using an ONNX model
//!
class UltraFaceOnnxEngine
{
    template <typename T>
    using InferenceUniquePtr = std::unique_ptr<T, inferenceCommon::InferDeleter>;

public:
    UltraFaceOnnxEngine(std::shared_ptr<UltraFaceInferenceParams> params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    std::unique_ptr<InferenceContext> get_inference_context();

private:
    std::shared_ptr<UltraFaceInferenceParams> mParams;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::shared_ptr<std::vector<BindingInfo>> mBindings;

    std::mutex mMutex;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(InferenceUniquePtr<nvinfer1::IBuilder>& builder,
        InferenceUniquePtr<nvinfer1::INetworkDefinition>& network, InferenceUniquePtr<nvinfer1::IBuilderConfig>& config,
        InferenceUniquePtr<nvonnxparser::IParser>& parser);
};

#endif