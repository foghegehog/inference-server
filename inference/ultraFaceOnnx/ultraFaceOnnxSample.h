#ifndef ULTRA_FACE_ONNX_SAMPLE_H
#define ULTRA_FACE_ONNX_SAMPLE_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>

//! \brief  The UltraFaceOnnxSample class implements the ONNX UltraFace inference
//!
//! \details It creates the network using an ONNX model
//!
class UltraFaceOnnxSample
{
    template <typename T>
    using InferenceUniquePtr = std::unique_ptr<T, inferenceCommon::InferDeleter>;

public:
    UltraFaceOnnxSample(const inferenceCommon::OnnxInferenceParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine
    //!
    bool infer();

private:
    inferenceCommon::OnnxInferenceParams mParams;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(InferenceUniquePtr<nvinfer1::IBuilder>& builder,
        InferenceUniquePtr<nvinfer1::INetworkDefinition>& network, InferenceUniquePtr<nvinfer1::IBuilderConfig>& config,
        InferenceUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const inferenceCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const inferenceCommon::BufferManager& buffers);
};

#endif