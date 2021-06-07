#ifndef INFERENCE_CONTEXT_H
#define INFERENCE_CONTEXT_H

#include "buffers.h"
#include "detection.h"

#include <opencv2/imgcodecs.hpp>
#include <memory>
#include <vector>

class InferenceContext
{
    template <typename T>
    using InferenceUniquePtr = std::unique_ptr<T, inferenceCommon::InferDeleter>;

public:
    InferenceContext(
        nvinfer1::IExecutionContext* executionContext,
        std::vector<BindingInfo>&& bindings,
        nvinfer1::Dims inputDims,
        const std::string& input_name)
        :mInputName(input_name)
    {
        mExecutionContext = InferenceUniquePtr<nvinfer1::IExecutionContext>(executionContext);
        mBufferManager = std::unique_ptr<inferenceCommon::BufferManager>(
            new inferenceCommon::BufferManager(executionContext, std::move(bindings)));
        mInputDims = inputDims;
    }

    //!
    //! \brief Runs the TensorRT inference engine
    //!
    bool infer(const std::vector<cv::Mat>& batch, std::vector<Detection>& detections);

private:

    bool preprocessInput(const std::vector<cv::Mat>& batch);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool parseOutput(std::vector<Detection>& detections);

    InferenceUniquePtr<nvinfer1::IExecutionContext> mExecutionContext;
    std::unique_ptr<inferenceCommon::BufferManager> mBufferManager;
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    std::string mInputName;
};

#endif