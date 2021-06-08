#ifndef INFERENCE_CONTEXT_H
#define INFERENCE_CONTEXT_H

#include "buffers.h"
#include "detection.h"
#include "ultraFaceInferenceParams.h"

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
        std::shared_ptr<std::vector<BindingInfo>> bindings,
        std::shared_ptr<UltraFaceInferenceParams> params)
        :mParams(params)
    {
        mExecutionContext = InferenceUniquePtr<nvinfer1::IExecutionContext>(executionContext);
        mBufferManager = std::unique_ptr<inferenceCommon::BufferManager>(
            new inferenceCommon::BufferManager(executionContext, bindings));
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
    std::shared_ptr<UltraFaceInferenceParams> mParams;
};

#endif