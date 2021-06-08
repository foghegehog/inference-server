#include "../include/inferenceContext.h"

#include <iostream> 

bool InferenceContext::infer(
    const std::vector<cv::Mat>& batch,
    std::vector<Detection>& detections)
{
    // Read the input data into the managed buffers
    if (!preprocessInput(batch))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    mBufferManager->copyInputToDevice();

    bool status = mExecutionContext->executeV2(mBufferManager->getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    mBufferManager->copyOutputToHost();

    // Verify results
    if (!parseOutput(detections))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool InferenceContext::preprocessInput(const std::vector<cv::Mat>& batch)
{
    const int batchSize = batch.size();
    const int inputC = mParams->mInputDims.d[1];
    const int inputH = mParams->mInputDims.d[2];
    const int inputW = mParams->mInputDims.d[3];
    std::array<float, 3>& pixelMean = mParams->mPreprocessingMeans;
    float pixelNorm = mParams->mPreprocessingNorm;

    float* hostDataBuffer = mBufferManager->getHostBuffer<float>(mParams->inputTensorNames[0]);

    // Host memory for input buffer
    //inference::gLogInfo << "Preprocessing image" << std::endl;
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                auto y = j / inputW;
                auto x = j % inputW;
                hostDataBuffer[i * volImg + c * volChl + j]
                    = (float(batch[i].at<cv::Vec3b>(y, x).val[c]) - pixelMean[c]) / pixelNorm;
            }
        }
    }

    return true;
}

//!
//! \brief Detects objects and verify result
//!
//! \return whether the output matches expectations
//!
bool InferenceContext::parseOutput(std::vector<Detection>& detections)
{

    const float* scores = mBufferManager->getHostBuffer<float>("scores");
    const float* boxes = mBufferManager->getHostBuffer<float>("boxes");

    for (int i = 0; i < mParams->mDetectionsCount; ++i)
    {
        auto faceScoreOffset = i * mParams->mNumClasses + mParams->mDetectionClassIndex;
        auto faceScore = *(scores + faceScoreOffset);

        if (faceScore > mParams->mDetectionThreshold)
        {
            std::array<float, Detection::mNumCorners> box;
            for (int c = 0; c < box.size(); c++)
            {
                box[c] = boxes[i * box.size() + c];
            }

            detections.emplace_back(faceScore, std::move(box));
        }
    }

    return true;
}