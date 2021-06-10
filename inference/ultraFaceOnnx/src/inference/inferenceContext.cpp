#include "../../include/inference/inferenceContext.h"

#include <iostream> 
#include <set>

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

float InferenceContext::get_intersection_area(const Detection & first, const Detection & second)
{
    auto intersection_left = first.mBox[0] > second.mBox[0] ? first.mBox[0] : second.mBox[0]; 
    auto intersection_right = first.mBox[2] < second.mBox[2] ? first.mBox[2] : second.mBox[2];
    auto w = intersection_right - intersection_left;
    auto intersection_top = first.mBox[1] > second.mBox[1] ? first.mBox[1] : second.mBox[1]; 
    auto intersection_bottom = first.mBox[3] < second.mBox[3] ? first.mBox[3] : second.mBox[3];  
    auto h = intersection_bottom - intersection_top;

    return w * h;
}

float InferenceContext::get_iou(const Detection & first, const Detection & second)
{
    auto intersection_area = get_intersection_area(first, second);
    auto union_area = first.get_box_area() + second.get_box_area() - intersection_area;
    return intersection_area / union_area;
}

void InferenceContext::nms(
        std::multiset<Detection, ScoreDescendingCompare>& all_detections,
        std::vector<Detection>& result_detections,
        float iou_threshold)
{
    while (!all_detections.empty())
    {
        auto proposal_it = all_detections.begin();
        auto proposal = *proposal_it;
        all_detections.erase(proposal_it);

        bool discard = false;
        for (const auto& other: all_detections)
        {
            auto iou = get_iou(proposal, other);
            if (iou > iou_threshold)
            {
                discard = true;
                break;
            }
        }

        if (!discard)
        {
            result_detections.push_back(proposal);
        }
    }
    
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

    std::multiset<Detection, ScoreDescendingCompare> all_detections; 

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

            all_detections.emplace(faceScore, std::move(box));
        }
    }

    nms(all_detections, detections, 0.5f);

    return true;
}