#include "../include/inferenceContext.h"


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
    // const int batchSize = mInputDims.d[0];
    const int batchSize = batch.size();
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    float* hostDataBuffer = static_cast<float*>(mBufferManager->getHostBuffer(mInputName));
    float pixelMean[3]{127.0f, 127.0f, 127.0f};

    // Host memory for input buffer
    inference::gLogInfo << "Preprocessing image" << std::endl;
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
                    = (float(batch[i].at<cv::Vec3b>(y, x).val[c]) - pixelMean[c]) / 128.0;
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
    inference::gLogInfo << "Output phase reached." << std::endl;

    const float* scores = static_cast<const float*>(mBufferManager->getHostBuffer("scores"));
    const float* boxes = static_cast<const float*>(mBufferManager->getHostBuffer("boxes"));

    // ofstream outfile;
    // outfile.open("13_24_320_240.txt");

    for (int i = 0; i < 4420; ++i)
    {
        // array<string, 4> separators = {" ", " ", " ", ""};
        auto backScore = *(scores + i * 2);
        auto faceScore = *(scores + i * 2 + 1);

        // outfile << back_score << endl;
        // outfile << face_score << endl;

        const int corners = 4;

        // for (int c = 0; c < corners; c++)
        //{

        // outfile << boxes[i * corners + c] << separators[c];
        //}
        // outfile << std::endl;

        if (faceScore > 0.9)
        {
            std::array<float, corners> box;
            for (int c = 0; c < corners; c++)
            {
                box[c] = boxes[i * corners + c];
            }

            detections.emplace_back(faceScore, box);
            // cout << std::endl;
        }
    }

    // outfile.close();

    return true;
}