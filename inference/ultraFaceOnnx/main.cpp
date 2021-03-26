//!
//! ultraFaceOnnx.cpp
//! It can be run with the following command line:
//! Command: ./ultra_face_onnx [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "detection.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "ultraFaceOnnx.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gInferenceName = "TensorRT.ultra_face_onnx";

//!
//! \brief Initializes members of the params struct using the command line args
//!
inferenceCommon::OnnxInferenceParams initializeInferenceParams(const inferenceCommon::Args& args)
{
    inferenceCommon::OnnxInferenceParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ultraface/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "ultraFace-RFB-320.onnx";
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("scores");
    params.outputTensorNames.push_back("boxes");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running inference
//!
void printHelpInfo()
{
    /*std::cout
        << "Usage: ./ultra_face_onnx" << std::endl;// [-h or --help] [-d or --datadir=<path to data directory>]
    [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories."
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;*/
}

int main(int argc, char** argv)
{
    inferenceCommon::Args args;
    bool argsOK = inferenceCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        inference::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto inferenceTest = inference::gLogger.defineTest(gInferenceName, argc, argv);

    inference::gLogger.reportTestStart(inferenceTest);

    auto params = initializeInferenceParams(args);
    UltraFaceOnnx inference(params);

    inference::gLogInfo << "Building and running a GPU inference engine for Onnx ultra face" << std::endl;

    if (!inference.build())
    {
        return inference::gLogger.reportFail(inferenceTest);
    }

    std::vector<std::string> imageList = {"13_24_320_240.ppm", "13_50_320_240.ppm", "13_72_320_240.ppm",
        "13_97_320_240.ppm", "13_140_320_240.ppm", "13_150_320_240.ppm", "13_178_320_240.ppm", "13_215_320_240.ppm",
        "13_219_320_240.ppm", "13_263_320_240.ppm", "13_295_320_240.ppm", "13_312_320_240.ppm", "13_698_320_240.ppm",
        "13_884_320_240.ppm"};

    std::vector<inferenceCommon::PPM<3, 240, 320>> ppms(params.batchSize);
    std::vector<Detection> detections;
    auto batches = ceil(imageList.size() / params.batchSize);
    for (auto b = 0; b < batches; b++)
    {
        for (int i = 0; i < params.batchSize; ++i)
        {
            auto image_index = (b * params.batchSize + i) % imageList.size();
            inference::gLogInfo << "Reading image " << imageList[image_index] << std::endl;
            auto path = locateFile(imageList[image_index], params.dataDirs);
            inference::gLogInfo << "Reading image from path " << path << std::endl;
            readPPMFile(path, ppms[i]);
        }

        if (!inference.infer(ppms, detections))
        {
            return inference::gLogger.reportFail(inferenceTest);
        }

        array<string, 4> separators = {", ", ", ", ", ", ""};
        for (const auto& d : detections)
        {
            inference::gLogInfo << "Detection: " << d.mScore << endl;
            for (int corners = 4, c = 0; c < corners; c++)
            {
                inference::gLogInfo << d.mBox[c] << separators[c];
            }

            inference::gLogInfo << std::endl;
        }

        inference::gLogger.reportPass(inferenceTest);
    }

    return 0;
}
