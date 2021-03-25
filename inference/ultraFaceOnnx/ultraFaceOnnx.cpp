//!
//! ultraFaceOnnx.cpp
//! It can be run with the following command line:
//! Command: ./ultra_face_onnx [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gInferenceName = "TensorRT.ultra_face_onnx";

//! \brief  The UltraFaceOnnx class implements the ONNX UltraFace inference
//!
//! \details It creates the network using an ONNX model
//!
class UltraFaceOnnx
{
    template <typename T>
    using InferenceUniquePtr = std::unique_ptr<T, inferenceCommon::InferDeleter>;

public:
    UltraFaceOnnx(const inferenceCommon::OnnxInferenceParams& params)
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

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool UltraFaceOnnx::build()
{
    auto builder
        = InferenceUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(inference::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = InferenceUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = InferenceUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = InferenceUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, inference::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), inferenceCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);
    // assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 4);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool UltraFaceOnnx::constructNetwork(InferenceUniquePtr<nvinfer1::IBuilder>& builder,
    InferenceUniquePtr<nvinfer1::INetworkDefinition>& network, InferenceUniquePtr<nvinfer1::IBuilderConfig>& config,
    InferenceUniquePtr<nvonnxparser::IParser>& parser)
{
    // parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 240, 320),
    // nvuffparser::UffInputOrder::kNCHW); parser->registerOutput(mParams.outputTensorNames[0].c_str());
    // parser->registerOutput(mParams.outputTensorNames[1].c_str());

    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(inference::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        inferenceCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    inferenceCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine
//!
//! \details This function is the main execution function. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool UltraFaceOnnx::infer()
{
    // Create RAII buffer manager object
    inferenceCommon::BufferManager buffers(mEngine);

    auto context = InferenceUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool UltraFaceOnnx::processInput(const inferenceCommon::BufferManager& buffers)
{
    const int batchSize = mInputDims.d[0];
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<std::string> imageList
        = {"13_24_320_240.ppm", "13_50.ppm", "13_72.ppm", "13_97.ppm", "13_140.ppm", "13_150.ppm", "13_178.ppm",
            "13_215.ppm", "13_219.ppm", "13_263.ppm", "13_295.ppm", "13_312.ppm", "13_698.ppm", "13_884.ppm"};

    std::vector<inferenceCommon::PPM<3, 240, 320>> ppms(batchSize);

    for (int i = 0; i < batchSize; ++i)
    {
        auto path = locateFile(imageList[i], mParams.dataDirs);
        std::cout << "Reading image from path " << path << endl;
        readPPMFile(path, ppms[i]);
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float pixelMean[3]{127.0f, 127.0f, 127.0f};

    // Host memory for input buffer
    std::cout << "Preprocessing image" << endl;
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j]
                    = (float(ppms[i].buffer[j * inputC + c]) - pixelMean[c]) / 128.0;
                //(float(ppms[i].buffer[j * inputC + 2 - c]) - pixelMean[c]) / 128.0;
                // if (j < 100)
                //{
                //    cout << hostDataBuffer[i * volImg + c * volChl + j] << endl;
                //}
            }
        }
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool UltraFaceOnnx::verifyOutput(const inferenceCommon::BufferManager& buffers)
{
    std::cout << "Output phase reached." << endl;

    const float* scores = static_cast<const float*>(buffers.getHostBuffer("scores"));
    const float* boxes = static_cast<const float*>(buffers.getHostBuffer("boxes"));

    ofstream outfile;
    outfile.open("13_24_320_240.txt");

    for (int i = 0; i < 4420; ++i)
    {
        array<string, 4> separators = {" ", " ", " ", ""};
        auto back_score = *(scores + i * 2);
        auto face_score = *(scores + i * 2 + 1);

        outfile << back_score << endl;
        outfile << face_score << endl;

        for (int corners = 4, c = 0; c < corners; c++)
        {
            outfile << boxes[i * corners + c] << separators[c];
        }
        outfile << std::endl;

        if (face_score > 0.5)
        {
            cout << i << " " << face_score << endl;
            for (int corners = 4, c = 0; c < corners; c++)
            {
                cout << boxes[i * corners + c] << separators[c];
            }
            cout << std::endl;
        }
    }

    outfile.close();

    return true;
}

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

    UltraFaceOnnx inference(initializeInferenceParams(args));

    inference::gLogInfo << "Building and running a GPU inference engine for Onnx ultra face" << std::endl;

    if (!inference.build())
    {
        return inference::gLogger.reportFail(inferenceTest);
    }
    if (!inference.infer())
    {
        return inference::gLogger.reportFail(inferenceTest);
    }

    return inference::gLogger.reportPass(inferenceTest);

    std::cout << "Launched inference!" << std::endl;
    return 0;
}
