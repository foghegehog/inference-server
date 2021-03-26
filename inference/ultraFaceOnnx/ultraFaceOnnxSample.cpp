#include "ultraFaceOnnxSample.h"

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool UltraFaceOnnxSample::build()
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
bool UltraFaceOnnxSample::constructNetwork(InferenceUniquePtr<nvinfer1::IBuilder>& builder,
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
bool UltraFaceOnnxSample::infer()
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
bool UltraFaceOnnxSample::processInput(const inferenceCommon::BufferManager& buffers)
{
    const int batchSize = mInputDims.d[0];
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<std::string> imageList = {"13_24_320_240.ppm", "13_50_320_240.ppm", "13_72_320_240.ppm",
        "13_97_320_240.ppm", "13_140_320_240.ppm", "13_150_320_240.ppm", "13_178_320_240.ppm", "13_215_320_240.ppm",
        "13_219_320_240.ppm", "13_263_320_240.ppm", "13_295_320_240.ppm", "13_312_320_240.ppm", "13_698_320_240.ppm",
        "13_884_320_240.ppm"};

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
bool UltraFaceOnnxSample::verifyOutput(const inferenceCommon::BufferManager& buffers)
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
