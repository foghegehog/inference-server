#include "logging.h"
#include "inference/bindingInfo.h"
#include "inference/ultraFaceOnnx.h"

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool UltraFaceOnnxEngine::build()
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
    assert(mInputDims.nbDims == 4);
    assert(network->getNbOutputs() == 4);

    mParams->mInputDims = network->getInput(0)->getDimensions();
    auto outputIndex = mEngine->getBindingIndex(mParams->outputTensorNames[0].c_str());
    auto scoresDims = mEngine->getBindingDimensions(outputIndex);
    mParams->mDetectionsCount = scoresDims.d[1];
    mParams->mNumClasses = scoresDims.d[2];
    //inference::gLogInfo << mParams->mDetectionsCount << std::endl;

    mBindings = std::make_shared<vector<BindingInfo>>();
    mBindings->reserve(mEngine->getNbBindings());
    //inference::gLogInfo << "Number of bindings: " << mEngine->getNbBindings() << std::endl;
    for (auto i = 0; i < mEngine->getNbBindings(); i++)
    {
        mBindings->emplace_back(
            mEngine->getBindingDataType(i),
            mEngine->getBindingDimensions(i),
            mEngine->getBindingVectorizedDim(i),
            mEngine->getBindingComponentsPerElement(i),
            mEngine->getBindingName(i),
            mEngine->bindingIsInput(i)
        );
    }

    return true;
}

std::unique_ptr<InferenceContext> UltraFaceOnnxEngine::get_inference_context()
{
    {
        const std::lock_guard<std::mutex> lock(mMutex);
        auto context = mEngine->createExecutionContext();
        if (!context)
        {
            throw logic_error("Failed to create execution context!");
        }

        return std::unique_ptr<InferenceContext>(new InferenceContext(context, mBindings, mParams));
    }
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool UltraFaceOnnxEngine::constructNetwork(InferenceUniquePtr<nvinfer1::IBuilder>& builder,
    InferenceUniquePtr<nvinfer1::INetworkDefinition>& network, InferenceUniquePtr<nvinfer1::IBuilderConfig>& config,
    InferenceUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams->onnxFileName, mParams->dataDirs).c_str(),
        static_cast<int>(inference::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
    if (mParams->fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams->int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        inferenceCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    inferenceCommon::enableDLA(builder.get(), config.get(), mParams->dlaCore);

    return true;
}



