#include "logging.h"
#include "../include/bindingInfo.h"
#include "../include/ultraFaceOnnx.h"

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
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 4);

    mBindings = std::make_shared<vector<BindingInfo>>();
    mBindings->reserve(mEngine->getNbBindings());
    //inference::gLogInfo << "Number of bindings: " << mEngine->getNbBindings() << std::endl;
    for (auto i = 0; i < mEngine->getNbBindings(); i++)
    {
        //inference::gLogInfo << "Binding name: " << mEngine->getBindingName(i);
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

InferenceContext UltraFaceOnnxEngine::get_inference_context()
{
    // Create RAII buffer manager object
    auto context = mEngine->createExecutionContext();
    if (!context)
    {
        throw logic_error("Failed to create execution context!");
    }

    return InferenceContext(context, mBindings, mInputDims, mParams.inputTensorNames[0]);
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



