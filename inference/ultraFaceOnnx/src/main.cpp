#include "logger.h"
#include "inference/detection.h"
#include "inference/ultraFaceInferenceParams.h"
#include "inference/ultraFaceOnnx.h"
#include "http/listener.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <array>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <thread>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

const std::string gInferenceName = "TensorRT.ultra_face_onnx";

//!
//! \brief Initializes members of the params struct using the command line args
//!
std::shared_ptr<UltraFaceInferenceParams> initializeInferenceParams(const inferenceCommon::Args& args)
{
    auto params = std::make_shared<UltraFaceInferenceParams>();
    params->dataDirs.push_back("data/ultraface/");
    params->onnxFileName = "ultraFace-RFB-320.onnx";
    params->inputTensorNames.push_back("input");
    params->outputTensorNames.push_back("scores");
    params->outputTensorNames.push_back("boxes");
    params->mPreprocessingMeans = {127.0f, 127.0f, 127.0f};
    params->mPreprocessingNorm = 128.0f;
    params->mDetectionThreshold = 0.9;
    params->mNumClasses = 2;
    params->mDetectionClassIndex = 1;
    params->dlaCore = args.useDLACore;
    params->int8 = args.runInInt8;
    params->fp16 = args.runInFp16;

    return params;
}

template<class Body, class Stream>
void write_response(
    http::response<Body>&& response, Stream& stream, bool& close, beast::error_code& ec)
{
    // Determine if we should close the connection after
    close = response.need_eof();

    // We need the serializer here because the serializer requires
    // a non-const file_body, and the message oriented version of
    // http::write only works with const messages.
    http::serializer<false, Body> sr{response};
    http::write(stream, sr, ec);
}


int main(int argc, char** argv)
{
        try
    {
        net::ip::address address;
        unsigned short port;
        std::string working_dir;
        int threads;
        if (argc == 1)
        {
            address = net::ip::make_address("0.0.0.0");
            port = 8080;
            working_dir = std::string("../../data/ultraface/");
            threads = 16;
        }
        else if (argc == 5)
        {
            address = net::ip::make_address(argv[1]);
            port = static_cast<unsigned short>(std::atoi(argv[2]));
            working_dir = std::string(argv[3]);
            threads = std::max<int>(1, std::atoi(argv[4]));
        }
        else
        {
            std::cerr <<
                "Usage: ultra_face_onnx <address> <port>\n" <<
                "Example:\n" <<
                "    ultra_face_onnx 0.0.0.0 8080 '../../data/ultraface/' 16\n";
            return EXIT_FAILURE;
        }


        auto inferenceTest = inference::gLogger.defineTest(gInferenceName, 0, {});
        inference::gLogger.reportTestStart(inferenceTest);

        inferenceCommon::Args args;
        bool argsOK = inferenceCommon::parseArgs(args, 0, {});
        auto params = initializeInferenceParams(args);
        UltraFaceOnnxEngine inferenceEngine(params);

        inference::gLogInfo << "Building and running a GPU inference engine for Onnx ultra face" << std::endl;

        if (!inferenceEngine.build())
        {
            inference::gLogger.reportFail(inferenceTest);
            return EXIT_FAILURE;
        }

        inference::gLogInfo << "The GPU inference engine is build." << std::endl;

            // The io_context is required for all I/O
        net::io_context ioc{threads};

        // Create and launch a listening port
        std::make_shared<listener>(
            ioc,
            tcp::endpoint{address, port},
            working_dir,
            inferenceEngine)->run();

        // Run the I/O service on the requested number of threads
        std::vector<std::thread> v;
        v.reserve(threads - 1);
        for(auto i = threads - 1; i > 0; --i)
            v.emplace_back(
            [&ioc]
            {
                ioc.run();
            });
        ioc.run();

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
