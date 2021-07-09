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
void fillInferenceParams(
    std::shared_ptr<UltraFaceInferenceParams>& params,
    const inferenceCommon::Args& args)
{
    params->dlaCore = args.useDLACore;
    params->int8 = args.runInInt8;
    params->fp16 = args.runInFp16;
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

void read_config(
    net::ip::address& address,
    unsigned short& port,
    std::string& working_dir,
    int& threads,
    std::shared_ptr<UltraFaceInferenceParams>& params)
{
    inference::gLogInfo << "Reading configuration." << std::endl;

    std::ifstream config("config.ini");
    const char separator = ' ';
    for(std::string line; std::getline(config, line); )
    {
        auto space_indx = line.find_first_of(separator);
        auto name = line.substr(0, space_indx);
        auto value = line.substr(space_indx + 1, line.size());
        inference::gLogInfo << name << ": ";
        if(name == "ADDRESS")
        {
            address = net::ip::make_address(value);
            inference::gLogInfo << address << std::endl;
            continue;
        }
        else if(name == "PORT")
        {
            port = stoi(value);
            inference::gLogInfo << port << std::endl;
            continue;
        }
        else if(name == "WORKING_DIR")
        {
            working_dir = std::move(value);
            inference::gLogInfo << working_dir << std::endl;
            continue;
        }
        else if(name == "THREADS")
        {
            threads = stoi(value);
            inference::gLogInfo << threads << std::endl;
            continue;
        }
        else if(name == "DATA_DIR")
        {
            params->dataDirs.push_back(std::move(value));
            inference::gLogInfo << params->dataDirs[0] << std::endl;
            continue;
        }
        else if(name == "ONNX_FILE_NAME")
        {
            params->onnxFileName = std::move(value);
            inference::gLogInfo << params->onnxFileName << std::endl;
            continue;
        }
        else if(name == "INPUT_TENSORS")
        {
            params->inputTensorNames.push_back(std::move(value));
            inference::gLogInfo << params->inputTensorNames[0] << std::endl;
            continue;
        }
        else if(name == "OUTPUT_TENSORS")
        {
            string::size_type start = 0;
            for(
                auto stop = value.find_first_of(separator);
                stop != string::npos;
                stop = value.find_first_of(separator, start))
            {
                params->outputTensorNames.push_back(value.substr(start, stop - start));
                start = stop + 1;
                inference::gLogInfo << params->outputTensorNames.back() << " ";
            }
            params->outputTensorNames.push_back(value.substr(start, value.size() - start));
            inference::gLogInfo << params->outputTensorNames.back() << std::endl;
            continue;
        }
        else if(name == "PREPROCESSING_MEANS")
        {
            string::size_type start = 0;
            auto channel = 0;
            for(
                auto stop = value.find_first_of(separator);
                stop != string::npos;
                stop = value.find_first_of(separator, start))
            {
                auto mean = stof(value.substr(start, stop - start));
                params->mPreprocessingMeans[channel++] = mean;
                start = stop + 1;
                inference::gLogInfo << mean << " ";
            }
            auto mean = stof(value.substr(start, value.size() - start));
            params->mPreprocessingMeans[channel] = mean;
            inference::gLogInfo << mean << std::endl;
            continue;
        }
        else if(name == "PREPROCESSING_NORM")
        {
            params->mPreprocessingNorm = stof(value);
            inference::gLogInfo << params->mPreprocessingNorm << std::endl;
            continue;
        }
        else if(name == "DETECTION_THRESHOLD")
        {
            params->mDetectionThreshold = stof(value);
            inference::gLogInfo << params->mDetectionThreshold << std::endl;
            continue;
        }
        else if(name == "NUM_CLASSES")
        {
            params->mNumClasses = stoi(value);
            inference::gLogInfo << params->mNumClasses << std::endl;
            continue;
        }
        else if(name == "DETECTION_CLASS")
        {
            params->mDetectionClassIndex = stoi(value);
            inference::gLogInfo << params->mDetectionClassIndex << std::endl;
            continue;
        }
    }
}

int main(int argc, char** argv)
{
    net::ip::address address;
    unsigned short port;
    std::string working_dir;
    int threads;
    std::shared_ptr<UltraFaceInferenceParams> inferenceParams;
    inferenceCommon::Args args;

    try
    {
        auto inferenceTest = inference::gLogger.defineTest(gInferenceName, 0, {});
        inference::gLogger.reportTestStart(inferenceTest);
        inferenceParams = std::make_shared<UltraFaceInferenceParams>();

        read_config(address, port, working_dir, threads, inferenceParams);
     
        if (argc > 1)
        {
            inference::gLogInfo << "Parsing parameters." << std::endl;
            address = net::ip::make_address(argv[1]);
            inference::gLogInfo << "Address: " << address << std::endl;
            port = static_cast<unsigned short>(std::atoi(argv[2]));
            inference::gLogInfo << "Port: " << port << std::endl;
            working_dir = std::string(argv[3]);
            inference::gLogInfo << "Working directory: " << working_dir << std::endl;
            threads = std::max<int>(1, std::atoi(argv[4]));
            inference::gLogInfo << "Num threads: " << threads << std::endl;
        }

        inferenceCommon::parseArgs(args, argc, argv);
        fillInferenceParams(inferenceParams, args);

        UltraFaceOnnxEngine inferenceEngine(inferenceParams);

        inference::gLogInfo << "Building and running a GPU inference engine for ultraFace Onnx" << std::endl;

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
