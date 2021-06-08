#include "logger.h"
#include "../include/detection.h"
#include "../include/ultraFaceInferenceParams.h"
#include "../include/ultraFaceOnnx.h"

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

void handle_request(
    http::request<http::string_body>&& req,
    tcp::socket& socket,
    UltraFaceOnnxEngine& inferenceEngine,
    beast::error_code& error)
{
    inference::gLogInfo << "Creating inference executing context" << std::endl;

    auto context = inferenceEngine.get_inference_context();

    inference::gLogInfo << "The GPU inference executing context is build. Start streaming." << std::endl;

    http::response<http::empty_body> res{http::status::ok, req.version()};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "multipart/x-mixed-replace; boundary=frame");
    res.keep_alive();
    http::response_serializer<http::empty_body> sr{res};
    http::write_header(socket, sr);

    std::vector<cv::Mat> batch;
    std::vector<Detection> detections;

    std::vector<uchar> buffer;
    for (auto f = 0; f < 4410; f++)
    {
        std::stringstream filename_stream;
        filename_stream <<  "../../data/ultraface/corridor/";
        filename_stream << std::setw(8) << std::setfill('0') << std::to_string(f) << ".jpg";  

        auto filepath = filename_stream.str();

        batch.clear();
        detections.clear();
        //inference::gLogInfo << "Reading image from path " << filepath << std::endl;
        auto frame = cv::imread(filepath);
        if (frame.empty())
        {
            std::cout << filepath << " is empty." << std::endl;
            continue;
        }
        cv::Mat input_frame;
        cv::resize(frame, input_frame, cv::Size(320, 240));
        batch.push_back(input_frame);

        if (!context.infer(batch, detections))
        {
            inference::gLogInfo << "Error during inference!" << std::endl;
            continue;
        }

        if (detections.empty())
        {
            continue;
        }

        int width = frame.cols;
        int height = frame.rows;
        for (const auto& detection: detections)
        {
            cv::rectangle(
                frame,
                cv::Point(detection.mBox[0] * width, detection.mBox[1] * height),
                cv::Point(detection.mBox[2] * width, detection.mBox[3] * height),
                cv::Scalar(0, 0, 255));
        }

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
        cv::imencode(".jpg", frame, buffer, std::vector<int> {cv::IMWRITE_JPEG_QUALITY, 95});
        auto const size = buffer.size();

        http::response<http::vector_body<unsigned char>> res{std::piecewise_construct,
                        std::make_tuple(std::move(buffer)),
                        std::make_tuple(http::status::ok, req.version())};
        res.set(http::field::body, "--frame");
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "image/jpeg");
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        http::write(socket, res, error);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// Report a failure
void
fail(beast::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// Handles an HTTP server connection
void
do_session(
    tcp::socket& socket,
    UltraFaceOnnxEngine& engine)
{
    bool close = false;
    beast::error_code ec;

    // This buffer is required to persist across reads
    beast::flat_buffer buffer;

    for(;;)
    {
        // Read a request
        http::request<http::string_body> req;
        http::read(socket, buffer, req, ec);
        if(ec == http::error::end_of_stream)
            break;
        if(ec)
            return fail(ec, "read");

        // Send the response
        handle_request(std::move(req), socket, engine, ec);
        if(ec)
            return fail(ec, "write");
        if(close)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            break;
        }
    }

    // Send a TCP shutdown
    socket.shutdown(tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
}

int main(int argc, char** argv)
{
        try
    {
        // Check command line arguments.
        if (argc != 3)
        {
            std::cerr <<
                "Usage: ultra_face_onnx <address> <port>\n" <<
                "Example:\n" <<
                "    ultra_face_onnx 0.0.0.0 8080\n";
            return EXIT_FAILURE;
        }
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::atoi(argv[2]));

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
        net::io_context ioc{1};

        // The acceptor receives incoming connections
        tcp::acceptor acceptor{ioc, {address, port}};
        for(;;)
        {
            // This will receive the new connection
            tcp::socket socket{ioc};

            // Block until we get a connection
            acceptor.accept(socket);

            // Launch the session, transferring ownership of the socket
            std::thread{std::bind(
                &do_session,
                std::move(socket),
                std::reference_wrapper<UltraFaceOnnxEngine>(inferenceEngine))}.detach();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
