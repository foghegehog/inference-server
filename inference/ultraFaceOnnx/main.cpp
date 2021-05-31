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
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
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
    beast::error_code& error)
{
    http::response<http::empty_body> res{http::status::ok, req.version()};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "multipart/x-mixed-replace; boundary=frame");
    res.keep_alive();
    http::response_serializer<http::empty_body> sr{res};
    http::write_header(socket, sr);

    std::vector<uchar> buffer;
    for (auto f = 0; f < 4410; f++)
    {
        std::stringstream filename_stream;
        filename_stream <<  "../../data/ultraface/corridor/";
        filename_stream << std::setw(8) << std::setfill('0') << std::to_string(f) << ".jpg";  

        auto filepath = filename_stream.str();

        cv::Mat frame = cv::imread(filepath);
        //std::cout << filename;
        if (frame.empty())
        {
            std::cout << filepath << " is empty." << std::endl;
            continue;
        }

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
        cv::rectangle(frame, cv::Point(200, 200), cv::Point(300, 300), cv::Scalar(0, 255, 0));
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
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
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
    tcp::socket& socket)
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
        //handle_request(*doc_root, std::move(req), lambda);
        //handle_request(*doc_root, std::move(req), socket, close, ec);
        handle_request(std::move(req), socket, ec);
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
        if (argc != 4)
        {
            std::cerr <<
                "Usage: http-server <address> <port> <doc_root>\n" <<
                "Example:\n" <<
                "    http-server 0.0.0.0 8080 .\n";
            return EXIT_FAILURE;
        }
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
        auto const doc_root = std::make_shared<std::string>(argv[3]);

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
                std::move(socket))}.detach();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    /*inferenceCommon::Args args;
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
        "13_884_320_240.ppm", "webcam_320_240.ppm", "corridor.jpg", "corridor1.jpg"};

    //std::vector<inferenceCommon::PPM<3, 240, 320>> ppms(params.batchSize);
    std::vector<cv::Mat> batch;
    std::vector<Detection> detections;
    auto batches = ceil(imageList.size() / params.batchSize);
    for (auto b = 0; b < batches; b++)
    {
        batch.clear();
        detections.clear();
        for (int i = 0; i < params.batchSize; ++i)
        {
            auto image_index = (b * params.batchSize + i) % imageList.size();
            inference::gLogInfo << "Reading image " << imageList[image_index] << std::endl;
            auto path = locateFile(imageList[image_index], params.dataDirs);
            inference::gLogInfo << "Reading image from path " << path << std::endl;
            //readPPMFile(path, ppms[i]);
            auto image = cv::imread(path);
            cv::Mat input_image;
            cv::resize(image, input_image, cv::Size(320, 240));
            batch.push_back(input_image);
        }

        //if (!inference.infer(ppms, detections))
        if (!inference.infer(batch, detections))
        {
            return inference::gLogger.reportFail(inferenceTest);
        }

        array<string, 4> separators = {" ", " ", " ", ""};
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

    return 0;*/
}
