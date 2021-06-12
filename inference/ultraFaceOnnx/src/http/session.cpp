#include "../../include/http/session.h"
#include "../../include/http/lib.h"

#include <functional> 
#include <boost/asio.hpp>
#include <boost/beast/websocket.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

namespace beast = boost::beast;
namespace http = beast::http;

void session::run()
{
    do_read();
}

void session::do_read()
{
    // Make the request empty before reading,
    // otherwise the operation behavior is undefined.
    m_req = {};

    // Read a request
    http::async_read(m_socket, m_buffer, m_req,
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_read,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2)));
}

void session::on_read(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    // This indicates that the session was closed
    if(ec == http::error::end_of_stream)
    {
        return do_close();
    }

    if(ec)
    {
        fail(ec, "read");
    }

    inference::gLogInfo << "Start streaming the GPU inference results." << std::endl;

    http::response<http::empty_body> res{http::status::ok, req.version()};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "multipart/x-mixed-replace; boundary=frame");
    res.keep_alive();

    // The lifetime of the response has to extend
    // for the duration of the async operation so
    // we use a shared_ptr to manage it.
    auto sr = std::make_shared<http::response_serializer<http::empty_body>>(res);

    http::async_write_header(
        socket,
        sr,
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_write,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2,
                0)));
}

void on_write(
    boost::system::error_code ec,
    std::size_t bytes_transferred,
    size_t frames_send)
{
    boost::ignore_unused(bytes_transferred);

    if(ec)
    {
        return fail(ec, "write");
    }

    const int total_frames = 4410;
    if (frames_send >= total_frames)
    {
        return do_close();
    }

    int frame_num = frames_send; 
    cv::Mat frame;
    cv::Mat input_frame;
    std::vector<cv::Mat> batch;
    std::vector<Detection> detections;

    do
    {
        frame_num += 1;
        std::stringstream filename_stream;
        filename_stream <<  m_base_folder;
        filename_stream << std::setw(8) << std::setfill('0') << std::to_string(frame_num) << ".jpg"; 

        auto filepath = filename_stream.str();

        inference::gLogInfo << "Reading image from path " << filepath << std::endl;
        frame = cv::imread(filepath);
        if (frame.empty())
        {
            std::cout << filepath << " is empty." << std::endl;
            continue;
        }
        
        cv::resize(frame, input_frame, cv::Size(320, 240));
        batch.push_back(input_frame);

        if (!context.infer(batch, detections))
        {
            inference::gLogInfo << "Error during inference!" << std::endl;
            continue;
        }
    }
    while(detections.empty() && (frame_num < total_frames));

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

   auto res = std::make_shared<http::response<http::vector_body<unsigned char>>>(
        std::piecewise_construct,
        std::make_tuple(std::move(buffer)),
        std::make_tuple(http::status::ok, req.version()));
    res->set(http::field::body, "--frame");
    res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res->set(http::field::content_type, "image/jpeg");
    res->content_length(size);
    res->keep_alive(req.keep_alive());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Write the response
    http::async_write(
        m_socket,
        *res,
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_write,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2,
                frame_num)));
    
}

void session::do_close()
{
    // Send a TCP shutdown
    boost::system::error_code ec;
    m_socket.shutdown(tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
}