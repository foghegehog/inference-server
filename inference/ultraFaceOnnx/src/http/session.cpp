#include "../../include/http/session.h"
#include "../../include/http/lib.h"

#include <functional> 
#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/http/write.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

namespace beast = boost::beast;
namespace http = beast::http;
using tcp = boost::asio::ip::tcp;

void session::run()
{
    do_read();
}

void session::do_read()
{
    // Make the request empty before reading,
    // otherwise the operation behavior is undefined.
    m_req = {};

    log("Started reading socket");

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

    // The lifetime of the response has to extend
    // for the duration of the async operation so
    // we use a shared_ptr to manage it.
    m_header_res = std::make_shared<http::response<http::empty_body>>(http::status::ok, m_req.version());
    m_header_res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
    m_header_res->set(http::field::content_type, "multipart/x-mixed-replace; boundary=frame");
    m_header_res->keep_alive();

    log("Writing M-JPEG header.");
    http::async_write(
        m_socket,
        *m_header_res,
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_write,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2,
                0)));
}

void session::on_write(
    boost::system::error_code ec,
    std::size_t bytes_transferred,
    int frames_send)
{
    auto processing_start = std::chrono::high_resolution_clock::now();

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
        log("Handling next frame.");
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
        batch.clear();
        batch.push_back(std::move(input_frame));

        inference::gLogInfo << "Running inference!" << std::endl;
        if (!m_inference_context->infer(batch, detections))
        {
            inference::gLogInfo << "Error during inference!" << std::endl;
            continue;
        }
        inference::gLogInfo << "Inference successfull." << std::endl;
    }
    while(detections.empty() && (frame_num < total_frames));

    inference::gLogInfo << "Drawing detections." << std::endl;
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
    std::vector<uchar> buffer;
    cv::imencode(".jpg", frame, buffer, std::vector<int> {cv::IMWRITE_JPEG_QUALITY, 95});
    auto const size = buffer.size();

    inference::gLogInfo << "Writing response."  << std::endl;
    m_res = std::make_shared<http::response<http::vector_body<unsigned char>>>(
        std::piecewise_construct,
        std::make_tuple(std::move(buffer)),
        std::make_tuple(http::status::ok, m_req.version()));
    m_res->set(http::field::body, "--frame");
    m_res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
    m_res->set(http::field::content_type, "image/jpeg");
    m_res->content_length(size);
    m_res->keep_alive(m_req.keep_alive());

    //std::this_thread::sleep_for(std::chrono::milliseconds(20));

    auto processing_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = processing_end - processing_start;
     using nano_duration = std::chrono::duration<int, std::nano>;
    auto pause = std::chrono::duration_cast<nano_duration>(m_frame_pause - elapsed);
    log(std::to_string(pause.count()));
    m_timer.expires_after(pause);
    m_timer.async_wait(
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_timer,
                shared_from_this(),
                std::placeholders::_1,
                frame_num)));
}

void session::on_timer(const boost::system::error_code& error, int frame_num)
{
    log("On timer pause");
       // Write the response
    http::async_write(
        m_socket,
        *m_res,
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