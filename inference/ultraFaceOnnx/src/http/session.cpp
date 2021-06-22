#include "../../include/http/session.h"
#include "../../include/http/lib.h"
#include "../../include/http/query.h"

#include <functional> 
#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/http/write.hpp>
#include <boost/regex.hpp>
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
    m_header_res->set(
        http::field::content_type,
        "multipart/x-mixed-replace; boundary=" + m_frame_boundary);
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
                std::placeholders::_2)));
}

void session::on_write(
    boost::system::error_code ec,
    std::size_t bytes_transferred)
{    
    boost::ignore_unused(bytes_transferred);

    if(ec)
    {
        return fail(ec, "write");
    }

    if (m_frame_reader.is_finished() && m_frame_buffers.empty())
    {
        log("Closing");
        do_close();
        return;
    }

    if (!m_frame_buffers.empty())
    {
        m_timer.expires_after(m_frame_pause);
        m_timer.async_wait(
            boost::asio::bind_executor(
                m_strand,
                std::bind(
                    &session::on_timer,
                    shared_from_this(),
                    std::placeholders::_1)));

        return;
    }

    auto pause = m_frame_pause;
    log("Start processing frames.");
    do
    {
        auto processing_start = std::chrono::high_resolution_clock::now();
        process_frame();
        auto processing_end = std::chrono::high_resolution_clock::now();
        auto processing_time = processing_end - processing_start;
        m_statistics.update_avg_processing(processing_time.count());
        pause -= processing_time;
    } while (!m_frame_reader.is_finished()
        && (pause.count() > m_statistics.get_avg_processing_time()));

    if (m_frame_reader.is_finished())
    {
        // Denotes end of images list
        log("Image list finished.");
        m_frame_buffers.push(std::vector<uchar>());
    }

    m_timer.expires_after(pause);
    m_timer.async_wait(
        boost::asio::bind_executor(
            m_strand,
            std::bind(
                &session::on_timer,
                shared_from_this(),
                std::placeholders::_1)));
}

void session::on_timer(const boost::system::error_code& error)
{
    auto buffer = std::move(m_frame_buffers.front());
    m_frame_buffers.pop();
    auto const size = buffer.size();

    if (size == 0)
    {
        // Writing termination boundary
        log("Writing termination boundary.");

        m_header_res = std::make_shared<http::response<http::empty_body>>(http::status::ok, m_req.version());
        m_header_res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
        m_header_res->set(http::field::body, "--" + m_frame_boundary + "--");

        http::async_write(
            m_socket,
            *m_header_res,
            boost::asio::bind_executor(
                m_strand,
                std::bind(
                    &session::on_write,
                    shared_from_this(),
                    std::placeholders::_1,
                    std::placeholders::_2)));
    }
    else
    {
        log("Writing response.");
        m_res = std::make_shared<http::response<http::vector_body<unsigned char>>>(
        std::piecewise_construct,
        std::make_tuple(std::move(buffer)),
        std::make_tuple(http::status::ok, m_req.version()));
        m_res->set(http::field::body, "--" + m_frame_boundary);
        m_res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
        m_res->set(http::field::content_type, "image/jpeg");
        m_res->content_length(size);
        m_res->keep_alive(m_req.keep_alive());

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
                    std::placeholders::_2)));
    }
}

void session::do_close()
{
    // Send a TCP shutdown
    boost::system::error_code ec;
    m_socket.shutdown(tcp::socket::shutdown_send, ec);
    m_socket.close();

    // At this point the connection is closed gracefully
}

void session::process_frame()
{
    cv::Mat frame;
    cv::Mat input_frame;
    std::vector<cv::Mat> batch;
    std::vector<Detection> detections;

    bool finished = false;
    do
    {
        log("Reading next frame");
        frame = m_frame_reader.read_frame();
        if (frame.empty())
        {
            log("Frame + is empty. Skipped.");
            continue;
        }
        
        cv::resize(
            frame,
            input_frame,
            cv::Size(
                m_inference_context->get_input_width(),
                m_inference_context->get_input_height()));
        batch.clear();
        batch.push_back(std::move(input_frame));

        inference::gLogInfo << "Running inference!" << std::endl;
        if (!m_inference_context->infer(batch, detections))
        {
            inference::gLogInfo << "Error during inference!" << std::endl;
            continue;
        }
        inference::gLogInfo << "Inference successfull." << std::endl;

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
        m_frame_buffers.push(std::move(buffer));
        inference::gLogInfo << "Frame ready." << std::endl;
    }
    while(frame.empty() && !m_frame_reader.is_finished());
    
    log("Finished processing frame.");
}