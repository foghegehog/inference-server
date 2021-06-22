#ifndef SESSION_H
#define SESSION_H

#include "../inference/inferenceContext.h"
#include "../frames/filesystem_frame_reader.h"
#include "../statistics.h"
#include "routing.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <string>

// Handles an HTTP server connection
class session : public std::enable_shared_from_this<session>
{
    boost::asio::ip::tcp::socket m_socket;
    boost::asio::strand<boost::asio::io_context::executor_type> m_strand;
    boost::beast::multi_buffer m_buffer;

    std::unique_ptr<InferenceContext> m_inference_context;

    boost::beast::http::request<boost::beast::http::string_body> m_req;

    std::shared_ptr<boost::beast::http::response<boost::beast::http::empty_body>> m_header_res;

    std::shared_ptr<boost::beast::http::response<boost::beast::http::vector_body<unsigned char>>> m_res;
    
    std::queue<std::vector<uchar>> m_frame_buffers;

    boost::asio::steady_timer m_timer;

    const std::chrono::nanoseconds m_frame_pause = std::chrono::nanoseconds(35000000);

    statistics m_statistics;

    routing m_routing;

    std::unique_ptr<frame_reader> m_frame_reader;

    const std::string m_frame_boundary = "frame";

public:
    // Take ownership of the stream
    session(boost::asio::io_context& ioc,
        boost::asio::ip::tcp::socket socket,
        const std::string& base_folder,
        std::unique_ptr<InferenceContext>&& inference_context)
        : m_socket(std::move(socket)),
        m_strand(m_socket.get_executor()),
        m_timer(ioc),
        m_routing(std::map<std::string, std::string>{{"base_dir", base_folder}}),
        m_inference_context(std::move(inference_context))
    {
    }

    void run();

private:

    void do_read();

    void on_read(
        boost::system::error_code ec,
        std::size_t bytes_transferred);

    void on_write(
        boost::system::error_code ec,
        std::size_t bytes_transferred);

    void on_timer(const boost::system::error_code& error);

    void do_close();

    void process_frame();
};

#endif