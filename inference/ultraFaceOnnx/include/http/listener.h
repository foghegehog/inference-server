#ifndef LISTENER_H
#define LISTENER_H

#include "../inference/ultraFaceOnnx.h"

#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>

#include <memory>


// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
    boost::asio::ip::tcp::acceptor m_acceptor;
    boost::asio::ip::tcp::socket m_socket;
    boost::asio::io_context& m_ioc;
    UltraFaceOnnxEngine& m_inference_engine;

public:
    listener(boost::asio::io_context& ioc,
        boost::asio::ip::tcp::endpoint endpoint,
        UltraFaceOnnxEngine& inferenceEngine);

    // Start accepting incoming connections
    void run();

private:
    void do_accept();

    void on_accept(boost::beast::error_code ec);
};

#endif