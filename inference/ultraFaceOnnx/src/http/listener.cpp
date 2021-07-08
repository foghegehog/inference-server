#include "http/lib.h"
#include "http/listener.h"
#include "http/session.h"
#include "inference/ultraFaceOnnx.h"

#include <boost/beast/http.hpp>
#include <boost/asio/strand.hpp>
#include <iostream>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

listener::listener(
    boost::asio::io_context& ioc,
    tcp::endpoint endpoint,
    const std::string& base_dir,
    UltraFaceOnnxEngine& inferenceEngine)
    :m_acceptor(ioc),
    m_socket(ioc),
    m_ioc(ioc),
    m_base_dir(base_dir),
    m_inference_engine(inferenceEngine)
{
    beast::error_code ec;

    // Open the acceptor
    m_acceptor.open(endpoint.protocol(), ec);
    if (ec)
    {
        fail(ec, "open");
        return;
    }

    // Allow address reuse
    m_acceptor.set_option(net::socket_base::reuse_address(true), ec);
    if (ec)
    {
        fail(ec, "set_option");
        return;
    }

    // Bind to the server address
    m_acceptor.bind(endpoint, ec);
    if (ec)
    {
        fail(ec, "bind");
        return;
    }

    // Start listening for connections
    m_acceptor.listen(net::socket_base::max_listen_connections, ec);
    if (ec)
    {
        fail(ec, "listen");
        return;
    }
}

void listener::run()
{
    if(!m_acceptor.is_open())
    {
        return;
    }        
    
    do_accept();
}

void listener::do_accept()
{
    log("Started to accept connections.");
    // The new connection gets its own strand
    m_acceptor.async_accept(
        m_socket,
        std::bind(
            &listener::on_accept,
            shared_from_this(),
            std::placeholders::_1));
}

void listener::on_accept(beast::error_code ec)
{
    if(ec)
    {
        fail(ec, "accept");
    }
    else
    {
        // Create the session and run it
        std::make_shared<session>(
            m_ioc,
            std::move(m_socket),
            m_base_dir,
            m_inference_engine.get_inference_context())->run();
    }

    // Accept another connection
    do_accept();
}