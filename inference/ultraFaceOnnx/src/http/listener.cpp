#include "../../include/http/lib.h"
#include "../../include/http/listener.h"
#include "../../include/http/session.h"

#include <boost/beast/http.hpp>
#include <boost/asio/strand.hpp>
#include <iostream>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

listener::listener(boost::asio::io_context& ioc,
        tcp::endpoint endpoint)
    :m_acceptor(ioc),
    m_socket(ioc)
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
    // The new connection gets its own strand
    m_acceptor.async_accept(
        m_socket,
        std::bind(
            &listener::on_accept,
            shared_from_this(),
            std::placeholders::_1));
}

void listener::on_accept(beast::error_code ec, tcp::socket socket)
{
    if(ec)
    {
        fail(ec, "accept");
    }
    else
    {
        // Create the session and run it
        std::make_shared<session>(std::move(m_socket))->run();
    }

    // Accept another connection
    do_accept();
}