#ifndef LISTENER_H
#define LISTENER_H

#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <memory>


// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
    boost::asio::ip::tcp::acceptor m_acceptor;
    tcp::socket m_socket;

public:
    listener(boost::asio::io_context& ioc,
        tcp::endpoint endpoint);

    // Start accepting incoming connections
    void run();

private:
    void do_accept();

    void on_accept(beast::error_code ec, tcp::socket socket);
};

#endif