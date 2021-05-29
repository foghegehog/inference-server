#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map> 
#include <memory>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

std::mutex multipart_mutex;
std::map<std::string, std::string> multiparts; 

//------------------------------------------------------------------------------

// Return a reasonable mime type based on the extension of a file.
beast::string_view
mime_type(beast::string_view path)
{
    using beast::iequals;
    auto const ext = [&path]
    {
        auto const pos = path.rfind(".");
        if(pos == beast::string_view::npos)
            return beast::string_view{};
        return path.substr(pos);
    }();
    if(iequals(ext, ".htm"))  return "text/html";
    if(iequals(ext, ".html")) return "text/html";
    if(iequals(ext, ".php"))  return "text/html";
    if(iequals(ext, ".css"))  return "text/css";
    if(iequals(ext, ".txt"))  return "text/plain";
    if(iequals(ext, ".js"))   return "application/javascript";
    if(iequals(ext, ".json")) return "application/json";
    if(iequals(ext, ".xml"))  return "application/xml";
    if(iequals(ext, ".swf"))  return "application/x-shockwave-flash";
    if(iequals(ext, ".flv"))  return "video/x-flv";
    if(iequals(ext, ".png"))  return "image/png";
    if(iequals(ext, ".jpe"))  return "image/jpeg";
    if(iequals(ext, ".jpeg")) return "image/jpeg";
    if(iequals(ext, ".jpg"))  return "image/jpeg";
    if(iequals(ext, ".gif"))  return "image/gif";
    if(iequals(ext, ".bmp"))  return "image/bmp";
    if(iequals(ext, ".ico"))  return "image/vnd.microsoft.icon";
    if(iequals(ext, ".tiff")) return "image/tiff";
    if(iequals(ext, ".tif"))  return "image/tiff";
    if(iequals(ext, ".svg"))  return "image/svg+xml";
    if(iequals(ext, ".svgz")) return "image/svg+xml";
    return "application/text";
}

// Append an HTTP rel-path to a local filesystem path.
// The returned path is normalized for the platform.
std::string
path_cat(
    beast::string_view base,
    beast::string_view path)
{
	if (base.empty())
		return std::string(path);
    std::string result(base);
    char constexpr path_separator = '/';
    if(result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
    return result;
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

/*template<class Body, class Allocator, class Stream>
void handle_request(
    beast::string_view doc_root,
    http::request<Body, http::basic_fields<Allocator>>&& req,
    Stream& response_stream,
    bool& should_close, 
    beast::error_code& error)
{
    // Returns a bad request response
    auto const bad_request =
    [&req](beast::string_view why)
    {
        http::response<http::string_body> res{http::status::bad_request, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = std::string(why);
        res.prepare_payload();
        return res;
    };

    // Returns a not found response
    auto const not_found =
    [&req](beast::string_view target)
    {
        http::response<http::string_body> res{http::status::not_found, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "The resource '" + std::string(target) + "' was not found.";
        res.prepare_payload();
        return res;
    };

    // Returns a server error response
    auto const server_error =
    [&req](beast::string_view what)
    {
        http::response<http::string_body> res{http::status::internal_server_error, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "An error occurred: '" + std::string(what) + "'";
        res.prepare_payload();
        return res;
    };

    // Make sure we can handle the method
    if( req.method() != http::verb::get &&
        req.method() != http::verb::head)
    {
        return write_response(bad_request("Unknown HTTP-method"), response_stream, should_close, error);
    }

    // Request path must be absolute and not contain "..".
    if( req.target().empty() ||
        req.target()[0] != '/' ||
        req.target().find("..") != beast::string_view::npos)
    {
        write_response(bad_request("Illegal request-target"), response_stream, should_close, error);
    }

    // Build the path to the requested file
    std::string path = path_cat(doc_root, req.target());
    if(req.target().back() == '/')
        path.append("index.html");

    // Attempt to open the file
    beast::error_code ec;
    http::file_body::value_type body;
    body.open(path.c_str(), beast::file_mode::scan, ec);

    // Handle the case where the file doesn't exist
    if(ec == beast::errc::no_such_file_or_directory)
    { 
        write_response(not_found(req.target()), response_stream, should_close, error);
    }

    // Handle an unknown error
    if(ec)
    {
        write_response(server_error(ec.message()), response_stream, should_close, error);
    }

    // Cache the size since we need it after the move
    auto const size = body.size();

    // Respond to HEAD request
    if(req.method() == http::verb::head)
    {
        http::response<http::empty_body> res{http::status::ok, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, mime_type(path));
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        write_response(std::move(res), response_stream, should_close, error);
    }

    // Respond to GET request
    http::response<http::file_body> res{
        std::piecewise_construct,
        std::make_tuple(std::move(body)),
        std::make_tuple(http::status::ok, req.version())};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, mime_type(path));
    res.content_length(size);
    res.keep_alive(req.keep_alive());
    write_response(std::move(res), response_stream, should_close, error);
}*/

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
        filename_stream <<  "./corridor/" << std::setw(8) << std::setfill('0') << std::to_string(f) << ".jpg";  

        auto filename = filename_stream.str();
        cv::Mat frame = cv::imread(filename);
        //std::cout << filename;
        if (frame.empty())
        {
            std::cout << " is empty." << std::endl;
            continue;
        }
        else
        {
            std::cout << std::endl;
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

//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
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
}