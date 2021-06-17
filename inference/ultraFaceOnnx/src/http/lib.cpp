#include "../../include/http/lib.h"

#include <iostream>
#include <string>

void fail(boost::system::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

void log(char const* message)
{
    std::cerr << message << std::endl;
}

void log(std::string message)
{
    std::cerr << message << std::endl;
}

std::time_t get_time()
{
    using clock = std::chrono::system_clock;

    auto time_point = clock::now();
    return clock::to_time_t(time_point);
}