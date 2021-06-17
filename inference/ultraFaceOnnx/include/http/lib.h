#ifndef LIB_H
#define LIB_H

#include <boost/beast/core/error.hpp>
#include <chrono>

void fail(boost::system::error_code ec, char const* what);

void log(char const* message);

void log(std::string message);

std::time_t get_time();



#endif