#include "../../include/http/lib.h"

#include <iostream>

void fail(boost::system::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

void log(char const* message)
{
    std::cerr << message << std::endl;
}