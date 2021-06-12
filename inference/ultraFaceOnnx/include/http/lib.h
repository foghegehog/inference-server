#ifndef LIB_H
#define LIB_H

#include <boost/beast/core/error.hpp>

void fail(boost::system::error_code ec, char const* what);

#endif