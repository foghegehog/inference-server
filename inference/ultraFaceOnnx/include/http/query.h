#ifndef QUERY_H
#define QUERY_H

#include <string>
#include <vector>
#include <utility>

struct query
{
    query(const std::string& query_string);
    std::vector<std::string> m_path;
    std::vector<std::pair<std::string, std::string>> m_parameters;
};

#endif