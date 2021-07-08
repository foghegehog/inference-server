#include "http/query.h"

query::query(const std::string& query_string)
{
    if (query_string.empty())
    {
        return;
    }

    const auto path_delimeter = '/';
    auto start = query_string[0] != path_delimeter ? 0 : 1; 

    for (auto end = query_string.find(path_delimeter, start);
        (start < query_string.size()) && (end != std::string::npos);
        end = query_string.find(path_delimeter, start))
    {
        m_path.push_back(query_string.substr(start, end - start));
        start = end + 1;
    }

    if (start >= query_string.size())
    {
        return;
    }

    const auto query_delimeter = '?';
    auto end = query_string.find(query_delimeter, start);
    if (end == std::string::npos)
    {
        m_path.push_back(query_string.substr(start));
        return;
    }
    else
    {
        m_path.push_back(query_string.substr(start, end - start));
    } 

    start = end + 1;
    const auto params_delimeter = '&';
    const auto key_value_delimeter = '=';

    for (end = query_string.find(params_delimeter, start);
        (start < query_string.size());
        end = query_string.find(params_delimeter, start))
    {
        if (end == std::string::npos)
        {
            end = query_string.size();
        }

        auto sep_pos = query_string.find(key_value_delimeter, start);
        m_parameters.emplace_back(
            query_string.substr(start, sep_pos - start),
            query_string.substr(sep_pos + 1, end - sep_pos));

        start = end + 1;
    }
}