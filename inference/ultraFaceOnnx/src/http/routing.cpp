#include "../../include/http/routing.h"

#include <boost/filesystem.hpp>

routing::routing(std::map<std::string, std::string> params)
    :m_params(params)
{
    m_routes["filesystem"] = [this](const query& q)
    {
        boost::filesystem::path path(m_params["base_dir"]);
        for (auto subdir = q.m_path.cbegin() + 1; subdir != q.m_path.cend(); ++subdir)
        {
            path /= *subdir;
        }

        std::string extention = ".jpg";
        for(const auto& pair: q.m_parameters)
        {
            if (pair.first == "ext")
            {
                extention = "." + pair.second;
                break;
            }
        }

        return std::unique_ptr<frame_reader>(
            new filesystem_frame_reader(path.string(), extention));
    };
}

std::unique_ptr<frame_reader> routing::create_reader(const std::string& type, const query& q)
{
    auto route = m_routes.find(type);
    if (route != m_routes.end())
    {
        return route->second(q);
    }
    else
    {
        return std::unique_ptr<frame_reader>(nullptr);
    }
}