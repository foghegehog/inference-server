#ifndef ROUTING_H
#define ROUTING_H

#include "query.h"
#include "../frames/frame_reader.h"
#include "../frames/filesystem_frame_reader.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

class routing
{
public:
    routing(std::map<std::string, std::string> params);

    std::unique_ptr<frame_reader> create_reader(const std::string& type, const query& q);

private:
    std::map<
        std::string,
        std::function<std::unique_ptr<frame_reader>(const query&)>> m_routes;

    std::map<std::string, std::string> m_params;
};
#endif