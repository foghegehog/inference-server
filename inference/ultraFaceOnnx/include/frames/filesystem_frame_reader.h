#ifndef FILESYSTEM_FRAME_READER_H
#define FILESYSTEM_FRAME_READER_H

#include "files_iterator.h"
#include "frame_reader.h"

class filesystem_frame_reader : public frame_reader
{
public:
    filesystem_frame_reader(const std::string& path, const std::string& extention)
        :m_files_iterator(path, extention)
    {}

    ~filesystem_frame_reader() override {}

    bool is_finished() override;
    
    cv::Mat read_frame() override;

private:
    files_iterator m_files_iterator;
};

#endif