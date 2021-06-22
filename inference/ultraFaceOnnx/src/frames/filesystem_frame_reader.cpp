#include "../../include/frames/filesystem_frame_reader.h"

#include <opencv2/imgcodecs.hpp>

bool filesystem_frame_reader::is_finished()
{
    return m_files_iterator.is_finished();
}

cv::Mat filesystem_frame_reader::read_frame()
{
    auto path = m_files_iterator.get_file_path();
    m_files_iterator.move_next();
    return cv::imread(path);
}