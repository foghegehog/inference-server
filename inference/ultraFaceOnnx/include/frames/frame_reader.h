#ifndef FRAME_READER_H
#define FRAME_READER_H

#include <opencv2/imgproc/imgproc.hpp>

class frame_reader
{
public:
    virtual bool is_finished() = 0;
    virtual cv::Mat read_frame() = 0;
    virtual ~frame_reader() = default;
};

#endif