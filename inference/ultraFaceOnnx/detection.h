#ifndef DETECTION_H
#define DETECTION_H

#include <array>

struct Detection
{
    float mScore;
    std::array<float, 4> mBox;

    Detection(float score, const std::array<float, 4> box)
        :mScore(score), mBox(box)
    {
    }
};

#endif