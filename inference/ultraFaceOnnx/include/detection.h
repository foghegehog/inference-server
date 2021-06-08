#ifndef DETECTION_H
#define DETECTION_H

#include <array>
#include <algorithm>

struct Detection
{
    float mScore;
    constexpr static const int mNumCorners = 4; 
    std::array<float, mNumCorners> mBox;

    Detection(float score, std::array<float, mNumCorners>&& box)
        : mScore(score), mBox(box)
    {
    }
};

#endif