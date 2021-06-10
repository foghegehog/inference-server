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

    float get_box_area() const
    {
        return (mBox[2] - mBox[0]) * (mBox[3] - mBox[1]);
    }

};

struct ScoreDescendingCompare
{
    bool operator()(const Detection& left, const Detection& right)
    {
        return left.mScore > right.mScore;
    }
};

#endif