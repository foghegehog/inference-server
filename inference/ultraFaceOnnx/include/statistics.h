#ifndef STATISTICS_H
#define STATISTICS_H

#include <chrono> 

class statistics
{
public:
    void update_avg_processing(double duration);
    double get_avg_processing_time() const;
private:
    double m_avg_processing_time = 0.0;
    long m_count = 0;
};

#endif