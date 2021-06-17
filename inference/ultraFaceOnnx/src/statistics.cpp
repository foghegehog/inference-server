#include "../include/statistics.h"

void statistics::update_avg_processing(double duration)
{
    m_avg_processing_time = ((m_avg_processing_time * m_count) + duration) / (m_count + 1);
    ++m_count;
}

double statistics::get_avg_processing_time() const
{
    return m_avg_processing_time;
}