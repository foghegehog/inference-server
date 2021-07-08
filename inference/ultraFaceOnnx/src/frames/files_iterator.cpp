#include "frames/files_iterator.h"

#include <iostream>

files_iterator::files_iterator(const std::string& path, const std::string& extention)
{
    using namespace boost::filesystem;

    directory_iterator end_iterator;

    for (auto current = directory_iterator(path); current != end_iterator; ++current)
    {
        auto path = current->path();
        if (boost::filesystem::is_regular_file(path.string())
            && (path.extension() == extention))
        {
            m_paths_sorted.push_back(path.string());
        } 
    }

    std::sort(m_paths_sorted.begin(), m_paths_sorted.end());

    m_current = m_paths_sorted.begin();
}

bool files_iterator::is_finished() const
{
    return m_current == m_paths_sorted.end();
}

bool files_iterator::move_next()
{
    ++m_current;
}

std::string files_iterator::get_file_path() const
{
    return *m_current;
}