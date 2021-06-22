#ifndef FILES_ITERATOR_H
#define FILES_ITERATOR_H

#include <boost/filesystem.hpp>
#include <vector>

class files_iterator
{
public:
    files_iterator(const std::string& path, const std::string& extention);

    bool is_finished() const;

    bool move_next();

    std::string get_file_path() const;

private:
    std::vector<std::string> m_paths_sorted;
    std::vector<std::string>::iterator m_current;
};


#endif 