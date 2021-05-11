cmake_minimum_required(VERSION 3.2)

project(http_server VERSION 0.0.1)

find_package(Boost COMPONENTS thread system REQUIRED)

add_executable (http-server
    ${BOOST_BEAST_FILES}
    http_server.cpp
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set_target_properties(http-server PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(http-server PROPERTIES
    INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR}
)

target_link_libraries(http-server PRIVATE
    ${Boost_LIBRARIES}
)

if (MSVC)
    target_compile_options(http-server PRIVATE
        /W4
    )
else ()
    target_compile_options(http-server PRIVATE
        -Wall -Wextra -pedantic -Werror
    )
endif()
