cmake_minimum_required(VERSION 3.10.0)
project(kalman)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FLAGS "-Wall -Wextra -Wunused-variable -Wno-comment")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

find_package(Qt5 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    )

find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

find_package(Eigen3 3.3 REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME}
    main.cpp
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE Qt5::Core
    PRIVATE Qt5::Gui
    PRIVATE Qt5::Widgets
    PRIVATE Threads::Threads
    PRIVATE ${OpenCV_LIBS}
    PRIVATE Boost::boost
    PRIVATE Eigen3::Eigen
    )
