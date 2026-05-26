cmake_minimum_required(VERSION 3.8)
project(eskin_ros2_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Force use of system Python (avoid Conda Python interfering with ROS 2)
set(Python3_EXECUTABLE /usr/bin/python3 CACHE FILEPATH "System Python3" FORCE)

# Ensure ROS 2 Python packages are findable even without ROS 2 sourced (e.g. VS Code CMake Tools)
set(ENV{PYTHONPATH} "/opt/ros/jazzy/lib/python3.12/site-packages:$ENV{PYTHONPATH}")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Eskin SDK library
set(ESKIN_SDK_DIR "/home/lenn/Workspace/eskin-finger-sdk" CACHE PATH "Path to eskin-finger-sdk")
add_library(eskin_finger_sdk SHARED IMPORTED)
set_target_properties(eskin_finger_sdk PROPERTIES
  IMPORTED_LOCATION "${ESKIN_SDK_DIR}/target/release/libeskin_finger_sdk.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(eskin_publisher src/eskin_publisher.cpp)
ament_target_dependencies(eskin_publisher rclcpp std_msgs)
target_link_libraries(eskin_publisher eskin_finger_sdk pthread)

add_executable(eskin_subscriber src/eskin_subscriber.cpp)
ament_target_dependencies(eskin_subscriber rclcpp std_msgs)

install(TARGETS
  eskin_publisher
  eskin_subscriber
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
