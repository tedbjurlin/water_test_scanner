# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

include_directories(../include)

add_library(lib_opencv SHARED IMPORTED)

set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)

set(WATER_TEST_SCANNER_DIR "../ios/Classes")

set(SOURCES

    ${WATER_TEST_SCANNER_DIR}/native_water_test_scanning.cpp

    ${WATER_TEST_SCANNER_DIR}/water_test_scanner.cpp

)

add_library(native_water_test_scanning SHARED ${SOURCES})

target_link_libraries(native_water_test_scanning lib_opencv)

find_library( # Defines the name of the path variable that stores the
              # location of the NDK library.
              log-lib

              # Specifies the name of the NDK library that
              # CMake needs to locate.
              log )

# Links your native library against one or more other native libraries.
target_link_libraries( # Specifies the target library.
                       native_water_test_scanning

                       # Links the log library to the target library.
                       ${log-lib} )