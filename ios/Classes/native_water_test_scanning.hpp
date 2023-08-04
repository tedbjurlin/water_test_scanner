#include <opencv2/opencv.hpp>

// avoids multi imports
#ifndef NATIVE_WATER_TEST_SCANNING_HPP
#define NATIVE_WATER_TEST_SCANNING_HPP

// This is the header file that is exposed to dart. It defines the function type signatures of all
// of the shared functions. See native_water_test_scanning.cpp for the actual functions.

// This struct is used to store the individual colors, along with the values calculated for them.
// The idx is the index on the strip. It is not used.
struct ColorOutput
{
    int32_t idx;
    int32_t red;
    int32_t green;
    int32_t blue;
    double value;
};

// This is the main struct returned that contains each of the colors on the strip.
// Colors are listed as individual items rather than an array because the dart ffi
// has dificulty decoding arrays.
// It also contains the size of the buffer that was allocated to the output image.
// This image is not returned here, but rather generated in place at a pointer supplied to
// the function.
// The exit code indicates the result of the function:
// 0: success
// 5: the input image was empty
// 3: the scanner could not find a color key
// 1: the scanner could not find a test strip
struct DetectionResult
{
    ColorOutput *color1;
    ColorOutput *color2;
    ColorOutput *color3;
    ColorOutput *color4;
    ColorOutput *color5;
    ColorOutput *color6;
    ColorOutput *color7;
    ColorOutput *color8;
    ColorOutput *color9;
    ColorOutput *color10;
    ColorOutput *color11;
    ColorOutput *color12;
    ColorOutput *color13;
    ColorOutput *color14;
    ColorOutput *color15;
    ColorOutput *color16;

    int32_t size;

    int32_t exitCode;
    
};

// the extern "C" ensures that the function is available in the linker. It is actually only needed
// on native_detect_colors because that is the only function called from the dart.
extern "C"
struct ColorOutput createColorOutput(cv::Scalar color, int idx, double value);

extern "C"
struct DetectionResult *create_detection_result(std::vector<ColorOutput> array, int size, int exit_code);

// Marking a function as extern "C" prevents the functions name from being mangled by the C++ linker.
// If this is not done, the dart ffi will not be able to find it.
// The attributes indicated here prevent the linker from throwing away the function signature after linking.
// This way, the function name is visible to outside applications.
extern "C" __attribute__((visibility("default"))) __attribute__((used))
struct DetectionResult *native_detect_colors(char *str, uchar *key, int length, uchar **encodedImage);

#endif