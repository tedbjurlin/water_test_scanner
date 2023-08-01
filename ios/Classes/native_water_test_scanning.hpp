#include <opencv2/opencv.hpp>

#ifndef NATIVE_WATER_TEST_SCANNING_HPP
#define NATIVE_WATER_TEST_SCANNING_HPP

struct ColorOutput
{
    int32_t idx;
    int32_t red;
    int32_t green;
    int32_t blue;
    double value;
};

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

    int32_t width;

    int32_t height;

    int32_t buffer_size;

    uint8_t * image;

    int32_t exitCode;
    
};

extern "C"
struct ColorOutput createColorOutput(cv::Scalar color, int idx, double value);

extern "C"
struct DetectionResult *create_detection_result(std::vector<ColorOutput> array, int32_t width, int32_t height, int32_t buffer_size, uint8_t *image, int exit_code);

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int encodeIm(int width, int height, uchar *rawBytes, uchar **encodedOutput);

extern "C" __attribute__((visibility("default"))) __attribute__((used))
struct DetectionResult *native_detect_colors(uint8_t *img_bytes, int32_t *iwidth, int32_t *iheight, uint8_t *key_bytes, int32_t *width, int32_t *height, bool *isYUV);

#endif