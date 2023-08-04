#include "native_water_test_scanning.hpp"
#include "water_test_scanner.hpp"
#include <stdlib.h>
#include <opencv2/opencv.hpp>

// The createColorOutput function creates a ColorOuptut struct. Nice and straightforward.
struct ColorOutput createColorOutput(cv::Scalar color, int idx, double value)
{
    struct ColorOutput coloroutput;
    coloroutput.blue = static_cast<int32_t>(color[0]);
    coloroutput.green = static_cast<int32_t>(color[1]);
    coloroutput.red = static_cast<int32_t>(color[2]);
    coloroutput.idx = static_cast<int32_t>(idx);
    coloroutput.value = value;
    return coloroutput;
}

// This doesn't need to be marked extern "C".
// This creates a pointer to a detection result. This is important because this kind of deliberate
// memory allocation is what allows us to avoid nasty errors when sending the data to dart.
extern "C"
struct DetectionResult *create_detection_result(vector<ColorOutput> array, int size, int exit_code)
{
    // By deliberatly allocating the memory for this struct, we can avoid the memory being freed before we
    // read it in dart.
    struct DetectionResult *detectionResult = (struct DetectionResult *)malloc(sizeof(struct DetectionResult));
    detectionResult->color1 = &array[0];
    detectionResult->color2 = &array[1];
    detectionResult->color3 = &array[2];
    detectionResult->color4 = &array[3];
    detectionResult->color5 = &array[4];
    detectionResult->color6 = &array[5];
    detectionResult->color7 = &array[6];
    detectionResult->color8 = &array[7];
    detectionResult->color9 = &array[8];
    detectionResult->color10 = &array[9];
    detectionResult->color11 = &array[10];
    detectionResult->color12 = &array[11];
    detectionResult->color13 = &array[12];
    detectionResult->color14 = &array[13];
    detectionResult->color15 = &array[14];
    detectionResult->color16 = &array[15];
    detectionResult->size = size;
    detectionResult->exitCode = exit_code;
    return detectionResult;
}

// This function is sets up the data properly to be fun in our detect_colors function.
// It takes a string - the path to the image captured by the camera, an array of unsigned bytes - 
// the binary data of an the reference image of the color correction card, encoded as a jpeg,
// the length of the color correction card bytelist, and a pointer to an array of bytes - the output image.
// 
// The array at encodedImage pointer is undefined. This allows us to allocate the memory for this array in
// the C++ code and write our output image to it, while Dart still knows where to look for it.
struct DetectionResult *native_detect_colors(char *str, uchar *key, int length, uchar **encodedImage)
{
    cv::Mat mat = cv::imread(str);

    // The color correction card is decoded as a color image. We don't care about the alpha channel.
    // The current key used is a jpeg, but any common image encoded can be used for the color correction key.
    // OpenCV will decode it regardless.
    // The bytelist has to be converted to a single channel OpenCV Matrix to be decoded.
    cv::Mat ref = cv::imdecode(cv::Mat(Size(1, length), CV_8UC1, key), cv::IMREAD_COLOR);

    // This just checks to make sure that both the image and the color card are valid images.
    if (mat.size().width == 0 || mat.size().height == 0 || ref.size().width == 0 || ref.size().width == 0) {

        vector<ColorOutput> out(16);

        for (int i = 0; i < 16; i++) {
            out[i] = createColorOutput(cv::Scalar(), i, 5);
        }

        vector<uchar> buf;

        cv::imencode(".png", mat, buf);

        // I will talk about this at the end of water_test_scanner.cpp
        *encodedImage = (unsigned char *) malloc(buf.size());
        for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];

        return create_detection_result(out, buf.size(), 2);
    }

    vector<ColorOutput> colors(16);
    DetectionResult *out = TestScanner::detect_colors(mat, ref, colors, encodedImage);

    return out;
}