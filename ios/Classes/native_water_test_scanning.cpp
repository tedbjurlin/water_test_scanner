#include "native_water_test_scanning.hpp"
#include "water_test_scanner.hpp"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>

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

extern "C"
struct DetectionResult *create_detection_result(std::vector<ColorOutput> array, int32_t width, int32_t height, int32_t buffer_size, uint8_t *image, int exit_code)
{
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
    detectionResult->width = width;
    detectionResult->height = height;
    detectionResult->buffer_size = buffer_size;
    detectionResult->image = image;
    detectionResult->exitCode = exit_code;
    return detectionResult;
}

int encodeIm(int width, int height, uchar *rawBytes, uchar **encodedOutput) {
    Mat mat = cv::Mat(Size(height, width + width / 2), CV_8UC1, rawBytes);
    cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_I420);
    vector<uchar> buf;
    cv::imencode(".jpeg", mat, buf);
    *encodedOutput = (unsigned char *) malloc(buf.size());
    for (int i = 0; i < buf.size(); i++) (*encodedOutput)[i] = buf[i];
    return (int) buf.size();
}

struct DetectionResult *native_detect_colors(uint8_t *img_bytes, int32_t *iwidth, int32_t *iheight, uint8_t *key_bytes, int32_t *width, int32_t *height, bool *isYUV)
{
    cv::Mat mat;
    if (*isYUV) {
        mat = cv::Mat(Size(*iheight, *iwidth + *iwidth / 2), CV_8UC1, img_bytes);
        cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_I420);
    } else {
        mat = cv::Mat(Size(*iheight, *iwidth), CV_8UC4, img_bytes);
        cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
    }

    cv::Mat ref = cv::Mat(Size(*width, *height), CV_8UC3, key_bytes);

    if (mat.size().width == 0 || mat.size().height == 0) {

        vector<ColorOutput> out(16);

        for (int i = 0; i < 16; i++) {
            out[i] = createColorOutput(cv::Scalar(), i, 0);
        }

        return create_detection_result(out, *iheight, *iwidth, 0, {}, 2);
    }

    vector<ColorOutput> out(16);

    for (int i = 0; i < 16; i++) {
        out[i] = createColorOutput(cv::Scalar(), i, 0);
    }

    vector<uchar> buffer;

    cv::imencode(".png", mat, buffer);

    uchar * image = &buffer[0];

    // uchar *image = mat.isContinuous()? mat.data: mat.clone().data;

    return create_detection_result(out, *iheight, *iwidth, buffer.size(), image, 5);

    // vector<ColorOutput> colors(16);
    // return TestScanner::detect_colors(mat, ref, colors);
}