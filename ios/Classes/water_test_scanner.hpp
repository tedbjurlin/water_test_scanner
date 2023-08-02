#include <opencv2/opencv.hpp>
#include "native_water_test_scanning.hpp"

using namespace cv;
using namespace std;

struct ColorCardResult {
    bool success;
    vector<vector<Point2f>> markerCorners;
    vector<int> markerIds;
};

class TestScanner
{
    public:
    static DetectionResult *detect_colors(Mat img, Mat ref, vector<ColorOutput> colors, uchar **encodedImage);

    private:
    static vector<float> cumsum(Mat src);
    static vector<float> calculate_cdf(Mat histogram);
    static Mat match_histograms(Mat input_image, Mat base_ref, Mat current_ref);
    static int searchResult(vector<int> arr, int k);
    static struct ColorCardResult find_color_card(Mat img, Mat outputImg);
    static Scalar ScalarBGR2Lab(uchar B, uchar G, uchar R);
    static double getClosest(Scalar value, vector<Scalar> key, vector<double> values);
    static bool findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices, double height);
};
