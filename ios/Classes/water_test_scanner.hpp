#include <opencv2/opencv.hpp>
#include "native_water_test_scanning.hpp"

using namespace cv;
using namespace std;

class TestScanner
{
    public:
    static DetectionResult *detect_colors( Mat img, Mat ref, vector<ColorOutput> colors);

    private:
    static bool findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices, double height);
    static vector<float> cumsum(Mat src);
    static vector<float> calculate_cdf(Mat histogram);
    static Mat match_histograms(Mat input_image, Mat base_ref, Mat current_ref);
    static int searchResult(vector<int> arr, int k);
    static bool find_color_card(Mat img, Mat outputImg, vector<vector<Point2f>> markerCorners, vector<int> markerIds);
    static Scalar ScalarBGR2Lab(uchar B, uchar G, uchar R);
};
