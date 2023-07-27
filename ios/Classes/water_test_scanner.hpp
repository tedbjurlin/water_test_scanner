#include <opencv2/opencv.hpp>
#include "native_water_test_scanning.hpp"

using namespace cv;
using namespace std;

class TestScanner
{
    public:
    static Result detect_colors( Mat img, vector<ColorOutput> colors);

    private:
    static bool findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices);
};
