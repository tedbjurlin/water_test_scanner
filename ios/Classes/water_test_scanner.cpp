#include "water_test_scanner.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <android/log.h>

using namespace cv;
using namespace std;
using namespace cv::aruco;

vector<float> TestScanner::cumsum(Mat src) {
    vector<float> sum(src.size().height);
    sum[0] = src.at<float>(0);
    for (unsigned int i = 1; i < src.size().height; i++) {
        sum[i] = src.at<float>(i) + sum[i-1];
    }

    return sum;
}

vector<float> TestScanner::calculate_cdf(Mat histogram) {
    vector<float> cdf = cumsum(histogram);


    for (unsigned int i = 0; i < cdf.size(); i++)
    {
        cdf[i] = cdf[i] / cdf[cdf.size() - 1];
    }

    return cdf;
}

vector<int> calculate_lookup(vector<float> src, vector<float> ref)
{
    vector<int> lookup_table(256);
    int lookup_value = 0;
    for (int i = 0; i < src.size(); i++)
    {
        for (int j = 0; j < ref.size(); j++) {
            if (ref[j] >= src[i])
            {
                lookup_value = j;
                break;
            }
        }
        lookup_table[i] = lookup_value;
    }
    return lookup_table;
}

Mat TestScanner::match_histograms(Mat input_image, Mat base_ref, Mat current_ref)
{
    Mat src_bands[3], base_bands[3], curr_bands[3];
    split(input_image, src_bands);
    split(base_ref, base_bands);
    split(current_ref, curr_bands);
    
    int histSize = 256;

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    
    bool uniform = true, accumulate =  false;

    Mat base_blue_hist, base_green_hist, base_red_hist, curr_blue_hist, curr_green_hist, curr_red_hist;
    calcHist( &base_bands[0], 1, 0, Mat(), base_blue_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &base_bands[1], 1, 0, Mat(), base_green_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &base_bands[2], 1, 0, Mat(), base_red_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[0], 1, 0, Mat(), curr_blue_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[1], 1, 0, Mat(), curr_green_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[2], 1, 0, Mat(), curr_red_hist, 1, &histSize, histRange, uniform, accumulate );
 
    vector<float> src_cdf_blue = calculate_cdf(base_blue_hist);
    vector<float> src_cdf_green = calculate_cdf(base_green_hist);
    vector<float> src_cdf_red = calculate_cdf(base_red_hist);
    vector<float> ref_cdf_blue = calculate_cdf(curr_blue_hist);
    vector<float> ref_cdf_green = calculate_cdf(curr_green_hist);
    vector<float> ref_cdf_red = calculate_cdf(curr_red_hist);
 
    vector<int> blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue);
    vector<int> green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green);
    vector<int> red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red);

    vector<Mat> res_bands {src_bands[0].clone(), src_bands[1].clone(), src_bands[2].clone()};
    LUT(src_bands[0], blue_lookup_table, res_bands[0]);
    LUT(src_bands[1], green_lookup_table, res_bands[1]);
    LUT(src_bands[2], red_lookup_table, res_bands[2]);

    Mat res;
    merge(res_bands, res);

    convertScaleAbs(res, res);

    return res;
}

int TestScanner::searchResult(vector<int> arr, int k){
    vector<int>::iterator it;
    it = find(arr.begin(), arr.end(), k);
    if(it != arr.end())
        return (it - arr.begin());
    else
        return -1;
}

struct ColorCardResult TestScanner::find_color_card(Mat img, Mat outputImg)
{
    Dictionary arucoDict = getPredefinedDictionary(DICT_4X4_250);

    DetectorParameters parameters = DetectorParameters();

    vector<vector< Point2f>> rejectedCandidates, markerCorners;

    vector<int> markerIds;

    ArucoDetector detector(arucoDict, parameters);

    detector.detectMarkers(img, markerCorners, markerIds, rejectedCandidates);

    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "Rejected candidates length: %lu", rejectedCandidates.size());
    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "markerIds length: %lu", markerIds.size());

    int topLeftIdx = searchResult(markerIds, 23);
    int topRightIdx = searchResult(markerIds, 42);
    int bottomRightIdx = searchResult(markerIds, 15);
    int bottomLeftIdx = searchResult(markerIds, 67);

    Point2f topLeft;
    Point2f topRight;
    Point2f bottomRight;
    Point2f bottomLeft;

    if (topLeftIdx != -1 && topRightIdx != -1 && bottomRightIdx != -1 && bottomLeftIdx != -1)
    {
        topLeft = markerCorners.at(topLeftIdx).at(0);
        topRight = markerCorners.at(topRightIdx).at(1);
        bottomRight = markerCorners.at(bottomRightIdx).at(2);
        bottomLeft = markerCorners.at(bottomLeftIdx).at(3);

        __android_log_print(ANDROID_LOG_DEBUG, "flutter", "found points");
    } else {
        ColorCardResult out;

        out.markerCorners = markerCorners;
        out.markerIds = markerIds;
        out.success = false;

        return out;
    }

    vector<Point2f> pts(4);
    vector<Point2f> dst(4);

    pts[0] = topLeft;
    pts[1] = topRight;
    pts[2] = bottomRight;
    pts[3] = bottomLeft;

    dst[0].x = 0;
    dst[0].y = 0;
    dst[1].x = 220;
    dst[1].y = 0;
    dst[2].x = 220;
    dst[2].y = 770;
    dst[3].x = 0;
    dst[3].y = 770;

    Mat pTrans;
    pTrans = getPerspectiveTransform(pts, dst);

        __android_log_print(ANDROID_LOG_DEBUG, "flutter", "calculated transform");

    Mat warped_img;
    warpPerspective(img, warped_img, pTrans, Size(220, 770));

        __android_log_print(ANDROID_LOG_DEBUG, "flutter", "warped image");

    // 220 x 770

    // clockwise: 23, 42, 15, 67

    warped_img.copyTo(outputImg);

        __android_log_print(ANDROID_LOG_DEBUG, "flutter", "copied image");

    ColorCardResult out;

    out.markerCorners = markerCorners;
    out.markerIds = markerIds;
    out.success = true;

    return out;

    
}

Scalar TestScanner::ScalarBGR2Lab(uchar B, uchar G, uchar R) {
    Mat lab;
    Mat bgr(1,1, CV_8UC3, Scalar(B, G, R));
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    return Scalar(lab.data[0], lab.data[1], lab.data[2]);
}

double TestScanner::getClosest(Scalar value, vector<Scalar> key, vector<double> values)
{
    Mat vmat = Mat::zeros(Size(1, 1), CV_32FC3);
    vmat.setTo(value);
    Mat kmat = Mat::zeros(Size(1, 1), CV_32FC3);
    kmat.setTo(key[0]);

    double dist = norm(vmat, kmat, NORM_L2);
    int idx = 0;
    for (int i = 1; i < values.size(); i++)
    {
        kmat.setTo(key[i]);
        double new_dist = norm(vmat, kmat, NORM_L2);
        if (new_dist < dist)
        {
            dist = new_dist;
            idx = i;
        }
    }
    return values[idx];
}

bool TestScanner::findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices, double height){
    
    list<RotatedRect> boxes;
    
    for(vector<Point> contour : contours)
    {
        // Compute minimal bounding box
        cv::RotatedRect box = cv::minAreaRect(Mat(contour));

        double exp = 0.037037;

        double act = box.size.aspectRatio();
        double boxheight = box.size.height;

        // other section of if  && ((2 * min(height, boxheight)) / (height + boxheight) > 0.90)

        if (((2 * min(exp, act)) / (exp + act) > 0.90))
        {
            boxes.push_back(box);
        }
    }

    if (boxes.size() == 0) {
        return false;
    }

    RotatedRect biggest_box = boxes.front();

    boxes.pop_front();

    for (RotatedRect curr_box : boxes) {
        if (curr_box.size.area() > biggest_box.size.area()) {
            biggest_box = curr_box;
        }
    }

    biggest_box.points(vertices);

    return true;
}

DetectionResult *TestScanner::detect_colors(Mat img, Mat ref, vector<ColorOutput> colors, uchar **encodedImage)
{
    Mat outputImg = Mat::zeros(Size(220, 770), img.type());


    ColorCardResult result = find_color_card(img, outputImg);

    vector<vector<Point2f>> markerCorners = result.markerCorners;
    vector<int> markerIds = result.markerIds;

    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "markerIds length: %lu", markerIds.size());
    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "markerCorners length: %lu", markerCorners.size());

    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "found color card");

    for (int i = 0; i < markerIds.size(); i++) {
        __android_log_print(ANDROID_LOG_DEBUG, "flutter", "Found id: %d", markerIds[i]);
    }

    if (!result.success)
    {
        for (int i = 0; i < 16; i++) {
            colors[i] = createColorOutput(Scalar(), i, -1.0);
        }

        vector<uchar> buf;

        imencode(".png", img, buf);

        *encodedImage = (unsigned char *) malloc(buf.size());
        for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];

        return create_detection_result(colors, buf.size(), 3);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "flutter", "passed if");

    int topLeftIdx = searchResult(markerIds, 23);
    int topRightIdx = searchResult(markerIds, 42);
    int bottomRightIdx = searchResult(markerIds, 15);
    int bottomLeftIdx = searchResult(markerIds, 67);

    Point2f topLeft = markerCorners.at(topLeftIdx).at(0);
    Point2f topRight = markerCorners.at(topRightIdx).at(1);
    Point2f bottomRight = markerCorners.at(bottomRightIdx).at(2);
    Point2f bottomLeft = markerCorners.at(bottomLeftIdx).at(3);

    vector<Point> poly_points(4);

    vector<double> src1 = {topLeft.x, topLeft.y};
    vector<double> src2 = {topRight.x, topRight.y};
    vector<double> src3 = {bottomRight.x, bottomRight.y};

    int padding = static_cast<int>(0.15 * norm(src1, src2, NORM_L2));

    poly_points[0].x = static_cast<int>(topLeft.x - padding);
    poly_points[0].y = static_cast<int>(topLeft.y - padding);
    poly_points[1].x = static_cast<int>(topRight.x + padding);
    poly_points[1].y = static_cast<int>(topRight.y - padding);
    poly_points[2].x = static_cast<int>(bottomRight.x + padding);
    poly_points[2].y = static_cast<int>(bottomRight.y + padding);
    poly_points[3].x = static_cast<int>(bottomLeft.x - padding);
    poly_points[3].y = static_cast<int>(bottomLeft.y + padding);

    Mat nokeyimg = img.clone();

    line(img, poly_points[0], poly_points[1], Scalar(0, 255, 0), 2);
    line(img, poly_points[1], poly_points[2], Scalar(0, 255, 0), 2);
    line(img, poly_points[2], poly_points[3], Scalar(0, 255, 0), 2);
    line(img, poly_points[3], poly_points[0], Scalar(0, 255, 0), 2);

    fillPoly(nokeyimg, poly_points, Scalar());

    resize(nokeyimg, nokeyimg, Size(nokeyimg.size().width / 4, nokeyimg.size().height / 4), INTER_LINEAR);

    Mat img_gray;
    cvtColor(nokeyimg, img_gray, COLOR_BGR2GRAY);

    Mat img_filtered;
    bilateralFilter(img_gray, img_filtered, 5, 40, 40);

    GaussianBlur(img_filtered, img_filtered, Size(5, 5), 0);

    Mat canny;
    Canny(img_filtered, canny, 100, 200);

    vector<vector<Point>> contours;
    findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Point2f vertices[4];    

    if (!findBoxFromContours(contours, vertices, norm(src2, src3, NORM_L2)))
    {
        for (int i = 0; i < 16; i++) {
            colors[i] = createColorOutput(Scalar(), i, -1.0);
        }
        cerr << "Could not find box" << endl;

        vector<uchar> buf;

        imencode(".png", img, buf);

        *encodedImage = (unsigned char *) malloc(buf.size());
        for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];

        return create_detection_result(colors, buf.size(), 1);
    }

    vector<Point2f> pts(4);
    vector<Point2f> dst(4);

    pts[0] = vertices[1];
    pts[1] = vertices[2];
    pts[2] = vertices[3];
    pts[3] = vertices[0];

    dst[0].x = 0;
    dst[0].y = 0;
    dst[1].x = 40;
    dst[1].y = 0;
    dst[2].x = 40;
    dst[2].y = 1080;
    dst[3].x = 0;
    dst[3].y = 1080;

    Mat pTrans;
    pTrans = getPerspectiveTransform(pts, dst);

    nokeyimg = match_histograms(nokeyimg, outputImg, ref);

    Mat warped_img;
    warpPerspective(nokeyimg, warped_img, pTrans, Size(40, 1080));

    Mat shift;
    pyrMeanShiftFiltering(warped_img, shift, 11, 21);

    vector<int> centerpoints{
        28,
        88,
        148,
        208,
        268,
        328,
        388,
        448,
        508,
        568,
        628,
        688,
        748,
        808,
        868,
        928
    };

    Mat points_img = shift.clone();

    vector<Scalar> id_colors(16);

    vector<Scalar> Lab_colors(16);

    for (size_t i = 0; i < 16; i++)
    {

        Mat labels;

        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 0.1);
        int flags = KMEANS_PP_CENTERS;

        Mat mask = Mat::zeros(1080, 40, CV_8U);

        rectangle(mask, Rect(Point(5, centerpoints[i] - 15), Point(35, centerpoints[i] + 15)), Scalar(255), -1);

        Mat centers, data;
        shift.convertTo(data, CV_32F);    
        // reshape into 3 columns (one per channel, in BGR order) and as many rows as the total number of pixels in img
        data = data.reshape(1, data.total()); 

        int nbWhitePixels = cv::countNonZero(mask);
        cv::Mat dataMasked = cv::Mat(nbWhitePixels, 3, CV_32F, cv::Scalar(0));
        cv::Mat maskFlatten = mask.reshape(1, mask.total());
            
        // filter data by the mask
        int idx = 0;
        for (int k = 0; k < mask.total(); k++)
        {
            int val = maskFlatten.at<uchar>(k, 0);          
            if (val != 0)
            {
                float val0 = data.at<float>(k, 0);
                float val1 = data.at<float>(k, 1);
                float val2 = data.at<float>(k, 2);
                dataMasked.at<float>(idx,0) = val0;
                dataMasked.at<float>(idx,1) = val1;
                dataMasked.at<float>(idx,2) = val2;
                idx++;
            }
        }

        // apply k-means    
        cv::kmeans(dataMasked, 2, labels, criteria, 10, flags, centers);

        vector<int> args(2);

        for (int j = 0; j < labels.size().height; j++) {
            args[labels.at<int>(j)]++;
        }

        // reshape to a single column of Vec3f pixels
        centers = centers.reshape(3, centers.rows);  
        dataMasked = dataMasked.reshape(3, dataMasked.rows);
        data = data.reshape(3, data.rows);

        if (args[0] > args[1]) {
            id_colors[i] = Scalar(centers.at<Vec3f>(0)[0], centers.at<Vec3f>(0)[1], centers.at<Vec3f>(0)[2]);
            Lab_colors[i] = ScalarBGR2Lab(centers.at<Vec3f>(0)[0], centers.at<Vec3f>(0)[1], centers.at<Vec3f>(0)[2]);
        } else {
            id_colors[i] = Scalar(centers.at<Vec3f>(1)[0], centers.at<Vec3f>(1)[1], centers.at<Vec3f>(1)[2]);
            Lab_colors[i] = ScalarBGR2Lab(centers.at<Vec3f>(1)[0], centers.at<Vec3f>(1)[1], centers.at<Vec3f>(1)[2]);
        }
    }

    vector< vector< Scalar > > key =
    {
    {
        Scalar(163.000, 172.000, 155.000),
        Scalar(170.000, 169.000, 155.000),
        Scalar(182.000, 158.000, 167.000),
        Scalar(201.000, 145.000, 172.000),
        Scalar(205.000, 126.000, 180.000),
        Scalar(196.000, 116.000, 173.000),
        Scalar(186.000, 111.000, 168.000),
    },
    {
        Scalar(174.000, 112.000, 102.000),
        Scalar(161.000, 119.000, 102.000),
        Scalar(152.000, 125.000, 101.000),
        Scalar(121.000, 134.000, 95.000),
        Scalar(113.000, 150.000, 100.000),
        Scalar(111.000, 163.000, 102.000),
    },
    {
        Scalar(255.000, 127.774, 129.058),
        Scalar(247.048, 129.000, 134.912),
        Scalar(231.000, 129.950, 133.050),
        Scalar(210.000, 131.000, 133.000),
        Scalar(195.950, 133.000, 134.950),
        Scalar(165.000, 133.050, 136.000),
        Scalar(106.100, 132.947, 134.053),
    },
    {
        Scalar(255.000, 128.000, 128.000),
        Scalar(228.998, 138.000, 136.000),
        Scalar(207.000, 148.000, 140.000),
        Scalar(192.053, 156.000, 146.997),
        Scalar(161.000, 171.905, 167.000),
        Scalar(140.000, 185.000, 190.000),
    },
    {
        Scalar(230.000, 126.000, 167.005),
        Scalar(212.987, 130.017, 161.976),
        Scalar(199.007, 124.000, 154.998),
        Scalar(184.002, 117.000, 146.000),
        Scalar(166.997, 111.011, 118.030),
        Scalar(123.997, 126.000, 88.000),
        Scalar(81.986, 138.094, 82.911),
    },
    {
        Scalar(220.003, 133.000, 177.000),
        Scalar(214.000, 137.000, 175.998),
        Scalar(207.013, 140.971, 173.029),
        Scalar(187.989, 153.000, 168.013),
        Scalar(161.991, 162.000, 159.000),
    },
    {
        Scalar(231.000, 125.993, 180.000),
        Scalar(208.995, 141.998, 163.998),
        Scalar(183.960, 158.018, 157.990),
        Scalar(168.000, 167.997, 145.003),
        Scalar(148.000, 173.000, 134.000),
        Scalar(134.000, 182.000, 119.000),
        Scalar(130.000, 182.045, 111.000),
    },
    {
        Scalar(249.000, 127.997, 135.003),
        Scalar(239.007, 131.000, 141.998),
        Scalar(236.000, 136.000, 133.000),
        Scalar(229.926, 139.000, 132.003),
        Scalar(215.955, 147.000, 130.000),
        Scalar(189.000, 162.000, 129.000),
        Scalar(175.005, 173.000, 124.000),
    },
    {
        Scalar(234.967, 136.055, 127.000),
        Scalar(215.002, 136.898, 118.017),
        Scalar(179.992, 145.995, 113.000),
        Scalar(160.008, 151.997, 110.008),
        Scalar(118.012, 159.991, 101.991),
        Scalar(92.019, 157.054, 93.962),
        Scalar(61.075, 158.990, 87.030),
    },
    {
        Scalar(250.000, 127.955, 127.998),
        Scalar(248.002, 131.045, 127.957),
        Scalar(237.096, 137.904, 125.045),
        Scalar(227.955, 143.000, 124.000),
        Scalar(218.955, 148.000, 124.000),
        Scalar(205.000, 155.955, 122.000),
        Scalar(176.000, 173.995, 121.002),
    },
    {
        Scalar(253.000, 128.000, 128.000),
        Scalar(248.002, 131.059, 127.941),
        Scalar(227.864, 143.000, 124.000),
        Scalar(212.000, 152.000, 122.091),
        Scalar(198.000, 160.000, 121.190),
        Scalar(158.955, 180.955, 119.000),
        Scalar(140.127, 188.000, 116.998),
    },
    {
        Scalar(190.957, 143.957, 112.005),
        Scalar(215.000, 134.000, 117.000),
        Scalar(218.011, 130.000, 120.000),
        Scalar(213.000, 129.000, 117.937),
        Scalar(207.991, 127.000, 115.000),
        Scalar(202.990, 125.010, 112.000),
    },
    {
        Scalar(162.000, 164.003, 130.997),
        Scalar(159.000, 163.000, 124.005),
        Scalar(155.008, 162.000, 117.000),
        Scalar(143.000, 156.000, 110.000),
        Scalar(141.000, 140.000, 95.000),
        Scalar(147.000, 126.000, 90.000),
    },
    {
        Scalar(176.000, 163.000, 156.000),
        Scalar(189.000, 154.000, 163.000),
        Scalar(194.000, 150.000, 170.000),
        Scalar(201.000, 144.000, 177.000),
        Scalar(211.000, 136.000, 191.000),
        Scalar(222.000, 127.000, 202.000),
    },
    {
        Scalar(147.000, 163.000, 154.000),
        Scalar(156.000, 160.000, 151.000),
        Scalar(165.000, 157.000, 150.008),
        Scalar(190.000, 147.000, 147.000),
        Scalar(221.000, 134.000, 146.000),
        Scalar(244.000, 121.000, 150.000),
    },
    {
        Scalar(226.000, 126.000, 179.000),
        Scalar(213.000, 122.995, 171.938),
        Scalar(192.008, 120.016, 161.984),
        Scalar(177.000, 107.000, 151.000),
        Scalar(128.000, 101.000, 124.000),
        Scalar(107.000, 110.000, 109.000),
    },
};

    vector< vector< double > > values =
    {
        {
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0
        },
        {
            0.0,
            25.0,
            50.0,
            100.0,
            250.0,
            425.0,
        },
        {
            0,
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            10.0
        },
        {
            0,
            0.3,
            0.5,
            1.0,
            3.0,
            5.0
        },
        {
            0,
            0.1,
            0.2,
            0.4,
            1.0,
            2.0,
            5.0
        },
        {
            0,
            5.0,
            15.0,
            30.0,
            50.0
        },
        {
            0,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0
        },
        {
            0,
            0.5,
            1.0,
            3.0,
            5.0,
            10.0,
            20.0
        },
        {
            0,
            0.002,
            0.005,
            0.01,
            0.02,
            0.04,
            0.08
        },
        {
            0,
            10.0,
            25.0,
            50.0,
            100.0,
            250.0,
            500.0
        },
        {
            0,
            1.0,
            5.0,
            10.0,
            20.0,
            40.0,
            80.0
        },
        {
            0,
            200.0,
            400.0,
            800.0,
            1200.0,
            1600.0
        },
        {
            0,
            5.0,
            10.0,
            30.0,
            50.0,
            100.0
        },
        {
            0,
            4.0,
            10.0,
            25.0,
            50.0,
            100.0
        },
        {
            0,
            100.0,
            250.0,
            500.0,
            1000.0,
            2000.0,
        },
        {
            0,
            40.0,
            80.0,
            120.0,
            180.0,
            240.0
        },
    };

    // convert colors to values used by tests here and return that instead of the color

    for (int i = 0; i < 16; i++) {
        double value = getClosest(Lab_colors[i], key[i], values[i]);
        colors[i] = createColorOutput(id_colors[i], i, value);
    }

    for (int i = 0; i < 4; ++i)
    {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, 8);
    }

    vector<uchar> buf;

    imencode(".png", shift, buf);

        *encodedImage = (unsigned char *) malloc(buf.size());
        for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];
    
    return create_detection_result(colors, buf.size(), 0);
}
