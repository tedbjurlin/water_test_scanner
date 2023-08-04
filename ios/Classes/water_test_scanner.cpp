#include "water_test_scanner.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;
// The aruco module is part of opencv-contrib, and requires a custom opencv sdk and include folder,
// which can be found on the Disco Tray teams. See the WaterQualityTester app wiki on github for 
// details on how to install it.
using namespace cv::aruco;

// cumsum calculates a cumulative sum accross the items of a 1-dimensional matrix of floats.
vector<float> TestScanner::cumsum(Mat src) {
    vector<float> sum(src.size().height);
    sum[0] = src.at<float>(0);
    for (unsigned int i = 1; i < src.size().height; i++) {
        sum[i] = src.at<float>(i) + sum[i-1];
    }

    return sum;
}

// calculate cdf calulates the cumulative distrobution function of the histogram by
// normalizing the cumulative sum of the histogram.
vector<float> TestScanner::calculate_cdf(Mat histogram) {
    vector<float> cdf = cumsum(histogram);


    for (unsigned int i = 0; i < cdf.size(); i++)
    {
        cdf[i] = cdf[i] / cdf[cdf.size() - 1];
    }

    return cdf;
}

// calculate_lookup calculates the lookup table from the src cdf to the ref cdf.
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

// match_histograms uses the lookup table calculated from the base_ref and current_ref and
// adjusts the colorbalance of the input_image
Mat TestScanner::match_histograms(Mat input_image, Mat base_ref, Mat current_ref)
{
    // split the images into their color channels
    Mat src_bands[3], base_bands[3], curr_bands[3];
    split(input_image, src_bands);
    split(base_ref, base_bands);
    split(current_ref, curr_bands);
    
    int histSize = 256;

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    
    bool uniform = true, accumulate =  false;

    // calculate the histograms of each channel of the base and current refs using the parameters defiend above.
    Mat base_blue_hist, base_green_hist, base_red_hist, curr_blue_hist, curr_green_hist, curr_red_hist;
    calcHist( &base_bands[0], 1, 0, Mat(), base_blue_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &base_bands[1], 1, 0, Mat(), base_green_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &base_bands[2], 1, 0, Mat(), base_red_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[0], 1, 0, Mat(), curr_blue_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[1], 1, 0, Mat(), curr_green_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &curr_bands[2], 1, 0, Mat(), curr_red_hist, 1, &histSize, histRange, uniform, accumulate );
 
    // calulate the cdfs of the base and current images
    vector<float> src_cdf_blue = calculate_cdf(base_blue_hist);
    vector<float> src_cdf_green = calculate_cdf(base_green_hist);
    vector<float> src_cdf_red = calculate_cdf(base_red_hist);
    vector<float> ref_cdf_blue = calculate_cdf(curr_blue_hist);
    vector<float> ref_cdf_green = calculate_cdf(curr_green_hist);
    vector<float> ref_cdf_red = calculate_cdf(curr_red_hist);
 
    // get the lookup tables between these images.
    vector<int> blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue);
    vector<int> green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green);
    vector<int> red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red);

    // Transform the input image channels by the lookup tables
    vector<Mat> res_bands {src_bands[0].clone(), src_bands[1].clone(), src_bands[2].clone()};
    LUT(src_bands[0], blue_lookup_table, res_bands[0]);
    LUT(src_bands[1], green_lookup_table, res_bands[1]);
    LUT(src_bands[2], red_lookup_table, res_bands[2]);

    // Merge and normalize the resulting image.
    Mat res;
    merge(res_bands, res);
    convertScaleAbs(res, res);

    return res;
}

// Searches through an array of integers for a specific one and returns the index.
// Analogous to Python's .index() function.
int TestScanner::searchResult(vector<int> arr, int k){
    vector<int>::iterator it;
    it = find(arr.begin(), arr.end(), k);
    if(it != arr.end())
        return (it - arr.begin());
    else
        return -1;
}

// find_color_card uses the aruco markers on the card to identify its location and crop out the rest
// of the image, to match histograms with later.
struct ColorCardResult TestScanner::find_color_card(Mat img, Mat outputImg)
{
    // The card has 4 by 4 markers on it, so the 4 by 4 dictionary is needed.
    Dictionary arucoDict = getPredefinedDictionary(DICT_4X4_250);

    DetectorParameters parameters = DetectorParameters();

    vector<vector< Point2f>> rejectedCandidates, markerCorners;

    vector<int> markerIds;

    ArucoDetector detector(arucoDict, parameters);

    detector.detectMarkers(img, markerCorners, markerIds, rejectedCandidates);

    // The markers on the card are, in clockwise order, 23, 42, 15, and 67.
    // These markers allow the image to be correctly matched, regardless of orientation.
    int topLeftIdx = searchResult(markerIds, 23);
    int topRightIdx = searchResult(markerIds, 42);
    int bottomRightIdx = searchResult(markerIds, 15);
    int bottomLeftIdx = searchResult(markerIds, 67);

    Point2f topLeft;
    Point2f topRight;
    Point2f bottomRight;
    Point2f bottomLeft;

    // This eliminates any cases where not all of the markers were found.
    if (topLeftIdx != -1 && topRightIdx != -1 && bottomRightIdx != -1 && bottomLeftIdx != -1)
    {
        topLeft = markerCorners.at(topLeftIdx).at(0);
        topRight = markerCorners.at(topRightIdx).at(1);
        bottomRight = markerCorners.at(bottomRightIdx).at(2);
        bottomLeft = markerCorners.at(bottomLeftIdx).at(3);
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

    // Using a perspective transform, we can warp the image to get a transformed image
    // of just the color card.
    Mat pTrans;
    pTrans = getPerspectiveTransform(pts, dst);

    Mat warped_img;
    warpPerspective(img, warped_img, pTrans, Size(220, 770));

    // 220 x 770 is the dimensions of the image of the card used in memory.

    warped_img.copyTo(outputImg);

    ColorCardResult out;

    out.markerCorners = markerCorners;
    out.markerIds = markerIds;
    out.success = true;

    return out;

    
}

// converts a scaler in BGR to CIELab
// The CIELab color space is designed such that the euclidean distances between colors
// match how humans would compare colors. Closer = more similar.
Scalar TestScanner::ScalarBGR2Lab(uchar B, uchar G, uchar R) {
    Mat lab;
    Mat bgr(1,1, CV_8UC3, Scalar(B, G, R));
    cvtColor(bgr, lab, COLOR_BGR2Lab);
    return Scalar(lab.data[0], lab.data[1], lab.data[2]);
}

// This identifies which color in the list of colors for that swatch is the closest to the
// one read off of the card using the euclidian distance. The matching value from the
// Varify key is then returned.
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

// This finds the rectangle most likely to be the color strip from the contours of the image.
// The rectangle is expected to be of similar height to the color key, and have an aspect ratio of about 1:27.
bool TestScanner::findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices, double height){
    
    list<RotatedRect> boxes;
    
    for(vector<Point> contour : contours)
    {
        // Compute minimal bounding box
        cv::RotatedRect box = cv::minAreaRect(Mat(contour));

        // the computed aspect ratio of a strip
        double exp = 0.037037;

        double act = box.size.aspectRatio();
        double boxheight = box.size.height > box.size.width ? box.size.height : box.size.width;

        // increasing the 0.90 in these equations whould decrease the tolerance for variation. I haven't played around
        // with the values to find the best balance between tolerance and consistency.
        if (((2 * min(exp, act)) / (exp + act) > 0.90) && ((2 * min(height, boxheight)) / (height + boxheight) > 0.90))
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

// detect colors is where it is all put together
// it returns a DetectionResult pointer, and places the image in the encodedImage pointer
DetectionResult *TestScanner::detect_colors(Mat img, Mat ref, vector<ColorOutput> colors, uchar **encodedImage)
{
    Mat outputImg = Mat::zeros(Size(220, 770), img.type());

    // find color card is called, and the identified card is placed in outputImg for later use.
    ColorCardResult result = find_color_card(img, outputImg);

    vector<vector<Point2f>> markerCorners = result.markerCorners;
    vector<int> markerIds = result.markerIds;

    if (!result.success)
    {
        for (int i = 0; i < 16; i++) {
            colors[i] = createColorOutput(Scalar(), i, -1.0);
        }

        vector<uchar> buf;

        imencode(".png", img, buf);
        // will talk about this at the end of the file
        *encodedImage = (unsigned char *) malloc(buf.size());
        for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];
        // if no color code is found, exit code 3 is returned.
        return create_detection_result(colors, buf.size(), 3);
    }

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

    // the key is outlined in green on the image outputted to the app if the test strip is not found.
    // Currently, the app does not use this image, but it is left as an option if it has educational value.
    line(img, poly_points[0], poly_points[1], Scalar(0, 255, 0), 2);
    line(img, poly_points[1], poly_points[2], Scalar(0, 255, 0), 2);
    line(img, poly_points[2], poly_points[3], Scalar(0, 255, 0), 2);
    line(img, poly_points[3], poly_points[0], Scalar(0, 255, 0), 2);

    // the color card is blacked out on the image to be processed so that it does not cause problems for the
    // strip finder.
    fillPoly(nokeyimg, poly_points, Scalar());

    // The image is resized to increase processing speed
    resize(nokeyimg, nokeyimg, Size(nokeyimg.size().width / 4, nokeyimg.size().height / 4), INTER_LINEAR);

    // turned black and white and blurred with multiple algorithms to decrease noise while retaining
    // obvious edges.
    Mat img_gray;
    cvtColor(nokeyimg, img_gray, COLOR_BGR2GRAY);

    Mat img_filtered;
    bilateralFilter(img_gray, img_filtered, 5, 40, 40);

    GaussianBlur(img_filtered, img_filtered, Size(5, 5), 0);

    // The canny edge detection algorithm is run to highlight edges in a monochrome image.
    Mat canny;
    Canny(img_filtered, canny, 100, 200);

    // The continous contours are then found in the image. Hopefully, one of these is the
    // test strip.
    vector<vector<Point>> contours;
    findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Point2f vertices[4];    

    // No strip is found, exit code 1 is returned.
    if (!findBoxFromContours(contours, vertices, norm(src2, src3, NORM_L2)))
    {
        for (int i = 0; i < 16; i++) {
            colors[i] = createColorOutput(Scalar(), i, -1.0);
        }
        cerr << "Could not find box" << endl;

        img = match_histograms(img, outputImg, ref);

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

    // If the strip is found, it is cropped and transformed onto its own image of only the strip.
    Mat pTrans;
    pTrans = getPerspectiveTransform(pts, dst);

    nokeyimg = match_histograms(nokeyimg, outputImg, ref);

    Mat warped_img;
    warpPerspective(nokeyimg, warped_img, pTrans, Size(40, 1080));

    // the pyramid mean shift filter smooths the colors of the image.
    Mat shift;
    pyrMeanShiftFiltering(warped_img, shift, 11, 21);

    // These are the approximated center points of each swatch, in pixels, if the tail of the
    // strip is down.
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

    // The swatch locations are iterated through.
    for (size_t i = 0; i < 16; i++)
    {

        Mat labels;

        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 0.1);
        int flags = KMEANS_PP_CENTERS;

        Mat mask = Mat::zeros(1080, 40, CV_8U);

        // A mask of only that swatch is created.
        rectangle(mask, Rect(Point(5, centerpoints[i] - 15), Point(35, centerpoints[i] + 15)), Scalar(255), -1);

        Mat centers, data;
        // copy the image to the data matrix as 32-bit floats
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

        // apply k-means looking for two groups. The colors in the swatch will likely only
        // be the color of the swatch and the white of the strip backing if the swatch was
        // off-center
        cv::kmeans(dataMasked, 2, labels, criteria, 10, flags, centers);

        vector<int> args(2);

        // caluclate which of the two dominant colors was more dominant. This
        // will be the swatch color.
        for (int j = 0; j < labels.size().height; j++) {
            args[labels.at<int>(j)]++;
        }

        // reshape to a single column of Vec3f pixels
        centers = centers.reshape(3, centers.rows);  
        dataMasked = dataMasked.reshape(3, dataMasked.rows);
        data = data.reshape(3, data.rows);

        // record the more dominant color as both BGR and Lab
        if (args[0] > args[1]) {
            id_colors[i] = Scalar(centers.at<Vec3f>(0)[0], centers.at<Vec3f>(0)[1], centers.at<Vec3f>(0)[2]);
            Lab_colors[i] = ScalarBGR2Lab(centers.at<Vec3f>(0)[0], centers.at<Vec3f>(0)[1], centers.at<Vec3f>(0)[2]);
        } else {
            id_colors[i] = Scalar(centers.at<Vec3f>(1)[0], centers.at<Vec3f>(1)[1], centers.at<Vec3f>(1)[2]);
            Lab_colors[i] = ScalarBGR2Lab(centers.at<Vec3f>(1)[0], centers.at<Vec3f>(1)[1], centers.at<Vec3f>(1)[2]);
        }
    }

    // The following two vectors are hardcoded arrays of the colors and values
    // on the varify key. The lighting that these colors were taken from was the afternoon
    // sun in august on campus. They are in the Lab colorspace for easy comparison.
    // the key and values vectors have the same dimensions, so the same indexes apply to the
    // same items on the key. The outer vector is the chemicals, in order of top to bottom.
    // The inner vector is the values for each chemical, in left to right order.
    vector< vector< Scalar > > key =
    {
        {
            Scalar(154.962, 157.407, 145.344),
            Scalar(163.224, 155.192, 144.307),
            Scalar(174.581, 149.461, 150.072),
            Scalar(185.250, 144.350, 151.292),
            Scalar(197.035, 133.762, 168.282),
            Scalar(199.610, 123.542, 168.967),
            Scalar(185.317, 116.205, 160.916),
        },
        {
            Scalar(122.503, 119.512, 92.393),
            Scalar(127.561, 122.787, 87.358),
            Scalar(130.087, 125.010, 88.347),
            Scalar(102.638, 131.209, 89.377),
            Scalar(117.235, 136.850, 104.033),
            Scalar(127.273, 146.005, 113.208),
        },
        {
            Scalar(221.036, 129.422, 122.162),
            Scalar(217.931, 128.633, 128.039),
            Scalar(215.441, 128.308, 126.730),
            Scalar(207.185, 130.364, 123.882),
            Scalar(199.183, 131.679, 129.193),
            Scalar(174.718, 133.899, 129.534),
            Scalar(121.241, 132.156, 126.314),
        },
        {
            Scalar(228.504, 128.998, 121.787),
            Scalar(208.990, 136.408, 128.945),
            Scalar(189.333, 144.712, 133.291),
            Scalar(180.595, 149.313, 137.226),
            Scalar(161.665, 156.233, 151.799),
            Scalar(136.857, 167.059, 159.670),
        },
        {
            Scalar(201.227, 132.773, 147.030),
            Scalar(200.694, 134.552, 147.009),
            Scalar(200.729, 128.002, 146.617),
            Scalar(173.784, 117.116, 135.540),
            Scalar(138.105, 114.597, 100.841),
            Scalar(99.460, 129.978, 86.518),
            Scalar(65.722, 133.540, 92.458),
        },
        {
            Scalar(181.023, 138.258, 153.434),
            Scalar(178.515, 140.467, 153.685),
            Scalar(179.051, 142.554, 152.120),
            Scalar(167.077, 149.378, 149.670),
            Scalar(152.803, 155.647, 148.295),
        },
        {
            Scalar(182.476, 134.885, 156.168),
            Scalar(174.465, 142.952, 143.280),
            Scalar(164.260, 150.573, 142.236),
            Scalar(153.144, 157.983, 137.682),
            Scalar(142.832, 162.035, 135.575),
            Scalar(138.602, 167.100, 130.848),
            Scalar(133.290, 165.515, 124.488),
        },
        {
            Scalar(215.053, 128.184, 126.294),
            Scalar(206.993, 130.777, 130.267),
            Scalar(205.534, 134.129, 125.042),
            Scalar(202.800, 137.367, 126.056),
            Scalar(188.531, 143.969, 127.617),
            Scalar(167.029, 153.725, 130.166),
            Scalar(158.103, 158.694, 131.933),
        },
        {
            Scalar(203.365, 135.011, 122.546),
            Scalar(189.661, 131.712, 113.542),
            Scalar(155.943, 141.505, 111.587),
            Scalar(143.296, 145.956, 112.850),
            Scalar(108.462, 147.548, 108.564),
            Scalar(77.134, 141.547, 101.810),
            Scalar(47.458, 138.753, 96.902),
        },
        {
            Scalar(213.400, 128.759, 120.015),
            Scalar(211.079, 130.103, 121.759),
            Scalar(202.135, 135.169, 121.981),
            Scalar(195.702, 138.606, 122.058),
            Scalar(186.960, 144.483, 123.954),
            Scalar(175.049, 149.672, 123.961),
            Scalar(153.338, 159.036, 129.835),
        },
        {
            Scalar(216.789, 128.860, 120.453),
            Scalar(216.178, 130.028, 122.565),
            Scalar(198.900, 138.845, 122.625),
            Scalar(182.997, 146.913, 123.876),
            Scalar(169.784, 151.904, 126.074),
            Scalar(147.926, 163.274, 130.037),
            Scalar(134.433, 169.652, 130.366),
        },
        {
            Scalar(171.151, 138.080, 112.343),
            Scalar(191.240, 128.261, 112.131),
            Scalar(190.967, 126.067, 111.537),
            Scalar(184.541, 124.413, 108.454),
            Scalar(175.644, 122.594, 102.431),
            Scalar(166.367, 120.861, 98.164),
        },
        {
            Scalar(151.976, 159.140, 134.326),
            Scalar(151.290, 157.763, 131.045),
            Scalar(150.971, 155.085, 124.677),
            Scalar(131.524, 147.517, 115.060),
            Scalar(99.938, 133.644, 91.856),
            Scalar(107.927, 128.690, 85.681),
        },
        {
            Scalar(156.890, 153.954, 145.838),
            Scalar(162.554, 150.462, 147.559),
            Scalar(164.583, 148.709, 151.944),
            Scalar(163.614, 146.383, 156.678),
            Scalar(161.860, 142.894, 169.201),
            Scalar(164.830, 137.755, 178.623),
        },
        {
            Scalar(138.519, 158.097, 149.799),
            Scalar(145.318, 156.035, 146.796),
            Scalar(155.911, 153.675, 143.388),
            Scalar(170.353, 145.978, 137.983),
            Scalar(192.935, 133.979, 135.511),
            Scalar(207.821, 124.100, 141.590),
        },
        {
            Scalar(179.063, 135.474, 159.466),
            Scalar(183.656, 130.931, 164.401),
            Scalar(167.239, 125.074, 156.776),
            Scalar(127.227, 105.006, 138.798),
            Scalar(75.765, 118.056, 107.730),
            Scalar(54.913, 129.413, 100.084),
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

    // find which color is closest to the one in the image for each chemical and create the
    // ColorOutput structs
    for (int i = 0; i < 16; i++) {
        double value = getClosest(Lab_colors[i], key[i], values[i]);
        colors[i] = createColorOutput(id_colors[i], i, value);
    }

    for (int i = 0; i < 4; ++i)
    {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, 8);
    }

    // the following is what allows us to access our output image from dart without memory errors.
    // first we create a buffer of bytes.
    vector<uchar> buf;

    // Then we encode our image (the exact encoding doesn't matter as long as it is a common one) into
    // that buffer. The Flutter Image.memory() function only reads images with file encoding.
    imencode(".png", shift, buf);

    // Then we allocate the memory for the array manually, to match the size of the buffer.
    // Finally we write the contents of the buffer to the array.
    //
    // Basically, Dart has a pointer to the pointer that starts this array. Because of this,
    // the array won't get deallocated before Dart can read it. And because dart has a pointer
    // to a pointer, and not a pointer directly to the array, we can allocate this array to any
    // size without affecting darts pointer.
    *encodedImage = (unsigned char *) malloc(buf.size());
    for (int i=0; i < buf.size(); i++) (*encodedImage)[i] = buf[i];
    
    return create_detection_result(colors, buf.size(), 0);
}
