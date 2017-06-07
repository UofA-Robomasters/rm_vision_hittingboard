#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <cstdlib>

#define COLOR "RED"
#define BINARY_SEARCH_ENABLED
// #define VIDEO

#define DEBUG
#define ERROR_STR(str) std::cerr << (str) << std::endl


std::string const k_detectorWindow {"blob detector window"};
std::string const k_binWindow {"binary window"};

// global variables
// images
cv::Mat origImg, grayImg, binImg;


// gaussian blur
int sigma = 3;
int const k_maxSigma = 15;

// preprocess image
int hsvTh = 110;
int bgrTh = 110;
int const k_maxTh = 255;

int const k_maxTargetNum = 5;

void DetectorConfig(cv::SimpleBlobDetector::Params& param /*out*/)
{
    // Setup SimpleBlobDetector parameters.

    // Change thresholds
    param.minThreshold = 0;
    param.maxThreshold = 255;

    // Filter by Area.
    param.filterByArea = true;
    param.minArea = 50;
    param.maxArea = 700;

    // Filter by Circularity
    param.filterByCircularity = false;


    // Filter by Convexity
    param.filterByConvexity = true;
    param.minConvexity = 0.7;

    // Filter by Inertia
    param.filterByInertia = true;
    param.minInertiaRatio = 0.01;
    param.maxInertiaRatio = 0.3;
    
    param.filterByColor = true;
    param.blobColor = 255;
}

void ExtractByColor(std::string const& color, cv::Mat const& in, cv::Mat& out/*out*/)
{
    int const colorIdx = (color == "BLUE") ? 0 : 2;
    std::vector<cv::Mat> ch(3);
    cv::split(in, ch);

    cv::Mat bgrImgSel = ch[colorIdx];

    cv::GaussianBlur(bgrImgSel, bgrImgSel, cv::Size(3, 3), sigma, sigma);
    cv::threshold(bgrImgSel, out, bgrTh, 255, cv::THRESH_BINARY);
}


void ExtractByBrightness(cv::Mat const& in, cv::Mat& out/*out*/)
{
    cv::Mat hsvImg;
    cv::cvtColor(in, hsvImg, CV_BGR2HSV);
    std::vector<cv::Mat> ch(3);
    cv::split(hsvImg, ch);

    cv::Mat hsvImgValue = ch[2];

    cv::GaussianBlur(hsvImgValue, hsvImgValue, cv::Size(3, 3), sigma, sigma);
    cv::threshold(hsvImgValue, out, hsvTh, 255, cv::THRESH_BINARY);
}

void MaskByCircle(cv::Mat const& in, int radii, cv::Point const center, cv::Mat& out /*out*/)
{
    in.copyTo(out);
    cv::Mat circleMask = cv::Mat::ones(in.size(), CV_8UC1);
    cv::circle(circleMask, center, radii, cv::Scalar(0), -1);
    out = out.setTo(cv::Scalar(0, 0, 0), circleMask);
}

void PreprocessImage()
{   
    cv::Mat origImgCopy;
    cv::Point circleCenter(origImg.cols / 2, origImg.rows / 2);
    int const circleRadii = 350; /*dummy*/
    MaskByCircle(origImg, circleRadii, circleCenter, origImgCopy/*out*/);
    
    cv::GaussianBlur(origImgCopy, origImgCopy, cv::Size(3, 3), sigma, sigma);
    cv::Mat maskColor, maskBrightness;
    ExtractByColor(COLOR, origImgCopy, maskColor);
    ExtractByBrightness(origImgCopy, maskBrightness);

    cv::bitwise_and(maskColor, maskBrightness, binImg);

    // erode and dilate
    cv::erode(binImg, binImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::dilate(binImg, binImg, cv::Mat(), cv::Point(-1, -1), 2);

    cv::imshow(k_binWindow, binImg);
}

void BlobDetect(int, void*)
{
    PreprocessImage();

    cv::SimpleBlobDetector::Params blobParam;
    DetectorConfig(blobParam);
    cv::SimpleBlobDetector detector(blobParam);
    std::vector<cv::KeyPoint> keypoints;

    detector.detect(binImg, keypoints);

#ifdef DEBUG
    std::cout << "# of keypoints: " << keypoints.size() << std::endl
              << "hsv th: " << hsvTh << std::endl
              << "bgr th: " << bgrTh << std::endl;
#endif // DEBUG

#ifdef BINARY_SEARCH_ENABLED

#define IN_RANGE(x, a, b) (x) >= (a) && (x) <= (b)

    int lo = 0, hi = 254;
    int& r_varTarget = bgrTh;
    while (lo <= hi)
    {
        keypoints.clear();
        int r_varTarget = (lo + hi) / 2;

        PreprocessImage();
        detector.detect(binImg, keypoints);

        int numKeyPoint = keypoints.size();
#ifdef DEBUG
        std::cout << "# of keypoints: " << keypoints.size() << std::endl
                  << "hsvTh: " << hsvTh << std::endl
                  << "bgrTh: " << bgrTh << std::endl;
#endif // DEBUG
        if ( IN_RANGE(numKeyPoint, k_maxTargetNum-3, k_maxTargetNum+3) )
        {
            break;
        }
        else if ( numKeyPoint > k_maxTargetNum)
        {
            lo = r_varTarget + 1;
        }
        else
        {
            hi = r_varTarget - 1;
        }
    }
#undef IN_RANGE
#endif // BINARY_SEARCH_ENABLED

    cv::Mat imgWithKeypoints;
    cv::drawKeypoints(origImg, keypoints, imgWithKeypoints, 
                      cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow(k_detectorWindow, imgWithKeypoints);

}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        ERROR_STR("Usage: detector /path/to/file\n"
                  "       detector 0: use webcam(video only)");
        return -1;
    }

    cv::namedWindow(k_binWindow);
    cv::namedWindow(k_detectorWindow);
    cv::createTrackbar("sigma", k_binWindow, &sigma, k_maxSigma, BlobDetect);
    cv::createTrackbar("hsv th", k_binWindow, &hsvTh, k_maxTh, BlobDetect);
    // cv::createTrackbar("bgr th", k_binWindow, &bgrTh, k_maxTh /*, BlobDetect*/);
#ifndef VIDEO
    // for images
    origImg = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    BlobDetect(0, 0);

    cv::waitKey(0);

#else
    // for video
    cv::VideoCapture cap;

    if (std::strcmp(argv[1], "0") == 0)
    {
        cap.open(0);
    }
    else
    {
        cap.open(argv[1]);
    }
 
    if (!cap.isOpened())
    {
        ERROR_STR("failed to open camera.");
        return -1;
    }
    
    for (;;)
    {
        cap >> origImg;

        BlobDetect(0, 0);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }

#endif // VIDEO
    return 0;
}