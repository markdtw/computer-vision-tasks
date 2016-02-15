#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;
int ddepth = CV_8U;

int blurThresh = 1;
Mat src, gaus, gray;
Mat grad, grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;

// define a trackbar callback
static void onTrackbar (int, void*) {
    // prevent blurThresh to be even
    blurThresh = (blurThresh%2==1)?blurThresh: blurThresh+1;
    // gaussian blur by function call
    GaussianBlur(src, gaus, Size(blurThresh, blurThresh), 0, 0);
    // convert dst to gray scale image
    cvtColor(gaus, gray, COLOR_BGR2GRAY);
    // calculate the derivatives in x and y directions
    Sobel(gray, grad_x, ddepth, 1, 0);
    Sobel(gray, grad_y, ddepth, 0, 1);
    // convert our partial results to CV_32F/CV_64F for pow
    grad_x.convertTo(grad_x, CV_64F);
    grad_y.convertTo(grad_y, CV_64F);
    // g = sqrt(g_x^2 + g_y^2)
    pow(grad_x, 2, grad_x);
    pow(grad_y, 2, grad_y);
    add(grad_x, grad_y, grad);
    sqrt(grad, grad);
    // convert it back to CV_8U and show it
    grad.convertTo(grad, ddepth);
    imshow("Result (gaussian+sobel)", grad);
}

int main (int argc, const char** argv) {
    if (argc!=2) {
        cout << "./out [image location]" << endl;
        return -1;
    }
    src = imread(argv[1], IMREAD_UNCHANGED);
    // create a window and show the original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", src);

    // create a window
    namedWindow("Result (gaussian+sobel)", WINDOW_AUTOSIZE);
    // create a toolbar
    createTrackbar("Gaussian", "Result (gaussian+sobel)", &blurThresh, 30, onTrackbar);
    // do gaussian blur, then convert to gray scale, then do sobel edge detection
    onTrackbar(0, 0);

    waitKey(0);
    return 0;
}
