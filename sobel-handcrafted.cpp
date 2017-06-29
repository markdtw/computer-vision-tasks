#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;
using namespace cv;
int ddepth = CV_8U;

Mat src, gaus, gray, sob;

void convolution (const Mat& in, Mat &Result, float k[][3]) {
    // modified from OpenCV-3.0.0-tutorial_mat_mast_operations
    CV_Assert(in.depth()==CV_8U); // accept only uchar images
    // create an output image with the same size and the same type
    // depending on the number of channels we may have one or more subcolumns
    Result.create(in.size(), in.type());
    const int nChannels = in.channels();
    // We'll use the plain C [] operator to access pixels.
    // Because we need to access multiple rows at the same time 
    //  we'll acquire the pointers for each of them (a previous, a current and a next line).
    // We need another pointer to where we're going to save the calculation.
    // Then simply access the right items with the [] operator.
    // For moving the output pointer ahead we simply increase this (with one byte) after each operation.
    for (int j=1; j<in.rows-1; j++) {
        const uchar* previous   = in.ptr<uchar>(j-1);
        const uchar* current    = in.ptr<uchar>(j);
        const uchar* next       = in.ptr<uchar>(j+1);

        uchar* output = Result.ptr<uchar>(j);

        for (int i=nChannels; i<nChannels*(in.cols - 1); i++) {
            // saturate cast is to limit the value to 0~255, it does: min(max(round(value), 0), 255)
            /* CORRELATION (NOT FLIPPING THE KERNEL)
            *output++ = saturate_cast<uchar>(
                    k[0][0]*next[i+nChannels]     + k[0][1]*next[i]     + k[0][2]*next[i-nChannels]     +
                    k[1][0]*current[i+nChannels]  + k[1][1]*current[i]  + k[1][2]*current[i-nChannels]  +
                    k[2][0]*previous[i+nChannels] + k[2][1]*previous[i] + k[2][2]*previous[i-nChannels]);

            */
            *output++ = saturate_cast<uchar>(
                    k[0][0]*previous[i-nChannels] + k[0][1]*previous[i] + k[0][2]*previous[i+nChannels] +
                    k[1][0]*current[i-nChannels]  + k[1][1]*current[i]  + k[1][2]*current[i+nChannels] +
                    k[2][0]*next[i-nChannels]     + k[2][1]*next[i]     + k[2][2]*next[i+nChannels]);
            
        }
    }
    // On the borders of the image the upper notation results inexistent pixel locations.
    // In these points our formula is undefined. A simple solution
    // is to not apply the kernel in these points and, set the pixels on the borders to zeros.
    Result.row(0).setTo(Scalar(0));             // The top row
    Result.row(Result.rows-1).setTo(Scalar(0)); // The bottom row
    Result.col(0).setTo(Scalar(0));             // The left column
    Result.col(Result.cols-1).setTo(Scalar(0)); // The right column
}

void Sobel_Edge_Detection () {
    // define derivatives mats
    float kernel_x[3][3] = 
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}

    };
    float kernel_y[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    // calculate Gx, Gy
    Mat grad_x, grad_y;
    convolution(gray, grad_x, kernel_x);
    convolution(gray, grad_y, kernel_y);
    // convert to CV_64F and do sob = sqrt(Gx^2+Gy^2), remember to convert back to CV_8U
    grad_x.convertTo(grad_x, CV_64F);
    grad_y.convertTo(grad_y, CV_64F);
    pow(grad_x, 2, grad_x);
    pow(grad_y, 2, grad_y);
    add(grad_x, grad_y, sob);
    sqrt(sob, sob);
    sob.convertTo(sob, ddepth);
    // done
}

void Gaussian () {
    float kernel[3][3] = 
    {
        {0.0625, 0.125, 0.0625},
        { 0.125,  0.25,  0.125},
        {0.0625, 0.125, 0.0625}
    
    };
    convolution(src, gaus, kernel);
    return;
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
    // gaussian blur
    Gaussian();
    // convert dst to gray scale image
    cvtColor(gaus, gray, COLOR_BGR2GRAY);
    // sobel edge detection
    Sobel_Edge_Detection();
    // done
    imshow("Result (gaussian+sobel)", sob);

    waitKey(0);
    return 0;
}
