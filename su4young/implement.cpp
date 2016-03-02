#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
void Gaussian(Mat& src, Mat& dst);
void Sobel(Mat& src, Mat& dst);
void filter(Mat& src, Mat& dst, int ddepth, Mat& kernel);

void Gaussian(Mat& src, Mat& dst)
{
#ifdef PARTA
    cout << "Gaussian smoothing part A" << endl;
    GaussianBlur( src, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
#else
    cout << "Gaussian smoothing part B" << endl;
    Mat kernel;
    kernel = (Mat_<double>(3, 3) << .0625, .125, .0625, .125, .25, .125, .0625, .125, .0625);
    cout << kernel << endl;

    namedWindow("filter", CV_WINDOW_AUTOSIZE);
    filter(src, dst, CV_8UC3, kernel);
    imshow("filter", dst);
    imshow("Result", src);
#endif
}


void Sobel(Mat& src, Mat& dst)
{
    Mat hor, ver, addup, sq;
#ifdef PARTA
    Mat hor_abs, ver_abs;
    
    cout << "Sobel edge detection part A" << endl;
    Sobel(src, hor, CV_16S, 1, 0, 3); // detect X edges
    Sobel(src, ver, CV_16S, 0, 1, 3); // detect Y edges
    
#else
    Mat kernel_X, kernel_Y;
    kernel_X = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    kernel_Y = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    filter(src, hor, CV_16S, kernel_X);
    filter(src, ver, CV_16S, kernel_Y);
#endif
    // combine two image by G = |Gx| + |Gy| (or G = sqrt(Gx^2 + Gy^2))
    
    // G = |Gx| + |Gy|
    // convertScaleAbs(hor, hor);
    // convertScaleAbs(ver, ver);
    // addWeighted(hor, 0.5, ver, 0.5, 0, dst);

    // G = sqrt(Gx^2 + Gy^2)
    pow(hor, 2, hor);
    pow(ver, 2, ver);
    add(hor, ver, addup); // Gx^2 + Gy^2
    addup.convertTo(addup, CV_32F); // can only perform sqrt when its in CV_32F
    sqrt(addup, sq);
    sq.convertTo(dst, CV_8U); // back to CV_8U

    imshow("Result", dst);
    return;

}

#define CONVSUM() (img.at<double>(x + u, y + v)* kernel.at<double>(ker_off + u, ker_off + v))
#define CONVSUMC3(CH) (img.at<Vec3d>(x + u, y + v)[CH] * kernel.at<double>(ker_off + u, ker_off + v))
#define IS3CH(X) (X == CV_8UC3 || X == CV_32FC3 || X == CV_64FC3)
void filter(Mat& src, Mat& dst, int ddepth, Mat& kernel) {
    Mat img;
    int ker_off =(kernel.rows - 1) / 2;
    int endX = src.rows - ker_off;
    int endY = src.cols - ker_off;
    int ch = src.channels();

#ifndef OD    
    if(ch == 3) {
        src.convertTo(img, CV_64FC3); // datatype is important as fuck.
        dst = Mat(src.rows, src.cols, CV_64FC3, CV_RGB(0, 0, 0));
    } else {
        src.convertTo(img, CV_64F); // datatype is important as fuck.
        dst = Mat(src.rows, src.cols, CV_64F);
    }

    for(int x = ker_off; x < endX; x++) {
        for(int y = ker_off; y < endY; y++) {
            double ch_0 = 0, ch_1 = 0, ch_2 = 0;
            for(int u = -ker_off; u <= ker_off; u++) {
                for(int v = -ker_off; v <= ker_off; v++) {
                    if(ch == 3) {
                        ch_0 += CONVSUMC3(0);
                        ch_1 += CONVSUMC3(1);
                        ch_2 += CONVSUMC3(2);
                    } else {
                        ch_0 += CONVSUM();
                    }
                }
            }
            if(ch == 3) {
                dst.at<Vec3d>(x, y)[0] = ch_0;
                dst.at<Vec3d>(x, y)[1] = ch_1;
                dst.at<Vec3d>(x, y)[2] = ch_2;
            } else {
                dst.at<double>(x, y) = ch_0;
            }
        }
    }
#else
    int e = ch * (src.cols - 1); // element per rows(include channel)
    int ksize = kernel.rows * kernel.cols;
    int begin = ch * ker_off;    // begin of col
    int end = e - ch *ker_off;   // end of col
    double* kptr = kernel.ptr<double>(0);
    dst.create(src.size(), src.type());
    cout << "1-d access" << endl;

    for(int i = ker_off; i < src.rows - ker_off; i++) {
        uchar* sptr[kernel.cols];
        uchar* dptr = dst.ptr<uchar>(i);
        
        // calculate rows that will be access by kernel and store it.
        // mat.ptr<type>(i) ==> point to row[i]
        for(int x = 0, y = -ker_off; x < kernel.rows; x ++, y++) {
            sptr[x] = src.ptr<uchar>(i + y);        
        } 

        for(int j = begin; j < end; j++) {
            double temp = 0;

            // perform convoultion
            for(int k = 0; k < ksize; k++) {
                int rows = k / kernel.cols ;            // compute corresponding rows
                int offset = k % kernel.cols - ker_off; // compute corresponding columns
                temp += saturate_cast<double> (
                    kptr[k] * sptr[rows][j + offset * ch]
                );
            }
            
            *dptr++ = saturate_cast<uchar>(temp); // move to next pixel 
        }

    }    


#endif
    cout << "complete" << endl;
    dst.convertTo(dst, ddepth);
}


