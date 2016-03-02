#include "implement.cpp"
using namespace cv;
using namespace std;


int main()
{
    Mat blur, gray, sobel;
    Mat src = imread("img/lbjc.jpg", CV_LOAD_IMAGE_UNCHANGED);
    namedWindow( "Result", CV_WINDOW_AUTOSIZE);
    Gaussian(src, blur);
    cvtColor(blur, gray, CV_BGR2GRAY); 
    Sobel(gray, sobel); 

    waitKey(0);
    return 0;
}


