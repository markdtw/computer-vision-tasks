#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int k, iters;

Mat color_quantization(Mat& labels, Mat& centers, Mat& src) {
    // assign result as input (8UC3).
    Mat rgb(src.rows, src.cols, CV_8UC3);

    // 'centers' has floating-point values, should be converted into uint8_t type
    //  with range [0, 255]. The number of channels is also translated into 3.
    centers.convertTo(centers, CV_8UC1, 255.0);
    centers = centers.reshape(3);

    // The label is replaced by the corresponding(B, G, R) to make the final image rgb.
    MatIterator_<Vec3b> it = rgb.begin<Vec3b>();
    MatConstIterator_<int> lit = labels.begin<int>();
    for (; it != rgb.end<Vec3b>(); it++, lit++) {
        const Vec3b& rgb = centers.ptr<Vec3b>(*lit)[0];
        *it = rgb;
    }

    return rgb;
}

Mat kmeans(Mat& src) {
    // firstly, reshape to 1 dimension to meet k means requirments.
    Mat reshaped, reshaped_32F;
    reshaped = src.reshape(1, src.cols*src.rows);
    reshaped.convertTo(reshaped_32F, CV_32FC1, 1.0/255.0);

    // do k means.
    Mat labels, centers;
    double eps = 1.0;
    kmeans(reshaped_32F, k, labels,
            TermCriteria(TermCriteria::COUNT, iters, eps),
                iters, KMEANS_RANDOM_CENTERS, centers);

    return color_quantization(labels, centers, src);
}

int main (int argc, const char** argv) {
    if (argc!=4) {
        cout << "./out [image location] [k] [iterations]" << endl;
        return -1;
    }
    Mat src = imread(argv[1], IMREAD_UNCHANGED);
    k = atoi(argv[2]);
    iters = atoi(argv[3]);

    // create windows.
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Original", src);

    Mat rst = kmeans(src);

    imshow("Result", rst);
    waitKey(0);
    return 0;
}
