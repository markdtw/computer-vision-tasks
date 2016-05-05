#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video.hpp>
#include <csignal>
#include <iostream>
using namespace std;
using namespace cv;

bool stop = false;

class MotionTrack {
    public:
        MotionTrack() {
            theObject[0] = 0;
            theObject[1] = 0;
            objectBoundingRectangle[0] = Rect(0, 0, 0, 0);
            objectBoundingRectangle[1] = Rect(0, 0, 0, 0);
            pause = false;
            objectDetected = false;
        }
        ~MotionTrack() {}

        void captureVideo (int webcam) {
            capture.release();
            capture.open(webcam);
            if (!capture.isOpened()) {
                cout << "CAM CANNOT BE OPENED" << endl;
                return;
            }
        }

        void searchForMovement(Mat thresholdImage, Mat &cameraFeed) {

            Mat temp;
            thresholdImage.copyTo(temp);

            // 2 vectors needed for output of findContours
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            // find contours of filtered image, retrieves external contours
            findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            // if contours vector is not empty, we have found some objects
            if (contours.size() > 0)
                objectDetected = true;
            else
                objectDetected = false;

            if (objectDetected) {

                // the largest contour is found at the end of the contours vector
                // simply assume that the biggest contour is the object we are looking for.
                vector<vector<Point> > largestContourVec;
                largestContourVec.push_back(contours.at(contours.size() - 1));

                // make a bounding rectangle around the largest contour then find its centroid
                // this will be the object's final estimated position.
                objectBoundingRectangle[0] = boundingRect(largestContourVec.at(0));

                // get position to draw
                /*
                int xpos = objectBoundingRectangle[0].x + objectBoundingRectangle[0].width/2;
                int ypos = objectBoundingRectangle[0].y + objectBoundingRectangle[0].height/2;
                int x = xpos;
                int y = ypos;
                circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
                */

                // draw rectangle x1, y1, x2, y2
                int x1 = objectBoundingRectangle[0].x;
                int y1 = objectBoundingRectangle[0].y;
                int x2 = x1 + objectBoundingRectangle[0].width;
                int y2 = y1 + objectBoundingRectangle[0].height;
                rectangle(cameraFeed, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

                // write the position of the object to the screen
                stringstream xx;
                stringstream yy;
                xx << (x1+x2)/2;
                yy << (y1+y2)/2;
                putText(cameraFeed, "Tracking at ("+xx.str()+","+yy.str()+")", Point((x1+x2)/2, (y1+y2)/2), 1, 1, Scalar(255, 0, 0), 2);
            }
        }

        void tracking(const int SENSITIVITY_VALUE, const int BLUR_SIZE) {
            // sensitivity value to be used in the absdiff() function
            // size of blur used to smooth the intensity image output from absdiff() function

            // original sensitivity_value is 20, blur_size is 10

            while (!stop) {

                // read first frame
                capture.read(frame1);
                // convert frame1 to gray scale for frame differencing
                cvtColor(frame1, grayImage1, COLOR_BGR2GRAY);

                // read second frame
                capture.read(frame2);
                cvtColor(frame2, grayImage2, COLOR_BGR2GRAY);

                // perform frame differencing with the sequential images. This will output an "intensity image"
                // do not confuse this with a threshold image, we will need to perform thresholding afterwards.
                absdiff(grayImage1, grayImage2, differenceImage);

                // threshold intensity image at a given sensitivity value
                threshold(differenceImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);

                // show the difference image and threshold image
                imshow("Difference Image", differenceImage);
                imshow("Threshold Image", thresholdImage);

                // blur the image to get rid of the noise. This will output an intensity image
                blur(thresholdImage, thresholdImage, Size(BLUR_SIZE, BLUR_SIZE));

                // threshold again to obtain binary image from blur output
                threshold(thresholdImage, thresholdImage, SENSITIVITY_VALUE, 255, THRESH_BINARY);

                //show the threshold image after blur
                imshow("Final Threshold Image", thresholdImage);

                searchForMovement(thresholdImage, frame1);

                //show our captured frame
                imshow("Camera Frame", frame1);
                waitKey(30);
            }
            capture.release();
        }

    private:

        VideoCapture capture;
        // just one object to search for and keep track of its position.
        int theObject[2];
        // bounding rectangle of the object, we will use the center of this as its position.
        Rect objectBoundingRectangle[2];

        // set up the matrices that we will need to compare
        Mat frame1, frame2;
        // their grayscale images (needed for absdiff() function)
        Mat grayImage1, grayImage2;
        // resulting difference image
        Mat differenceImage;
        // thresholded difference image (for use in findContours() function)
        Mat thresholdImage;

        // key for waitKey
        int key;
        // determine whether pause or not
        bool pause;
        // for method 'searchForMovement'
        bool objectDetected;
};

void signal_handler (int signal) { stop = true; }

int main (int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: ./out <WEBCAM> <SENSITIVITY_VALUE> <BLUR_SIZE>" << endl;
        return EXIT_FAILURE;
    }

    namedWindow("Difference Image");
    namedWindow("Threshold Image");
    namedWindow("Final Threshold Image");
    namedWindow("Camera Frame");

    MotionTrack *mt = new MotionTrack();
    mt->captureVideo(atoi(argv[1]));

    signal(SIGINT, signal_handler);
    mt->tracking(atoi(argv[2]), atoi(argv[3]));

    destroyAllWindows();
    return EXIT_SUCCESS;
}
