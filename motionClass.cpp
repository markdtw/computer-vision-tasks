#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video.hpp>
#include <iostream>
using namespace std;
using namespace cv;

class MotionTrack {
    public:
        MotionTrack() {
            theObject[0] = 0;
            theObject[1] = 0;
            objectBoundingRectangle = Rect(0, 0, 0, 0);
            pause = false;
            objectDetected = false;
        }
        ~MotionTrack() {}

        void captureVideo (int webcam) {
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
                objectBoundingRectangle = boundingRect(largestContourVec.at(0));
                int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width/2;
                int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height/2;

                //update the objects positions by changing the 'theObject' array values
                theObject[0] = xpos;
                theObject[1] = ypos;
            }

            int x = theObject[0];
            int y = theObject[1];

            //draw crosshairs around the object
            circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
            line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
            line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
            line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
            line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);

            //write the position of the object to the screen
            stringstream xx;
            stringstream yy;
            xx << x;
            yy << y;
            putText(cameraFeed, "Tracking at (" + xx.str() + "," + yy.str() + ")", Point(x, y), 1, 1, Scalar(255, 0, 0), 2);
        }

        void tracking(const int SENSITIVITY_VALUE, const int BLUR_SIZE) {
            // sensitivity value to be used in the absdiff() function
            // size of blur used to smooth the intensity image output from absdiff() function

            // original sensitivity_value is 20, blur_size is 10

            for (;;) {

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
                threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);

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
        Rect objectBoundingRectangle;

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

    mt->tracking(atoi(argv[2]), atoi(argv[3]));

    destroyAllWindows();
    return EXIT_SUCCESS;
}
