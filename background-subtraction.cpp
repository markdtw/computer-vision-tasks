#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/video.hpp>
#include <csignal>
#include <iostream>
using namespace cv;
using namespace std;

bool stop = false;

class BgSubtraction {
    public:
        BgSubtraction () {
            //create Background Subtractor objects
            pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
        }
        ~BgSubtraction () {}

        void captureVideo (int webcam) {
            capture.open(webcam);
            if (!capture.isOpened()) {
                cout << "CAM CANNOT BE OPENED" << endl;
                return;
            }
        }

        void applyBS () {
            //read input data. ESC or 'q' for quitting
            while (!stop) {
                //read the current frame
                capture >> frame;
                //update the background model
                pMOG2->apply(frame, fgMaskMOG2);
                //show the current frame and the fg masks
                imshow("Frame", frame);
                imshow("FG Mask MOG 2", fgMaskMOG2);
                waitKey(30);
            }
            //delete capture object
            capture.release();
        }

    private:
        VideoCapture capture;
        Mat frame;                          // current frame
        Mat fgMaskMOG2;                     // fg mask generated by MOG2 method
        Ptr<BackgroundSubtractor> pMOG2;    // MOG2 Background subtractor
};

void signal_handler (int signal) { stop = true; }

int main(int argc, char** argv) {
    if(argc != 2) {
        cout << "Usage: ./bg <webcam id>" << endl;
        cout << "Notice: this requirs your webcam!" << endl;
        return EXIT_FAILURE;
    }
    //create GUI windows
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");

    BgSubtraction *bgs = new BgSubtraction();
    bgs->captureVideo(atoi(argv[1]));

    signal(SIGINT, signal_handler);
    bgs->applyBS();

    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}