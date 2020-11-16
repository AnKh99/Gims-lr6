#include "opencv2/highgui.hpp"
#include <vector>

#include "Layer.h"


// Theory - https://habr.com/ru/company/abbyy/blog/218285/

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::string;
using std::vector;
using namespace cv;

// white-black  uchar
// color        Vec3B
#define PIXEL_TYPE uchar
int main()
{
    // paths to source images
    //string path = "input.jpg";    // 1920 x 2606 x 24bit  || color with complex layout
    //string path = "1.jpg";        // 398 x 529 x 24bit    || color with complex layout
    string path = "2.jpg";          // 1920 x 2606 x 24bit  || white-black with complex layout
    //string path = "3.jpg";        // 1920 x 2560 x 24bit  || white-black with simple layout
    //string path = "3.png";        // 20 x 20 x 32bit      || color without text
    //string path = "4.png";        // 4096 x 4096 x 32bit  || color without text
    //string path = "5.png";        // 350 x 350 x 8bit     || color with complex layout(too low quality)
	

    Mat src = imread(path, IMREAD_GRAYSCALE);
    if (src.empty()) {
        return 1;
    }

    vector<Layer<PIXEL_TYPE>*> pyramid = createPyramid<Layer<PIXEL_TYPE>*, PIXEL_TYPE>(src);
    generateThresholds(pyramid);

    // first layer's threshold didn't generate, so need do it manual
    imshow("result", recontrast<PIXEL_TYPE>(src, getTresholdMat(*pyramid[0]->threshold), 3));
    waitKey(0);
    return 0;
}