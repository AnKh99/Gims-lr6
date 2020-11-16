#pragma once

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <opencv2/core/mat.hpp>

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::vector;
using namespace cv;

// useless??
struct Threshold {
    float min;
    float max;
    float avg;
};

// TODO: need to create methods(maximum, minimum, average) for UCHAR and Vec3b decided
template <typename T>
T maximum(T first, T second, T third, T fourth) {
    return max(first, max(second, max(third, fourth)));
}

template <typename T>
T minimum(T first, T second, T third, T fourth) {
    return min(first, min(second, min(third, fourth)));
}

template <typename T>
T average(T first, T second, T third, T fourth) {
    return (first + second + third + fourth) / 4;
}

uchar localAvg(const Mat table, int row, int column, int window_size = 3);

template <typename T>
Mat summedAreaTable(Mat img);

template <typename T>
Mat recontrast(Mat img, Mat treshold, double k = 6);

Mat getTresholdMat(Mat _tresh);

template <typename Vec_T, typename T>
auto createPyramid(const Mat& src);

template <typename Vec_T>
void generateThresholds(vector<Vec_T>& pyramid);


template <typename T>
class Layer {
public:
    Layer<T>(Mat img);

    Layer<T>(const Layer& previousLayer);

    void setLvl(int _lvl);

    ~Layer();

    void calculateTreshold();
    void recalculate(Mat _threshold);


    Mat* max;
    Mat* min;
    Mat* avg;
    Mat* threshold;
    int width;
    int height;
    int lvl;
private:
    int type;
};

template<typename T>
inline Layer<T>::Layer(Mat img) {
    lvl = 0;
    int columns = img.cols - img.cols % 2;
    int rows = img.rows - img.rows % 2;
    width = columns / 2;
    height = rows / 2;
    type = img.type();
    if (width >= 2 && height >= 2) {
        max = new Mat(height, width, type);
        min = new Mat(height, width, type);
        avg = new Mat(height, width, type);
        resize(img, *max, { width, height });
        resize(img, *min, { width, height });
        resize(img, *avg, { width, height });

        for (int i = 0; i < height - 1; i++)
            for (int j = 0; j < width - 1; j++) {
                max->at<T>(i, j) = maximum(
                    img.at<T>(i * 2, j * 2),
                    img.at<T>(i * 2 + 1, j * 2),
                    img.at<T>(i * 2, j * 2 + 1),
                    img.at<T>(i * 2 + 1, j * 2 + 1)
                );
                min->at<T>(i, j) = minimum(
                    img.at<T>(i * 2, j * 2),
                    img.at<T>(i * 2 + 1, j * 2),
                    img.at<T>(i * 2, j * 2 + 1),
                    img.at<T>(i * 2 + 1, j * 2 + 1)
                );
                avg->at<T>(i, j) = average(
                    img.at<T>(i * 2, j * 2),
                    img.at<T>(i * 2 + 1, j * 2),
                    img.at<T>(i * 2, j * 2 + 1),
                    img.at<T>(i * 2 + 1, j * 2 + 1)
                );
            }
    }
    else {
        width = height = 0;
        max = nullptr;
        min = nullptr;
        avg = nullptr;
    }
    threshold = nullptr;
}

template<typename T>
Layer<T>::Layer(const Layer& previousLayer)
{
    const Mat _max = *previousLayer.max;
    const Mat _min = *previousLayer.min;
    const Mat _avg = *previousLayer.avg;

    int columns = previousLayer.width - previousLayer.width % 2;
    int rows = previousLayer.height - previousLayer.height % 2;
    width = columns / 2;
    height = rows / 2;
    type = _max.type();
    lvl = previousLayer.lvl + 1;

    if (width >= 2 && height >= 2) {
        max = new Mat(height, width, type);
        min = new Mat(height, width, type);
        avg = new Mat(height, width, type);

        for (int i = 0; i < height - 1; i++)
            for (int j = 0; j < width - 1; j++) {
                max->at<T>(i, j) = maximum(
                    _max.at<T>(i * 2, j * 2),
                    _max.at<T>(i * 2 + 1, j * 2),
                    _max.at<T>(i * 2, j * 2 + 1),
                    _max.at<T>(i * 2 + 1, j * 2 + 1)
                );
                min->at<T>(i, j) = minimum(
                    _min.at<T>(i * 2, j * 2),
                    _min.at<T>(i * 2 + 1, j * 2),
                    _min.at<T>(i * 2, j * 2 + 1),
                    _min.at<T>(i * 2 + 1, j * 2 + 1)
                );
                avg->at<T>(i, j) = average(
                    _avg.at<T>(i * 2, j * 2),
                    _avg.at<T>(i * 2 + 1, j * 2),
                    _avg.at<T>(i * 2, j * 2 + 1),
                    _avg.at<T>(i * 2 + 1, j * 2 + 1)
                );
            }
    }
    else {
        width = height = 0;
        max = nullptr;
        min = nullptr;
        avg = nullptr;
    }
    threshold = nullptr;
}

// useless??
template<typename T>
inline void Layer<T>::setLvl(int _lvl)
{
    lvl = _lvl;
}

template<typename T>
inline Layer<T>::~Layer()
{
    delete[] max, min, avg;
}

template<typename T>
inline void Layer<T>::calculateTreshold() {
    threshold = new Mat(height, width, type);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            threshold->at<T>(i, j) = avg->at<T>(i, j);
        }
    }
}

template<typename T>
inline void Layer<T>::recalculate(Mat _threshold) {
    threshold = new Mat(height, width, type);
    resize(_threshold, *threshold, { width, height }, INTER_MAX);
    //resize(_threshold, *threshold, { width, height });

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            auto current = threshold->at<T>(i, j);
            auto mx = max->at<T>(i, j);
            auto mn = min->at<T>(i, j);
            auto difference = mx - mn;
            if (threshold->at<T>(i, j) < abs(max->at<T>(i, j) - min->at<T>(i, j))) // current < abs(difference) ??
                threshold->at<T>(i, j) = (max->at<T>(i, j) + min->at<T>(i, j) + avg->at<T>(i, j)) / 3; // current = (mx+mn+avg.at<T>(i,j)/3 ?? 
        }
    }
}

Mat getTresholdMat(Mat _tresh) {
    Mat* threshold = new Mat(_tresh.cols * 2, _tresh.rows * 2, _tresh.type());
    resize(_tresh, *threshold, { _tresh.cols * 2, _tresh.rows * 2 });
    return *threshold;
}

template <typename T>
Mat recontrast(Mat img, Mat treshold, double k) {
    Mat new_image = Mat::zeros(img.size(), img.type());
    /*
    // for color contrast
    double alpha = 1.0;
    int beta = 0;
    */

    for (int i = 0; i < img.rows - 1; i++) {
        for (int j = 0; j < img.cols - 1; j++) {
            //new_image.at<T>(i, j) = saturate_cast<uchar>(k*img.at<T>(i, j)+beta);
            if (img.at<T>(i, j) > treshold.at<T>(i, j))
                new_image.at<T>(i, j) = 255;
            else
                new_image.at<T>(i, j) = 0;
            // color contrast method
            // saturate_cast<uchar>(alpha*img.at<T>(i, x)[c] + beta);
        }
    }
    return new_image;
}


// useless???
template <typename T>
Mat summedAreaTable(Mat img) {
    Mat table = Mat::zeros(img.size(), img.type());
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            table.at<T>(i, j) =
                img.at<T>(i, j) +
                (i > 0 ? table.at<T>(i - 1, j) : 0) +
                (j > 0 ? table.at<T>(i, j - 1) : 0) -
                (i > 0 && j > 0 ? table.at<T>(i - 1, j - 1) : 0);
        }
    }
    return table;
}

// useless???
uchar localAvg(const Mat table, int row, int column, int window_size) {
    uchar local;
    uchar A, B, C, D;
    A = table.at<uchar>(max(row - window_size / 2, 0), max(column - window_size / 2, 0));
    B = table.at<uchar>(max(row - window_size / 2, 0), min(column + window_size / 2, table.cols - 1));
    C = table.at<uchar>(min(row + window_size / 2, table.rows - 1), max(column - window_size / 2, 0));
    D = table.at<uchar>(min(row + window_size / 2, table.rows - 1), min(column + window_size / 2, table.cols - 1));
    return  D + A - B - C;
}

template <typename Vec_T>
void generateThresholds(vector<Vec_T>& pyramid) {

    //for from end to start with calculating threshold map
    for (auto current = rbegin(pyramid); current != rend(pyramid); current++) {
        if (current == rbegin(pyramid)) // for last layer have to use another method
        {
            (*current)->calculateTreshold();
        }
        else {
            (*current)->recalculate(*(*prev(current, 1))->threshold);
        }
    }
}

template <typename Vec_T, typename T>
auto createPyramid(const Mat& src) {
    vector<Vec_T> pyramid;

    //initialize first layer
    Layer<T>* l = new Layer<T>(src);
    pyramid.push_back(l);

    //while layer.width != 0 create new layer based on previous
    for (; l->width != 0; l = new Layer<T>(*l)) {
        pyramid.push_back(l);
    }

    return pyramid;
}