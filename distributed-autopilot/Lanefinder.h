#ifndef LANEFINDER_H_
#define LANEFINDER_H_

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

class Lanefinder {
	static int canny_thresh;
	//static cv::VideoWriter w2;

public:
	static void initialize();
	static float scoreColour(cv::Vec3b col);
	static bool isRoad(cv::Vec3b col);
	static void find(cv::Mat& frame, std::vector<std::vector<cv::Point> >& lane_lines);
	static void scorePixels(cv::Mat& frame, std::vector<std::vector<std::pair<cv::Point, cv::Point> > >& lane_lines);
};

#endif
