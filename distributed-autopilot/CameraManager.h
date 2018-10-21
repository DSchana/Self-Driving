#ifndef CAMERA_MANAGER_H_
#define CAMERA_MANAGER_H_

#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

// Camera properties (Microsoft HD 3000)
#define FOCAL_LENGTH 60
#define FLANGE_FOCAL_DISTANCE 14.0169
#define CAMERA_DIFFERENCE 95
#define LIGHT_SENSOR_WIDTH 17.3
#define LIGHT_SENSOR_HEIGHT 13
#define HORIZONTAL_FOV 61

class CameraManager {
	static std::vector<cv::VideoCapture> captures;
	static std::vector<cv::Mat> frames;

public:
	static bool update();
	static void release();
	static bool addCapture(int id);
	static bool addCapture(std::string s);
	static cv::Mat* getFrame(int i);
	static cv::VideoCapture* getCapture(int i);
};

#endif
