#include <iostream>
#include "CameraManager.h"

std::vector<cv::VideoCapture> CameraManager::captures;
std::vector<cv::Mat> CameraManager::frames;

/*
 * Description:	Get new frames
 * Parameters:	void
 * Returns:	Boolean - True if all frames captured fine
**/
bool CameraManager::update() {
	bool stat = true;

	for (int i = 0; i < captures.size(); i++) {
		stat = stat && captures[i].read(frames[i]);
	}

	return stat;
}

/*
 * Description:	Release capture resources
 * Parameters:	void
 * Returns:	void
**/
void CameraManager::release() {
	for (int i = 0; i < captures.size(); i++) {
		captures[i].release();
	}
}

/*
 * Description:	New capture source
 * Parameters:	Integer: id - Camera id
 * Returns:	Boolean - True if capture opens correctly
**/
bool CameraManager::addCapture(int id) {
	cv::VideoCapture cap(id);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 850);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (cap.isOpened()) {
		captures.push_back(cap);
		frames.push_back(cv::Mat());

		return true;
	}

	std::cout << "Camera " << id << " not available" << std::endl;

	return false;
}

/*
 * Description:	New capture source
 * Parameters:	String: s - Name of file to capture
 * Returns:	Boolean - True if capture opens correctly
**/
bool CameraManager::addCapture(std::string s) {
	cv::VideoCapture cap(s);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 850);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (cap.isOpened()) {
		captures.push_back(cap);
		frames.push_back(cv::Mat());

		return true;
	}

	std::cout << "Failed to load " << s << std::endl;

	return false;
}

/*
 * Description:	Get captured frame
 * Parameters:	Integer: i - Index of frame
 * Returns:	Matrix Pointer - Frame of index i. NULL if IOB
**/
cv::Mat* CameraManager::getFrame(int i) {
	if (i >= frames.size() || i < 0) return NULL;

	return &frames[i];
}

/*
 * Description:	Get capture resource
 * Parameters:	Integer: i = Index of capture
 * Returns:	Capture Pointer - Capture of index i. NULL if IOB
**/
cv::VideoCapture* CameraManager::getCapture(int i) {
	if (i >= captures.size() || i < 0) return NULL;

	return &captures[i];
}

