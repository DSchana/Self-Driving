#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <netinet/in.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "../CameraManager.h"

#define FREQ 4
#define LAST_SEEN 5
#define TILE_SIZE 10

using namespace std;
using namespace cv;

struct LaneCurve {
	double a, b, c;
	int line_segs_index;
	int freq, last_seen;

	LaneCurve() {}

	LaneCurve(double n_a, double n_b, double n_c, int f, int l) :
		a(n_a),
		b(n_b),
		c(n_c),
		freq(f),
		last_seen(l)
	{   
	}   

	LaneCurve(tuple<double, double, double> c_param, int f, int l) :
		a(get<0>(c_param)),
		b(get<1>(c_param)),
		c(get<2>(c_param)),
		freq(f),
		last_seen(l)
	{   
	}   

	Point vertex() {
		double h = -b / (2 * a == 0 ? numeric_limits<double>::min() : 2 * a); 
		double k = c - (a * pow(h, 2));

		return Point(h, k); 
	}   

	double eval_y(int x) {
		return a * pow(x, 2) + b * x + c;
	}

	int* eval_x(int y) {
		static int ret[2];
		double dis = pow(b, 2) - 4 * a * (c - y);

		if (dis < 0)
			return NULL;

		ret[0] = (int)((pow(b, 2) - sqrt(dis)) / (2 * a));
		ret[1] = (int)((pow(b, 2) + sqrt(dis)) / (2 * a));

		return ret;
	}
};

void drawCurve(Mat& frame, LaneCurve l, Scalar col = Scalar(23, 152, 213));

int main() {
	Mat frame, frame_crop, frame_gray, frame_hsv, frame_hls, frame_lab, detected_edges;
	Rect crop;
	vector<vector<Point> > lane_lines;

	// Out socket setup
	int out_sock;
	struct sockaddr_in out_dst;

	out_sock = socket(AF_INET, SOCK_STREAM, 0);

	memset(&out_dst, 0, sizeof(out_dst));
	out_dst.sin_family = AF_INET;
	out_dst.sin_addr.s_addr = inet_addr("192.168.0.117");
	out_dst.sin_port = htons(8060);

	// In socket setup
	int in_sock;
	struct sockaddr_in in_dst;
	struct sockaddr_in in_serv;
	socklen_t sock_size = sizeof(struct sockaddr_in);

	memset(&in_serv, 0, sizeof(in_serv));
	in_serv.sin_family = AF_INET;
	in_serv.sin_addr.s_addr = htonl(INADDR_ANY);
	in_serv.sin_port = htons(8061);

	in_sock = socket(AF_INET, SOCK_STREAM, 0);
	bind(in_sock, (struct sockaddr*)&in_serv, sizeof(struct sockaddr));

	// Sensor setup
	CameraManager::addCapture("../../Data/Demo sidewalk.mp4");

	pid_t pid;

	if ((pid = fork()) == 0) {  // Child fork for display
		listen(in_sock, 1);
		int consock = accept(in_sock, (struct sockaddr*)&in_dst, &sock_size);

		vector<LaneCurve> lane_memory(2);

		while (true) {
			recv(consock, &lane_memory, sizeof(lane_memory), 0);

			for (int i = 0; i < lane_memory.size(); i++) {
				drawCurve(*CameraManager::getFrame(0), lane_memory[i], i >= lane_memory.size() / 2 ? Scalar(0, 0, 255) : Scalar(255, 0, 0));
			}

			imshow("Lanes", *CameraManager::getFrame(0));
			waitKey(10);
		}
	}
	else {
		///connect(out_sock, (struct out_sockaddr*)&out_dst, sizeof(struct out_sockaddr_in));

		while (true) {
			CameraManager::update();

			frame = CameraManager::getFrame(0)->clone();

			crop = Rect(0, 0, frame.cols, frame.rows);
			//crop = Rect(0, frame.rows / 2 + 40, frame.cols, frame.rows - frame.rows / 2 - 40);  // Bottom half

			Mat road_mask(frame.rows, frame.cols, CV_8UC1, Scalar(0));
			Mat element = getStructuringElement(0, Size(5, 5), Point(2, 2));

			// Image wrap parameters
			Point2f input_quad[4], output_quad[4];
			Mat lambda(2, 4, CV_32FC1);

			lambda = Mat::zeros(frame.rows, frame.cols, frame.type());

			// Original coords
			/*
			input_quad[0] = Point2f(300, frame.rows / 2 + 25);
			input_quad[1] = Point2f(frame.cols - 300, frame.rows / 2 + 25);
			input_quad[2] = Point2f(frame.cols, frame.rows - 10);
			input_quad[3] = Point2f(0, frame.rows - 10);
			*/
			input_quad[0] = Point2f(100, frame.rows / 2 - 10);
			input_quad[1] = Point2f(frame.cols - 100, frame.rows / 2 - 10);
			input_quad[2] = Point2f(frame.cols, frame.rows - 10);
			input_quad[3] = Point2f(0, frame.rows - 10);

			// Destination coords
			output_quad[0] = Point2f(0, 0);
			output_quad[1] = Point2f(frame.cols, 0);
			output_quad[2] = Point2f(frame.cols, frame.rows);
			output_quad[3] = Point2f(0, frame.rows);

			// Warp perspective
			lambda = getPerspectiveTransform(input_quad, output_quad);
			warpPerspective(frame, frame, lambda, frame.size());
		
			frame_crop = frame(crop);

			erode(frame_crop, frame_crop, element);
			dilate(frame_crop, frame_crop, element);
			GaussianBlur(frame_crop, frame_crop, Size(5, 5), 0, 0);

			cvtColor(frame_crop, frame_gray, CV_BGR2GRAY);
			cvtColor(frame_crop, frame_hsv, CV_BGR2HSV);
			cvtColor(frame_crop, frame_hls, CV_BGR2HLS);
			cvtColor(frame_crop, frame_lab, CV_BGR2Lab);

			// Score Pixels
			Mat score = Mat(frame_crop.rows, frame_crop.cols, CV_8UC3, Scalar(0, 0, 0));
			Mat tmp, tmp_show;
			vector<Mat> chan(3);

			Ptr<CLAHE> clahe = createCLAHE();
			clahe->setTilesGridSize(Size(TILE_SIZE, TILE_SIZE));
			clahe->setClipLimit(2);

			// Extract b channel
			split(frame_lab, chan);
			clahe->apply(chan[2], tmp);
			tmp.copyTo(chan[2]);
			merge(chan, tmp);
			normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);

			threshold(tmp, tmp, 150 * 0.85, 1, THRESH_BINARY);  // 150 - day
			add(score, tmp, score);

			// Extract l channel
			split(frame_hls, chan);
			clahe->apply(chan[1], tmp);
			tmp.copyTo(chan[1]);
			merge(chan, tmp);
			normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);

			threshold(tmp, tmp, 210 * 0.85, 1, THRESH_BINARY);  // 210 - day
			add(score, tmp, score);

			// Extract v channel
			clahe->setClipLimit(6);
			split(frame_hsv, chan);
			clahe->apply(chan[2], tmp);
			tmp.copyTo(chan[2]);
			merge(chan, tmp);
			normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);

			threshold(tmp, tmp, 220 * 0.85, 1, THRESH_BINARY);  // 220 - old
			add(score, tmp, score);
			normalize(score, score, 0, 255, NORM_MINMAX);

			resize(score, score, Size(), 1.0 / TILE_SIZE, 1.0 / TILE_SIZE);

			imshow("score", score * 255);  // TMP

			Mat b(score.rows, score.cols, CV_8UC1);
			Mat l(score.rows, score.cols, CV_8UC1);
			Mat v(score.rows, score.cols, CV_8UC1);

			Mat out[] = { b, l, v };
			int from_to[] = { 0,0, 1,1, 2,2 };

			mixChannels(&score, 1, out, 3, from_to, 3);

			Mat mask = b.mul(l) + b.mul(-l).mul(-v) + (-b).mul(l);
			cout << sizeof(mask) << " " << mask.cols << " " << mask.rows << endl;

			normalize(mask, mask, 0, 255, NORM_MINMAX);
			imshow("mask", mask);
			waitKey(10);

			send(out_sock, &mask, sizeof(mask), 0);
		}
	}

	return 0;
}

/*
 * Description: Display curve
 * Parameters:  Matrix: frame - Where to draw the curve
 *              Float:  a     - a value of curve in vertex form
 *              Point:  v     - vertex of curve
 *              Scalar: col   - colour to draw curve in
 * Returns:     void
**/
void drawCurve(Mat& frame, LaneCurve l, Scalar col) {
	for (int x = 0; x < frame.cols; x++) {
		circle(frame, Point(x, l.eval_y(x)), 1, col, 2);
	}
}
