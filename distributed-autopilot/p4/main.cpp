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
	vector<LaneCurve> lane_memory;
	Mat frame(850, 480, CV_8UC3, Scalar(0, 0, 0));  // TODO: Set size

	while (true) {
		for (int i = 0; i < lane_memory.size(); i++) {
			drawCurve(frame, lane_memory[i], i >= lane_memory.size() / 2 ? Scalar(0, 0, 255) : Scalar(255, 0, 0));
		}

		imshow("FRAME", frame);
		waitKey(10);
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
