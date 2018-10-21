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
#include <SFML/Window/Joystick.hpp>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>

#include "config.h"
#include "CameraManager.h"
#include "Lanefinder.h"

#define FREQ 4
#define LAST_SEEN 5

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

double calculateCurvature(Mat& frame, LaneCurve l);
double distanceToCurve(float a_o, float b_o, float c_o, float x, float y);
double distanceToCurve(float a_o, float b_o, float c_o, Point p);
double distanceToCurve(LaneCurve l, Point p);
Point traverseCurve(LaneCurve l, double x, int dist);
void intHandler(int x);
void drawCurve(Mat& frame, LaneCurve l, Scalar col = Scalar(23, 152, 213));
tuple<double, double, double> curveOfBestFit(vector<Point> points);

// Global network link
int sock;

int main(int arg_c, char** arg_v) {
	signal(SIGINT, intHandler);

	// Flags
	bool debug = false;
	bool show_hud = true;
	bool paused = false;
	bool is_autonomous = false;
	bool send_cmd = true;

	// Parse command line args
	for (int i = 1; i < arg_c; i++) {
		if (strcmp(arg_v[i], "-d") == 0) {
			debug = true;
			cout << "AutoPilot: Debuging mode..." << endl;
		}
		if (strcmp(arg_v[i], "-nohud") == 0) {
			show_hud = false;
		}
		if (strcmp(arg_v[i], "-nobody") == 0) {
			send_cmd = false;
		}
	}

	cout << "AutoPilot: Initiating system" << endl;

	if (debug) {
		CameraManager::addCapture("../Data/Demo sidewalk.mp4");
		CameraManager::addCapture("../Data/shadow-test.mp4");
	}
	else {
		CameraManager::addCapture(0);
		//CameraManager::addCapture("Data/Demo sidewalk.mp4");
	}

	Lanefinder::initialize();
	CameraManager::update();

	Mat original_frame;  // Raw frame

	// Lane parameters
	vector<vector<Point> > lane_lines[2];
	vector<tuple<double, double, double> > curves;  // a, b, c
	vector<LaneCurve> lane_memory;

	Point left_lane_anchor(0, CameraManager::getFrame(0)->rows);
	Point right_lane_anchor(CameraManager::getFrame(0)->cols, CameraManager::getFrame(0)->rows);
	vector<vector<LaneCurve>::iterator> curr_lanes;
	vector<LaneCurve>::iterator curr_left_lane, curr_right_lane;

	bool first_loop = true;

	chrono::milliseconds now, then, nt_diff;

	// Speed control parameters
	float drv = 50;
	float brk = 50;

	// Steer control parameters
	float theory_wheel_angle = CV_PI / 2;
	float str_pos = 0;
	float str_accel = 50;
	float curr_trgt = 0, prev_trgt = 0;
	float curr_trgt_right = 0, curr_trgt_left = 0, prev_trgt_right = 0, prev_trgt_left = 0;

	// Networking setup
	char cmd[50];
	struct sockaddr_in dst;

	sock = socket(AF_INET, SOCK_STREAM, 0);

	memset(&dst, 0, sizeof(dst));
	dst.sin_family = AF_INET;
	dst.sin_addr.s_addr = inet_addr("169.254.0.2");
	//dst.sin_addr.s_addr = inet_addr("192.168.0.116");
	dst.sin_port = htons(8061);

	if (send_cmd)
		connect(sock, (struct sockaddr*)&dst, sizeof(struct sockaddr_in));

	cout << "AutoPilot: ready" << endl;
	//cout << flush;

	//VideoWriter w("rec Frame.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(CameraManager::getFrame(0)->cols, CameraManager::getFrame(0)->rows), true);

	while (true) {
		while (paused) {
			paused = char(waitKey(10)) != 'p';
		}

		//now = time(NULL);
		now = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());

		CameraManager::update();
		CameraManager::update();  // Every other frame
		CameraManager::update();
		CameraManager::update();
		CameraManager::update();
		sf::Joystick::update();

		// Raw frame
		original_frame = CameraManager::getFrame(0)->clone();

		if (sf::Joystick::isButtonPressed(0, 9))
			is_autonomous = !is_autonomous;

		if (sf::Joystick::isButtonPressed(0, 8))
			theory_wheel_angle = CV_PI / 2;

		if (!is_autonomous) {
			str_accel = (((sf::Joystick::getAxisPosition(0, sf::Joystick::X) / 100.0) + 1.0) / 2) * 100;
			drv = (((sf::Joystick::getAxisPosition(0, sf::Joystick::V) / 100.0) + 1.0) / 2 * 50) + 50;

			brk = 50;
			brk += sf::Joystick::isButtonPressed(0, 4) ? 50 : 0; 
			brk += sf::Joystick::isButtonPressed(0, 5) ? -50 : 0;
		}
		else {
			lane_lines[0].clear();
			lane_lines[1].clear();
			curves.clear();
			lane_memory.clear();
			curr_lanes.clear();

			Lanefinder::find(*CameraManager::getFrame(0), lane_lines[0]);
			//Lanefinder::find(*CameraManager::getFrame(1), lane_lines[1]);

			// Map curves of best fit
			for (auto it = lane_lines[0].begin(); it != lane_lines[0].end(); it++) {
				curves.push_back(curveOfBestFit(*it));
			}

			// Update lane memory
			for (int i = 0; i < curves.size(); i++) {
				int m_index;
				for (m_index = 0; m_index < lane_memory.size(); m_index++) {
					// Curve parameters for quick use
					double a1 = get<0>(curves[i]);
					double b1 = get<1>(curves[i]);
					double c1 = get<2>(curves[i]);
					double a2 = lane_memory[m_index].a;
					double b2 = lane_memory[m_index].b;
					double c2 = lane_memory[m_index].c;

					// PsOI parameters
					double x[4];

					double det = pow(b2 - b1, 2) - 4 * (a2 - a1) * (c2 - c1);

					x[0] = 0;
					x[1] = std::max(det < 0 ? 0 : (- (b2 - b1) - sqrt(det)) / (2 * (a2 - a1)), 0.0);
					x[2] = std::min(det < 0 ? CameraManager::getFrame(0)->cols : (- (b2 - b1) + sqrt(det)) / (2 * (a2 - a1)), (double)CameraManager::getFrame(0)->cols);
					x[3] = CameraManager::getFrame(0)->cols;

					double diff = 0.0;  // Area between curves - No
					for (int i = 0; i < 3; i++) {
						diff += abs((pow(x[i + 1], 3) * (a1 - a2) / 3 + pow(x[i + 1], 2) * (b1 - b2) / 2 + x[i + 1] * (c1 - c2)) - (pow(x[i], 3) * (a1 - a2) / 3 + pow(x[i], 2) * (b1 - b2) / 2 + x[i] * (c1 - c2)));
					}

					//if (diff < 40000)
					//	break;
				}

				if (m_index == lane_memory.size()) {  // New lane
					lane_memory.push_back(LaneCurve(curves[i], 0, 0));
				}
				else {
					lane_memory[m_index].a = get<0>(curves[i]);
					lane_memory[m_index].b = get<1>(curves[i]);
					lane_memory[m_index].c = get<2>(curves[i]);
					lane_memory[m_index].last_seen = -1;
				}

				lane_memory[m_index].line_segs_index = i;
				lane_memory[m_index].freq++;
			}

			// Show Current lanes
			if (show_hud) {
				for (int i = 0; i < lane_memory.size(); i++) {
					drawCurve(*CameraManager::getFrame(0), lane_memory[i], i >= lane_memory.size() / 2 ? Scalar(0, 0, 255) : Scalar(255, 0, 0));
				}

				/*
				if (curr_left_lane != lane_memory.end()) {
					drawCurve(*CameraManager::getFrame(0), *curr_left_lane, Scalar(255, 0, 0));
				}
				if (curr_right_lane != lane_memory.end()) {
					drawCurve(*CameraManager::getFrame(0), *curr_right_lane, Scalar(0, 0, 255));
				}
				*/
			}

			// ------ CONTROL LOGIC ------
			curr_trgt = 0;
			curr_trgt_right = 0;
			curr_trgt_left = 0;

			int count = 0;
			for (auto l : lane_memory) {
				bool is_right = count >= lane_memory.size() / 2;  // true if curve represents right lane

				// All derivatives of polynomial
				float derv_0 = (l.eval_y(is_right ? CameraManager::getFrame(0)->cols : 0) - CameraManager::getFrame(0)->rows) / 100;
				float derv_1 = 0;  // TODO: Use first derivative to assign this vallue
				float derv_2 = (is_right ? -1 : 1) * calculateCurvature(*CameraManager::getFrame(0), l);

				int* low_x = l.eval_x(CameraManager::getFrame(0)->rows);
				if (low_x != NULL) {
					if (abs(low_x[0] - CameraManager::getFrame(0)->cols / 2) <= abs(low_x[1] - CameraManager::getFrame(0)->cols / 2))
						derv_1 += (CV_PI / 2) - atan2(2 * l.a * low_x[0] + l.b, 1.0) + CV_PI;
					else if (abs(low_x[0] - CameraManager::getFrame(0)->cols / 2) > abs(low_x[1] - CameraManager::getFrame(0)->cols / 2))
						derv_1 += (CV_PI / 2) - atan2(2 * l.a * low_x[1] + l.b, 1.0) + CV_PI;
				}

				if (is_right) {
					curr_trgt_right = derv_0 + -(derv_1 == 0 ? 0 : 1 / derv_1) + derv_2;
					curr_trgt_right = abs(curr_trgt_right - prev_trgt_right) > 50 ? 0 : (abs(curr_trgt_right) <= 100 ? curr_trgt_right : 100 * curr_trgt_right / abs(curr_trgt_right));
				}
				else {
					curr_trgt_left = derv_0 + (derv_1 == 0 ? 0 : 1 / derv_1) + derv_2;
					curr_trgt_left = abs(curr_trgt_left - prev_trgt_left) > 50 ? 0 : (abs(curr_trgt_left) <= 100 ? curr_trgt_left : 100 * curr_trgt_left / abs(curr_trgt_left));
				}
				count++;
			}

			curr_trgt += curr_trgt_left / 2.0 + curr_trgt_right / 2.0;

			// Cap steering
			if (abs(str_pos) > 100)
				str_pos = 100 * str_pos / abs(str_pos);

			// Frame difference calculation of steering
			str_pos = curr_trgt - prev_trgt;
			curr_trgt = (curr_trgt + str_pos) / 2;
			str_accel = 50 + 50 * ((CV_PI / 2 - ((-curr_trgt + 100) / 200) * CV_PI)) / (11 * CV_PI / 36);  // 50 + 50 * (curr_angle - trgt_angle) / max_angle

			// Aplify output if steering is not enough
			if (str_pos != 0 && abs(curr_trgt) - abs(prev_trgt) > 0) {  // Moving away from target
				str_accel = (str_accel - 50) * 2 + 50;
			}

			// Speed
			drv = 60;
			brk = 50;
		}

		// Update time diff
		if (first_loop) {
			prev_trgt = curr_trgt;
			then = now;
			first_loop = false;

			continue;
		}
		else {
			nt_diff = now - then;

			theory_wheel_angle += (193 * CV_PI / 1440) * -(floor(str_accel) / 50.0 - 1.0) * nt_diff.count() / 1000;
			// Bound angle
			/*
			if (theory_wheel_angle + (193 * CV_PI / 1440) * -(floor(str_accel) / 50.0 - 1.0) * nt_diff.count() / 1000 < CV_PI / 2 + 83 * CV_PI / 360 &&
			    theory_wheel_angle + (193 * CV_PI / 1440) * -(floor(str_accel) / 50.0 - 1.0) * nt_diff.count() / 1000 > CV_PI / 2 - 11 * CV_PI / 36)
			{
				theory_wheel_angle += (193 * CV_PI / 1440) * -(floor(str_accel) / 50.0 - 1.0) * nt_diff.count() / 1000;
			}
			else {
				str_accel = 50;
			}
			*/
		}

		// Show data
		if (show_hud) {
			int gval = 100;
			putText(*CameraManager::getFrame(0), "TRGT: " + to_string((int)curr_trgt), Point(30, 60), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2); 
			putText(*CameraManager::getFrame(0), "POS: " + to_string(str_pos), Point(30, 90), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2); 
			putText(*CameraManager::getFrame(0), "ACCEL: " + to_string(str_accel - 50), Point(30, 120), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2);
			putText(*CameraManager::getFrame(0), ((abs(curr_trgt - prev_trgt) < 1.6) ? "Staight" : ((curr_trgt - prev_trgt > 0) ? "Left" : "Right")), Point(30, 150), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2);
			putText(*CameraManager::getFrame(0), "DIFF RIGHT: " + to_string(curr_trgt_right - prev_trgt_right), Point(30, 180), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2);
			putText(*CameraManager::getFrame(0), "DIFF LEFT: " + to_string(curr_trgt_left - prev_trgt_left), Point(30, 210), FONT_HERSHEY_PLAIN, 1.0, Scalar(gval, gval, gval), 2);

			/*
			circle(original_frame, Point(100, original_frame.rows / 2 - 15), 2, Scalar(255, 0, 0), 3);
			circle(original_frame, Point(original_frame.cols - 100, original_frame.rows / 2 - 15), 2, Scalar(255, 0, 0), 3);
			circle(original_frame, Point(original_frame.cols, original_frame.rows - 10), 2, Scalar(255, 0, 0), 3);
			circle(original_frame, Point(0, original_frame.rows - 10), 2, Scalar(255, 0, 0), 3);
			*/

			// Wheel indicator
			float steer_angle = ((-str_pos + 100) / 200) * CV_PI;  // Pink
			float trgt_steer_angle = ((-curr_trgt + 100) / 200) * CV_PI;  // Black
			line(*CameraManager::getFrame(0), Point(original_frame.cols / 2, 300), Point(original_frame.cols / 2 + 100 * cos(steer_angle), 300 - 100 * sin(steer_angle)), Scalar(145, 45, 243), 2);
			line(*CameraManager::getFrame(0), Point(original_frame.cols / 2, 300), Point(original_frame.cols / 2 + 100 * cos(trgt_steer_angle), 300 - 100 * sin(trgt_steer_angle)), Scalar(0, 0, 0), 2);
			line(*CameraManager::getFrame(0), Point(original_frame.cols / 2, 300), Point(original_frame.cols / 2 + 100 * cos(theory_wheel_angle), 300 - 100 * sin(theory_wheel_angle)), Scalar(180, 180, 180), 2);


			rectangle(*CameraManager::getFrame(0), Point(0, 0), Point(200, 15), Scalar(168, 168, 168), CV_FILLED);

			// Top bar min-HUD
			putText(*CameraManager::getFrame(0), to_string(nt_diff.count()) + "ms", Point(5, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255), 1);  // Latency
			putText(*CameraManager::getFrame(0), is_autonomous ? "auto" : "remote", Point(65, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255), 1);  // Control mode
			putText(*CameraManager::getFrame(0), to_string((int)drv), Point(120, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255), 1);  // Drive
			putText(*CameraManager::getFrame(0), to_string((int)str_accel), Point(150, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255), 1);  // Steer
			putText(*CameraManager::getFrame(0), to_string((int)brk), Point(180, 12), FONT_HERSHEY_PLAIN, 0.8, Scalar(255, 255, 255), 1);  // Break

			imshow("Original", original_frame);
			imshow("Frame", *CameraManager::getFrame(0));
			//w.write(*CameraManager::getFrame(0));

			if (char(waitKey(10)) == 'p')
				paused = !paused;
		}

		// Send Data
		strcpy(cmd, ("drv " + to_string((int)drv) + " str " + to_string((int)((str_accel - 50) / 1.0 + 50)) + " brk " + to_string((int)brk) + "\n").c_str());
		//cout << cmd << flush;

		if (send_cmd)
			send(sock, cmd, strlen(cmd), 0);

		prev_trgt = curr_trgt;
		prev_trgt_right = curr_trgt_right;
		prev_trgt_left = curr_trgt_left;

		//lane_lines_1.clear();
		//lane_lines_2.clear();
		curves.clear();

		then = now;
	}

	return 0;
}

double calculateCurvature(Mat& frame, LaneCurve l) {
	// Derivative parameters
	float a_d = 2 * l.a;
	float b_d = l.b;

	return 2 * l.a * 1000;
}

double distanceToCurve(float a_o, float b_o, float c_o, float x_o, float y_o) {
	// Parameters of the derivative of the given curve
        float a = 2 * pow(a_o, 2);
        float b = 3 * b_o * a_o;
        float c = (pow(b_o, 2) + 2 * c_o * a_o - 2 * a_o * y_o + 1);
        float d = (c_o * b_o - b_o * y_o - x_o);

        float f = (3 * c / a - pow(b, 2) / pow(a, 2)) / 3;
        float g = (2 * pow(b, 3) / pow(a, 3) - 9 * b * c / pow(a, 2) + 27 * d / a) /  27;
        float h = pow(g, 2) / 4 + pow(f, 3) / 27;

        float y = numeric_limits<float>::max();
        float x = numeric_limits<float>::infinity();

        if (h <= 0) {  // 3 distinct roots
                float roots[3];

                float i = sqrt(pow(g, 2) / 4 - h);
                float k = acos(g / (-2 * i));
                float n = sqrt(3) * sin(k / 3);

                roots[0] = 2 * cbrt(i) * cos(k / 3) - b / (3 * a);
                roots[1] = -cbrt(i) * (cos(k / 3) + n) - b / (3 * a);
                roots[2] = -cbrt(i) * (cos(k / 3) - n) - b / (3 * a);

                // Find loweset y for roots
                for (int i = 0; i < 3; i++) {
                        float y_tmp = a_o * pow(roots[i], 2) + b_o * roots[i] + c_o;

                        if (y_tmp == (y = min(y_tmp, y))) {
                                x = roots[i];
                        }
                }
        }
        else if (h == 0 && f == 0 && g == 0) {  // 1 distinct root
                x = -cbrt(d / a);
                y = a_o * pow(x, 2) + b_o * x + c_o;
        }
        else if (h > 0) {  // 1 real root
                float r = -g / 2 + sqrt(h);
                float t = -g / 2 - sqrt(h);

                x = cbrt(r) + cbrt(t) - b / (3 * a);
                y = a_o * pow(x, 2) + b_o * x + c_o;
        }

        return sqrt(pow(x - x_o, 2) + pow(y - y_o, 2));
}

double distanceToCurve(float a_o, float b_o, float c_o, Point p) {
	return distanceToCurve(a_o, b_o, c_o, p.x, p.y);
}

double distanceToCurve(LaneCurve l, Point p) {
	return distanceToCurve(l.a, l.b, l.c, p.x, p.y);
}

/*
 * Description:	Get point that is dist distance along the curve
 *		from the point at x on the curve
 * Parameters:	LaneCurve: l    - Lane to traverse
 *		Double:     x    - x value of start point
 *		Integer:    dist - Distance to traverse
 * Returns:	Point - Coords after traversal
**/
Point traverseCurve(LaneCurve l, double x, int dist) {
        float curr_d = 0;
        Point p2;

        while (curr_d < abs(dist)) {
                Point p1(x, l.eval_y(x));
                x += dist / abs(dist);
                p2 = Point(x, l.eval_y(x));

                Point diff = p1 - p2; 
                curr_d += abs(sqrt(pow(diff.x, 2) + pow(diff.y, 2)));
        }   

        return p2; 
}

void intHandler(int x) {
	cout << "AutoPilot: Shutting down" << endl;
	CameraManager::release();

	close(sock);

	exit(0);
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
        for (int x = 0; x < 850; x++) {
                circle(frame, Point(x, l.eval_y(x)), 1, col, 2);
        }
}

/*
 * Description:	Calculate curve of best fit
 * Parameters:	List of Points: points - Points to find best fit curve from
 * Returns:	3-Tuple - a, b, c
**/
tuple<double, double, double> curveOfBestFit(vector<Point> points) {
	Mat design(points.size(), 3, CV_64FC1);
	Mat y_vec(points.size(), 1, CV_64FC1);

	for (int i = 0; i < points.size(); i++) {
		design.at<double>(i, 0) = 1.0;
		design.at<double>(i, 1) = points[i].x;
		design.at<double>(i, 2) = pow(points[i].x, 2);

		y_vec.at<double>(i, 0) = points[i].y;
	}

	Mat std_poly = ((design.t() * design).inv() * design.t()) * y_vec;

	double a = std_poly.at<double>(0, 2);
	double b = std_poly.at<double>(0, 1);
	double c = std_poly.at<double>(0, 0);

	return tuple<double, double, double>(a, b, c);
}
