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

int main() {
	// Out socket setup
	int out_sock;
	struct sockaddr_in out_dst;

	out_sock = socket(AF_INET, SOCK_STREAM, 0);

	memset(&out_dst, 0, sizeof(out_dst));
	out_dst.sin_family = AF_INET;
	out_dst.sin_addr.s_addr = inet_addr("192.168.0.116");
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

	listen(in_sock, 1);
	int consock = accept(in_sock, (struct sockaddr*)&in_dst, &sock_size);

	// Lane parameters
	vector<vector<Point> > lane_lines(2);
	vector<tuple<double, double, double> > curves;
	vector<LaneCurve> lane_memory(2);
	vector<vector<LaneCurve>::iterator> curr_lanes;
	vector<LaneCurve>::iterator curr_left_lane, curr_right_lane;

	while (true) {
		// TODO: Know sizeof lane_lines
		recv(consock, &lane_lines, sizeof(lane_lines), 0);

		// Map curves of best fit
		for (auto it = lane_lines.begin(); it != lane_lines.end(); it++) {
			vector<Point> points = *it;

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

			curves.push_back(tuple<double, double, double>(a, b, c));
		}

		// Update lane memory
		for (int i = 0; i < curves.size(); i++) {
			lane_memory[i] = LaneCurve(curves[i], 0, 0);
		}

		send(out_sock, &lane_memory, sizeof(lane_memory), 0);
	}

	return 0;
}
