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

#define LANE_PROBABILITY_SEARCH_RANGE 12
#define LANE_PROBABILITY_THRESHOLD 0.5

using namespace std;
using namespace cv;

int main() {
	// TODO: Get mask from in_sock
	
	// Out socket setup
	int out_sock;
	struct sockaddr_in out_dst;

	out_sock = socket(AF_INET, SOCK_STREAM, 0);

	memset(&out_dst, 0, sizeof(out_dst));
	out_dst.sin_family = AF_INET;
	out_dst.sin_addr.s_addr = inet_addr("192.168.0.118");
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

	Mat mask(85, 48, CV_8UC1, Scalar(0));

	// Find left and right lane edges
	vector<vector<Point> > lane_lines(2);

	while (true) {
		recv(consock, &mask, sizeof(mask), 0);

		int void_counter = 0;

		Point prev_lane_mark_left, prev_lane_mark_right;
		float avg_left_x_diff = 0, avg_right_x_diff = 0;
		int left_count = 0, right_count = 0;

		for (int y = 0; y < mask.rows; y++) {
			bool l_found = false, r_found = false;
			float max_col_score_left = 0.0, max_col_score_right = 0.0;
			bool curr_is_road = false, prev_is_road = false;

			vector<tuple<Point, Point, double> > pot_road_seg;  // (Start, End, Probability of being road)
			tuple<Point, Point, double> most_prob_seg;  // Most probable road segment

			for (int x = 0; x < mask.cols; x++) {
				Point curr(x, y);
				float l_avg = 0, r_avg = 0;
				int l_count = 0, r_count = 0;

				for (int c = x - LANE_PROBABILITY_SEARCH_RANGE; c <= x + LANE_PROBABILITY_SEARCH_RANGE; c += 2) {
					if (c < 0 || c >= mask.cols) {
						continue;
					}

					if (c < x) {  // Left of point
						l_avg += mask.at<unsigned int>(Point(c, y)) ? 1 : 0;
						l_count++;
					}
					else if (c > x) {  // Right of point
						r_avg += mask.at<unsigned int>(Point(c, y)) ? 1 : 0;
						r_count++;
					}
				}

				l_avg = l_count == 0 ? 0 : l_avg / (float)l_count;
				r_avg = r_count == 0 ? 0 : r_avg / (float)r_count;

				float curr_prob_road = l_avg + r_avg;

				if (l_count != 0 && r_count != 0) 
					curr_prob_road /= 2;

				curr_is_road = curr_prob_road > LANE_PROBABILITY_THRESHOLD;

				// Update road segments
				if (curr_is_road && !prev_is_road)  // Start new segment
					pot_road_seg.push_back(make_tuple(curr, Point(mask.cols + 1, y), 0));
				else if (!curr_is_road && prev_is_road)  // End segment
					get<1>(*(pot_road_seg.end() - 1)) = curr;

				prev_is_road = curr_is_road;
			}

			int longest_chain = 0;

			for (auto seg : pot_road_seg) {
				int curr_chain = get<1>(seg).x - get<0>(seg).x;

				// Calculate percentage of current segment is above previously accepted one
				if (y >= mask.rows - 1) {  // First row of road segments
					get<2>(seg) = 1.0;
				}
				else {
					double below_count = get<1>(seg).x - get<0>(seg).x;
					below_count -= get<0>(seg).x < prev_lane_mark_left.x ? abs(prev_lane_mark_left.x - get<0>(seg).x) : 0;
					below_count -= get<1>(seg).x > prev_lane_mark_right.x ? abs(prev_lane_mark_right.x - get<1>(seg).x) : 0;

					get<2>(seg) = below_count / (get<1>(seg).x - get<0>(seg).x);
				}

				// New probable road seg found
				if (get<2>(seg) >= 0.5 && curr_chain > longest_chain) {
					longest_chain = curr_chain;
					most_prob_seg = seg;
				}
			}

			Point lane_mark_left = get<0>(most_prob_seg);
			Point lane_mark_right = get<1>(most_prob_seg);

			Point diff_left = lane_mark_left - prev_lane_mark_left;
			Point diff_right = lane_mark_right - prev_lane_mark_right;

			double dist_left = sqrt(pow(diff_left.x, 2) + pow(diff_left.y, 2));
			double dist_right = sqrt(pow(diff_right.x, 2) + pow(diff_right.y, 2));

			if (lane_mark_left != Point(0, y)) {
				lane_lines[0].push_back(lane_mark_left);

				avg_left_x_diff += abs(lane_mark_left.x - prev_lane_mark_left.x);
				left_count++;
			}

			if (lane_mark_right != Point(mask.cols + 1, y)) {
				lane_lines[1].push_back(lane_mark_right);

				avg_right_x_diff += abs(lane_mark_right.x - prev_lane_mark_right.x);
				right_count++;
			}

			prev_lane_mark_left = lane_mark_left;
			prev_lane_mark_right = lane_mark_right;
		}

		avg_left_x_diff = left_count == 0 ? -1 : avg_left_x_diff / (float)left_count;
		avg_right_x_diff = right_count == 0 ? -1 : avg_right_x_diff / (float)right_count;

		int noise_thresh = 50;
		int consecutive_del_count = 0;

		// Filter vertical outliers
		for (int i = 0; i < lane_lines.size(); i++) {
			if (lane_lines[i].size() > 3) {
				for (auto it = lane_lines[i].begin() + 1; it != lane_lines[i].end() - 1; it++) {
					int diff_next = sqrt(pow(it->x - (it + 1)->x, 2) + pow(it->y - (it + 1)->y, 2));
					int diff_prev = sqrt(pow(it->x - (it - 1)->x, 2) + pow(it->y - (it - 1)->y, 2));

					if (abs(diff_next - avg_left_x_diff) > noise_thresh * (consecutive_del_count + 1) || abs(diff_prev - avg_left_x_diff) > noise_thresh * (consecutive_del_count + 1)) {
						it = --lane_lines[i].erase(it);
						consecutive_del_count++;
					}
					else {
						consecutive_del_count = 0;
					}
				}
			}
		}

		lane_lines.shrink_to_fit();
		send(out_sock, &lane_lines, sizeof(lane_lines), 0);
	}

	return 0;
}
