#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include "Lanefinder.h"

#define TILE_SIZE 10
#define LANE_PROBABILITY_SEARCH_RANGE 6
#define LANE_PROBABILITY_THRESHOLD 0.5

int Lanefinder::canny_thresh;
//cv::VideoWriter Lanefinder::w2;

/*
 * Description:	Set Lane values
 * Parameters:	void
 * Returns:	void
**/
void Lanefinder::initialize() {
	canny_thresh = 50;
	//w2.open("rec HistMapi.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, cv::Size(853, 480));
}

/*
 * Description:	Score a colour to be compared in road finding
 * Parameters:	Vec3b: col - Colour to be scored
 * Returns:	Float: Score of colour
**/
float Lanefinder::scoreColour(cv::Vec3b col) {
	cv::Mat3b score(col), score_lab;
	cvtColor(score, score_lab, CV_BGR2Lab);

	return sqrt(pow(col[1], 2) + pow(col[2], 2));

	return (pow(col[0], 2) + pow(col[1], 3) + col[2]) / (col[2] == 0 ? 1 : col[2] / 2.0);
}

/*
 * Description:	Determine if colour pixel is part of road or not
 * PParameters:	Vec3b: col - Pixel to analyze
 * Returns:	Boolean: True if pixel is part of the road
**/
bool Lanefinder::isRoad(cv::Vec3b col) {
	return (col[0] && col[1]) || (col[0] && !col[1] && !col[2]) || (!col[0] && col[1]);
}

/*
 * Description:	Find lanes in frame
 * Parameters:	Ref to Matrix:                         frame      - Frame to search for lane in
 *		Ref to List of List of Pair of Points: lane_lines - Destination of line segments which represent the lanes
 * Returns:	void
**/
void Lanefinder::find(cv::Mat& frame, std::vector<std::vector<cv::Point> >& lane_lines) {
	using namespace std;
	using namespace cv;

	Rect crop(0, 0, frame.cols, frame.rows);  // Full frame
	//Rect crop(0, frame.rows / 2 + 40, frame.cols, frame.rows - frame.rows / 2 - 40);  // Bot Half frame
	//Rect crop(0, frame.rows / 2, 850, frame.rows - frame.rows / 2);  // TMP
	//Rect crop(0, 50, frame.cols - 2, frame.rows - 50);
	//cout << frame.cols << " " << frame.rows << endl;

	Mat frame_crop, frame_gray, frame_hsv, frame_hls, frame_lab, detected_edges;
	//Mat road_mask(frame.rows / TILE_SIZE + 1, frame.cols / TILE_SIZE + 1, CV_8UC1, Scalar(0));
	Mat road_mask(frame.rows, frame.cols, CV_8UC1, Scalar(0));
	Mat element = getStructuringElement(0, Size(5, 5), Point(2, 2));

	// Image warp parameters
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

	// Crop
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
	/*
	for (int y = 0; y < tmp.rows; y += TILE_SIZE) {
		for (int x = 0; x < tmp.cols; x += 2 * TILE_SIZE) {
			putText(frame, to_string(tmp.at<Vec3b>(y, x)[1]), Point(x, y), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
		}
	}
	*/
	normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);
	//imshow("TMP1", tmp_show);

	threshold(tmp, tmp, 150 * 0.85, 1, THRESH_BINARY);  // 150 - day
	add(score, tmp, score);

	// Extract l channel
	split(frame_hls, chan);
	clahe->apply(chan[1], tmp);
	tmp.copyTo(chan[1]);
	merge(chan, tmp);
	/*
	for (int y = 0; y < tmp.rows; y += TILE_SIZE) {
		for (int x = 0; x < tmp.cols; x += 2 * TILE_SIZE) {
			putText(frame, to_string(tmp.at<Vec3b>(y, x)[1]), Point(x, y + 6), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
		}
	}
	*/
	normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);
	//imshow("TMP2", tmp_show);

	threshold(tmp, tmp, 210 * 0.85, 1, THRESH_BINARY);  // 210 - day
	add(score, tmp, score);

	// Extract v channel
	clahe->setClipLimit(6);
	split(frame_hsv, chan);
	clahe->apply(chan[2], tmp);
	tmp.copyTo(chan[2]);
	merge(chan, tmp);
	/*
	for (int y = 0; y < tmp.rows; y += TILE_SIZE) {
		for (int x = 0; x < tmp.cols; x += 2 * TILE_SIZE) {
			putText(frame, to_string(tmp.at<Vec3b>(y, x)[1]), Point(x, y + 12), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
		}
	}
	*/
	normalize(tmp, tmp_show, 0, 255, NORM_MINMAX);
	//imshow("TMP3", tmp_show);

	threshold(tmp, tmp, 220 * 0.85, 1, THRESH_BINARY);  // 220 - old
	add(score, tmp, score);

	lane_lines.push_back(vector<Point>());  // Left set
	lane_lines.push_back(vector<Point>());  // Right set

	// Find left and right lane edges
	int void_counter = 0;  // Spaces of blanks, ignore if too many

	Mat score_tmp = score.clone();
	normalize(score_tmp, score_tmp, 0, 255, NORM_MINMAX);
	//score_tmp *= 255;

	/*
	for (int i = 0; i < 400; i += 50) {
		cout << score.at<Vec3b>(Point(i, 400)) << endl;
	}
	*/

	Point prev_lane_mark_left, prev_lane_mark_right;
	float avg_left_x_diff = 0, avg_right_x_diff = 0;
	int left_count = 0, right_count = 0;

	for (int y = score.rows - TILE_SIZE; y > 0; y -= TILE_SIZE) {
		bool l_found = false, r_found = false;  // Find one point per line
		float max_col_score_left = 0.0, max_col_score_right = 0.0;
		bool curr_is_road = false, prev_is_road = false;

		vector<tuple<Point, Point, double> > pot_road_seg;  // Potential road segments (Start point, End point, Probability of being road)
		tuple<Point, Point, double> most_prob_seg;  // Most probable road segment

		for (int x = 0; x < score.cols; x += 2 * TILE_SIZE) {
			// Update road_mask
			Point curr(x, y);

			float l_avg = 0, r_avg = 0;
			int l_count = 0, r_count = 0;

			// Probability of current point being part of the road
			for (int c = x - LANE_PROBABILITY_SEARCH_RANGE * TILE_SIZE * 2; c <= x + LANE_PROBABILITY_SEARCH_RANGE * TILE_SIZE * 2; c += TILE_SIZE * 2) {
				if (c < 0 || c >= score.cols) {
					continue;
				}

				if (c < x) {  // Left of point
					l_avg += isRoad(score.at<Vec3b>(Point(c, y))) ? 1 : 0;
					l_count++;
				}
				else if (c > x) {  // Right of point
					r_avg += isRoad(score.at<Vec3b>(Point(c, y))) ? 1 : 0;
					r_count++;
				}
			}

			l_avg = l_count == 0 ? 0 : l_avg / (float)l_count;
			r_avg = r_count == 0 ? 0 : r_avg / (float)r_count;

			float curr_prob_road = l_avg + r_avg;  // Probability that the current point is actually road

			if (l_count != 0 && r_count != 0)
				curr_prob_road /= 2;

			road_mask.at<uchar>(Point(x, y)) = curr_prob_road > LANE_PROBABILITY_THRESHOLD ? 255 : 0;

			//putText(score_tmp, to_string(score.at<Vec3b>(curr)[0]), Point(x, y), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
			//putText(score_tmp, to_string(score.at<Vec3b>(curr)[1]), Point(x + 3, y), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
			//putText(score_tmp, to_string(score.at<Vec3b>(curr)[2]), Point(x + 6, y), FONT_HERSHEY_PLAIN, 0.5, Scalar(255, 255, 255), 1);
			putText(score_tmp, "8", Point(x, y + 6), FONT_HERSHEY_PLAIN, 0.5, isRoad(score.at<Vec3b>(curr)) ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 1);

			// Analyze road_mask
			curr_is_road = road_mask.at<uchar>(Point(x, y)) == 255;

			if (curr_is_road && !prev_is_road) {  // Start new segment
				//l_found = true;
				pot_road_seg.push_back(make_tuple(curr, Point(frame.cols + 1, y), 0));
			}
			else if (!curr_is_road && prev_is_road) {  // End segment
				//r_found = true;
				get<1>(*(pot_road_seg.end() - 1)) = curr;
			}

			prev_is_road = curr_is_road;

		}

		int longest_chain = 0;

		//cout << pot_road_seg.size() << endl;

		for (auto seg : pot_road_seg) {
			circle(road_mask, get<0>(seg), 5, Scalar(255, 255, 255), 6);
			circle(road_mask, get<1>(seg), 5, Scalar(255, 255, 255), 6);

			int curr_chain = (get<1>(seg).x - get<0>(seg).x) / TILE_SIZE;

			// Calculate percentage of current segment is above previously accepted one
			if (y >= score.rows - TILE_SIZE) {  // First row of road segments
				get<2>(seg) = 1.0;
			}
			else {
				double below_count = get<1>(seg).x - get<0>(seg).x;
				below_count -= get<0>(seg).x < prev_lane_mark_left.x ? abs(prev_lane_mark_left.x - get<0>(seg).x) : 0;
				below_count -= get<1>(seg).x > prev_lane_mark_right.x ? abs(prev_lane_mark_right.x - get<1>(seg).x) : 0;

				//cout << get<0>(seg) << " " << prev_lane_mark_left << "  " << get<1>(seg) << " " << prev_lane_mark_right << endl;
				//cout << (get<0>(seg).x < prev_lane_mark_left.x ? abs(prev_lane_mark_left.x - get<0>(seg).x) : 0) << " " << (get<1>(seg).x > prev_lane_mark_right.x ? abs(prev_lane_mark_right.x - get<1>(seg).x) : 0) << " ";

				get<2>(seg) = below_count / (get<1>(seg).x - get<0>(seg).x);
				//cout << get<2>(seg) << endl;
			}

			//cout << get<2>(seg) << endl;
			if (get<2>(seg) >= 0.5 && curr_chain > longest_chain) { // && curr_chain > (int)(10 * y) / frame.rows) {  // New probable road seg found
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

		//cout << prev_lane_mark_left.x << " " << prev_lane_mark_left.y << endl;

		putText(frame, to_string(dist_left), Point(prev_lane_mark_left.x + diff_left.x / 1.0, prev_lane_mark_left.y + diff_left.y / 1.0), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
		putText(frame, to_string(dist_right), Point(prev_lane_mark_right.x + diff_right.x / 1.0, prev_lane_mark_right.y + diff_right.y / 1.0), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1);

		//cout << lane_mark_left << " " << lane_mark_right << endl;

		if (lane_mark_left != Point(0, y)) {
			lane_lines[0].push_back(lane_mark_left);
			//prev_lane_mark_left = lane_mark_left;

			avg_left_x_diff += abs(lane_mark_left.x - prev_lane_mark_left.x);
			left_count++;
		}

		if (lane_mark_right != Point(frame.cols + 1, y)) {
			lane_lines[1].push_back(lane_mark_right);
			//prev_lane_mark_right = lane_mark_right;

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
					circle(frame, *it, 5, (i == 0 ? Scalar(255, 0, 0) : Scalar(0, 0, 255)), 6);
					consecutive_del_count = 0;
				}
			}
		}
	}

	//normalize(score, score, 0, 255, NORM_MINMAX);

	/*
	for (int i = 0; i < 400; i += 50) {
		circle(score_tmp, Point(i, 400), 5, Scalar(255, 255, 255), 6);
	}
	*/

	imshow("Score", score_tmp);
	//imshow("Road mask", road_mask);

	//w2.write(score_tmp);
}
