#ifndef MOTOR_H_
#define MOTOR_H_

#include <initializer_list>
#include <vector>

class Motor {
	std::vector<int> control_pins;
	std::vector<int> control_vals;

public:
	Motor(std::initializer_list<int> pins);
	void initialize(std::initializer_list<int> pins);
	void update();
	void speed(int v);
};

#endif
