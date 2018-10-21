#include <softPwm.h>
#include <cmath>
#include <iostream>
#include <pigpio.h>
#include "Motor.h"

/*
 * Description:	Motor constructor
 * Parameters:	List of Integers: pins - pins used to control motor
 * Returns:	none
**/
Motor::Motor(std::initializer_list<int> pins) {
	initialize(pins);
}

/*
 * Description:	Create motor controls
 * Parameters:	List of Integers: pins - pins used to control motor
 * Returns:	void
**/
void Motor::initialize(std::initializer_list<int> pins) {
	for (int pin : pins) {
		gpioSetMode(pin, PI_OUTPUT);
		gpioSetPWMfrequency(pin, 333);
		gpioPWM(pin, 128);

		control_vals.push_back(128);
		control_pins.push_back(pin);
	}
}

/*
 * Description:	Set motor control values
 * Parameters:	void
 * Returns:	void
**/
void Motor::update() {
	for (size_t i = 0; i < control_pins.size(); i++) {
		gpioPWM(control_pins[i], control_vals[i]);
	}
}

/*
 * Description:	Set motor speed
 * Parameters:	Integer: v - Value of new speed [0, 100]
 * Returns:	void
**/
void Motor::speed(int v) {
	if (control_vals.size() < 1) return;

	// Map [0, 100] to [85, 170]
	double val = (double)v;

	val = std::max(0.0, std::min(100.0, val));
	val = 85.0 * (val / 100.0);
	val += 85.0;

	control_vals[0] = (int)val;
}

