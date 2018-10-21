#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <netinet/in.h>
#include <pigpio.h>
#include <sstream>
#include <cstring>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Motor.h"

using namespace std;

void intHandler(int x);

int main() {
	signal(SIGINT, intHandler);

	cout << "Pi: controls launching" << endl;

	gpioInitialise();

	map<string, Motor*> motors;
	list<string> commands;
	ifstream cmd_dump;

	motors["drv"] = new Motor{ 18 };  // Drive motor
	motors["str"] = new Motor{ 12 };  // Steer motor
	motors["brk"] = new Motor{ 17 };  // Break motor

	// Network parameters
	struct sockaddr_in dst;
	struct sockaddr_in serv;
	int sock;
	socklen_t sock_size = sizeof(struct sockaddr_in);

	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = htonl(INADDR_ANY);
	serv.sin_port = htons(8061);

	sock = socket(AF_INET, SOCK_STREAM, 0);

	bind(sock, (struct sockaddr*)&serv, sizeof(struct sockaddr));
	listen(sock, 1);
	cout << "Pi: Waiting for control system" << endl;
	int consock = accept(sock, (struct sockaddr*)&dst, &sock_size);

	cout << "Pi: ready" << endl;

	while (consock) {
		string cmd;

		for (auto& m : motors)
			m.second->update();

		char buf[50];
		recv(consock, buf, 49, 0);
		commands.push_back(buf);

		if (!commands.empty()) {
			//cout << "Processing Command" << endl;
			cmd = commands.front();
			commands.pop_front();

			//cout << "Command: " << cmd << endl;

			istringstream ss(cmd);
			string tok;

			int count = 0;
			string op;
			while (getline(ss, tok, ' ')) {
				if (count % 2 == 0) {
					op = tok;
					//cout << "Operation: " << op << endl;
				}
				else {
					//cout << "Value: " << atoi(tok.c_str()) << endl;

					if (motors.find(op) != motors.end())
						motors.at(op)->speed(atoi(tok.c_str()));
				}
				count++;
			}
		}
	}

	close(sock);

	for (auto& m : motors) {
		m.second->speed(50);
		m.second->update();
	}

	return 0;
}

void intHandler(int x) {
	cout << "Pi: Shutting down" << endl;
	gpioTerminate();

	exit(0);
}

