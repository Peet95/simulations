#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// Constants
const int kGridSize = 101;
const int kOutputSize = 100;
const double kTension = 40.0;
const double kDensity = 0.01;
const double kEndTime = 80.0;
const double kTimeStep = 1.0;
const double kInitialDisturbance = 0.00125;
const double kDisturbanceDecline = 0.005;
const double kScaleFactor = 300.0;

double calculate_wave_speed(double tension, double density) {
    return std::sqrt(tension / density);
}

void initialize_grid(std::vector<std::vector<double>>& xi) {
    for (int i = 0; i < 81; ++i) {
        xi[i][0] = kInitialDisturbance * i;
    }
    for (int i = 81; i < kGridSize; ++i) {
        xi[i][0] = 0.1 - kDisturbanceDecline * (i - 80);
    }
}

void update_grid(std::vector<std::vector<double>>& xi, double ratio) {
    for (int i = 1; i < kOutputSize - 1; ++i) {
        xi[i][1] = xi[i][0] + 0.5 * ratio * (xi[i + 1][0] + xi[i - 1][0] - 2.0 * xi[i][0]);
    }
    for (int i = 1; i < kOutputSize; ++i) {
        xi[i][2] = 2.0 * xi[i][1] - xi[i][0] + ratio * (xi[i + 1][1] + xi[i - 1][1] - 2.0 * xi[i][1]);
    }
}

void write_data(std::ofstream& myfile, const std::vector<std::vector<double>>& xi, double t) {
    for (int i = 0; i < kOutputSize; ++i) {
        double xa = 2.0 * i - kOutputSize;
        double ya = kScaleFactor * xi[i][2];
        myfile << t << " " << xa << " " << ya << std::endl;
    }
}

int main() {
    std::ofstream myfile("data.txt");

    double t = 0.0;
    double wave_speed = calculate_wave_speed(kTension, kDensity);
    double ratio = wave_speed * wave_speed / (wave_speed * wave_speed);

    // Initialize grid
    std::vector<std::vector<double>> xi(kGridSize, std::vector<double>(3, 0.0));
    initialize_grid(xi);

    // Write initial state
    write_data(myfile, xi, t);

    // Time loop
    while (t < kEndTime) {
        update_grid(xi, ratio);
        write_data(myfile, xi, t);

        // Update time
        t += kTimeStep;

        // Advance state
        for (int i = 0; i < kGridSize; ++i) {
            xi[i][0] = xi[i][1];
            xi[i][1] = xi[i][2];
        }
    }

    myfile.close();
    return 0;
}
