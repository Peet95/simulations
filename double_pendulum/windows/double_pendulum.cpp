#include <iostream>
#include <chrono>
#include <cmath>
#include <GL/glut.h>
#include "Vector.h"

constexpr double kShift = 1.118926;
constexpr int kWindowSize = 800;
constexpr double kCircleRadius = 15;
constexpr double kMaxVelocity = 15;

constexpr double kRodLength1 = 1.0;
constexpr double kRodLength2 = 1.0;
constexpr double kMass1 = 1.0;
constexpr double kMass2 = 1.0;
constexpr double kTotalMass = kMass1 + kMass2;
constexpr double kGravity = 0.5;
constexpr double kDragCoefficient = 0.01;

constexpr double kTimeStep = 3.0e-3;
constexpr double kMaxTime = 1000000;

struct Pendulum {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    int step_counter = 0;
    int menu_choice = 1;
    int prev_menu_choice = 1;
    bool reset = false;

    double mouse_x = 0, mouse_y = 0;
    bool mouse_left_down = false;
    bool prev_mouse_left_down = false;
    bool first_move_entry = false;
    bool second_move_entry = false;

    Vector<double> state{0, 0, 0, 0}; // (angle1, angle2, angular velocity1, angular velocity2)
    Vector<double> state_on_click{0, 0, 0, 0};
    double max_vel1 = 0.05, max_vel2 = 0.05;
    double current_time = 0;

    double first_circle_center_x = 0, first_circle_center_y = 0;
    double second_circle_center_x = 0, second_circle_center_y = 0;
    double first_circle_eq = 0, second_circle_eq = 0;

    Vector<double> calculate_derivatives(double t, Vector<double> y) {
        double norm_factor = kTotalMass + kMass1 - kMass2 * std::cos(2 * y[0] - 2 * y[1]);
        double angle_diff1 = std::sin(y[0] - y[1]);
        double angle_diff2 = std::cos(y[0] - y[1]);
        double drag1 = 2 * sq(kRodLength1) * y[2] + kRodLength1 * kRodLength2 * y[3] * angle_diff2;
        double third = (-kGravity * (kTotalMass + kMass1) * std::sin(y[0]) - kMass2 * kGravity * std::sin(y[0] - 2 * y[1]) - 2 * angle_diff1 * kMass2 * (sq(y[2]) * kRodLength1 * angle_diff2 + sq(y[3]) * kRodLength2)) / (kRodLength1 * norm_factor) - kDragCoefficient * drag1;
        double drag2 = sq(kRodLength2) * y[3] + kRodLength1 * kRodLength2 * y[2] * angle_diff2;
        double fourth = (2 * angle_diff1 * (kTotalMass * sq(y[2]) * kRodLength1 + kGravity * kTotalMass * std::cos(y[0]) + sq(y[3]) * kRodLength2 * kMass2 * angle_diff2)) / (kRodLength2 * norm_factor) - kDragCoefficient * drag2;
        return Vector<double>{y[2], y[3], third, fourth};
    }

    void run_rk4() {
        if (current_time + kTimeStep > kMaxTime) {
            glutDestroyWindow(1);
            return;
        }

        Vector<double> k1 = calculate_derivatives(current_time, state); 
        Vector<double> k2 = calculate_derivatives(current_time + kTimeStep * 0.5, state + (kTimeStep * 0.5) * k1);
        Vector<double> k3 = calculate_derivatives(current_time + kTimeStep * 0.5, state + (kTimeStep * 0.5) * k2);
        Vector<double> k4 = calculate_derivatives(current_time + kTimeStep, state + kTimeStep * k3);

        state += (k1 + k4 + 2.0 * (k2 + k3)) * (kTimeStep / 6);
        current_time += kTimeStep;
        step_counter++;
    }

    void display_pendulum() {
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw first rod
        glBegin(GL_LINES);
        glVertex2d(kWindowSize / 2, kWindowSize / 2);
        glVertex2d(first_circle_center_x, first_circle_center_y);
        glEnd();

        // Draw first ball
        glPushMatrix();
        glTranslated(first_circle_center_x, first_circle_center_y, 0.0);
        glutWireSphere(kCircleRadius, 50, 50);
        glPopMatrix();

        // Draw second rod
        glBegin(GL_LINES);
        glVertex2d(first_circle_center_x, first_circle_center_y);
        glVertex2d(second_circle_center_x, second_circle_center_y);
        glEnd();

        // Draw second ball
        glPushMatrix();
        glTranslated(second_circle_center_x, second_circle_center_y, 0.0);
        glutWireSphere(kCircleRadius, 50, 50);
        glPopMatrix();

        glFlush();
    }

    void display_phase_space(int choice) {
        if (step_counter == 7) {
            glPushMatrix();
            if (choice == 2) {
                glTranslated(kWindowSize / 2 + 190 * std::asin(sin(state[0])), kWindowSize / 2 - (300 / max_vel1) * state[2], 0.0);
            } else if (choice == 3) {
                glTranslated(kWindowSize / 2 + 190 * std::asin(sin(state[1])), kWindowSize / 2 - (300 / max_vel2) * state[3], 0.0);
            }
            max_vel1 = std::max(max_vel1, std::abs(state[2]));
            max_vel2 = std::max(max_vel2, std::abs(state[3]));
            glutWireSphere(0.5, 50, 50);
            glPopMatrix();
            glFlush();
            step_counter = 0;
        }
    }

    void idle() {
        if (reset) {
            state = {0, 0, 0, 0};
            reset = false;
        }

        if (menu_choice != prev_menu_choice) {
            glClear(GL_COLOR_BUFFER_BIT);
        }

        if (menu_choice == 1 || menu_choice == 4) {
            handle_mouse_interactions();
            calculate_pendulum_positions();

            if (first_move_entry || second_move_entry) {
                display_pendulum();
            } else {
                run_rk4();
                display_pendulum();
            }

            glClear(GL_COLOR_BUFFER_BIT);
        }

        if (menu_choice == 2 || menu_choice == 3) {
            run_rk4();
            display_phase_space(menu_choice);
        }

        prev_mouse_left_down = mouse_left_down;
        prev_menu_choice = menu_choice;
    }

    void handle_mouse_interactions() {
        if (mouse_left_down && !prev_mouse_left_down) {
            start_time = std::chrono::high_resolution_clock::now();
            state_on_click = state;
        }

        if (!mouse_left_down && prev_mouse_left_down) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            if (first_move_entry) {
                state[2] = ((state[0] - state_on_click[0]) / elapsed) * 400;
                state[2] = std::clamp(state[2], -kMaxVelocity, kMaxVelocity);
            }

            if (second_move_entry) {
                state[3] = ((state[1] - state_on_click[1]) / elapsed) * 400;
                state[3] = std::clamp(state[3], -kMaxVelocity, kMaxVelocity);
            }

            first_move_entry = second_move_entry = false;
        }

        if (second_move_entry && first_move_entry) {
            first_move_entry = false;
        }
    }

    void calculate_pendulum_positions() {
        double l1sin = 180 * kRodLength1 * std::sin(state[0]);
        double l1cos = 180 * kRodLength1 * std::cos(state[0]);
        double l2sin = 180 * kRodLength2 * std::sin(state[1]);
        double l2cos = 180 * kRodLength2 * std::cos(state[1]);

        first_circle_center_x = kWindowSize / 2 + l1sin;
        first_circle_center_y = kWindowSize / 2 + l1cos;

        second_circle_center_x = kWindowSize / 2 + l1sin + l2sin;
        second_circle_center_y = kWindowSize / 2 + l1cos + l2cos;

        first_circle_eq = sq(mouse_x - first_circle_center_x) + sq(mouse_y * kShift - first_circle_center_y);
        second_circle_eq = sq(mouse_x - second_circle_center_x) + sq(mouse_y * kShift - second_circle_center_y);

        // Handle first ball movement
        if ((mouse_left_down && first_circle_eq < sq(5 * kCircleRadius)) || (mouse_left_down && prev_mouse_left_down && first_move_entry)) {
            first_move_entry = true;
            state[0] = std::atan2(mouse_x - kWindowSize / 2, mouse_y * kShift - kWindowSize / 2);
            state[2] = state[3] = 0;
            display_pendulum();
        }

        if (first_move_entry) {
            second_move_entry = false; // Prevent simultaneous entry
        }

        // Handle second ball movement
        if ((mouse_left_down && second_circle_eq < sq(5 * kCircleRadius)) || (mouse_left_down && prev_mouse_left_down && second_move_entry)) {
            second_move_entry = true;
            state[1] = std::atan2(mouse_x - (kWindowSize / 2 + l1sin), mouse_y * kShift - (kWindowSize / 2 + l1cos));
            state[2] = state[3] = 0;
            display_pendulum();
        }
    }
};

Pendulum pendulum;

// Mouse interaction functions
void mouse(int button, int state, int x, int y) {
    pendulum.mouse_left_down = (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
    pendulum.mouse_x = x;
    pendulum.mouse_y = y;
}

void motion(int x, int y) {
    pendulum.mouse_x = x;
    pendulum.mouse_y = y;
}

// Menu selection function
void menu(int choice) {
    pendulum.menu_choice = choice;
    switch (choice) {
        case 1:
            glutSetWindowTitle("Double Pendulum");
            break;
        case 2:
            glutSetWindowTitle("Double Pendulum - First ball phase space");
            break;
        case 3:
            glutSetWindowTitle("Double Pendulum - Second ball phase space");
            break;
        case 4:
            pendulum.reset = true;
            glutSetWindowTitle("Double Pendulum");
            break;
    }
}

void idle() {
    pendulum.idle();
}

int main(int argc, char **argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(kWindowSize, kWindowSize);
    glutCreateWindow("Double Pendulum");

    glViewport(0, 0, kWindowSize, kWindowSize);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, kWindowSize, kWindowSize, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Register callback functions
    glutDisplayFunc(idle);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // Create menu
    glutCreateMenu(menu);
    glutAddMenuEntry("Double Pendulum", 1);
    glutAddMenuEntry("First ball's phase space", 2);
    glutAddMenuEntry("Second ball's phase space", 3);
    glutAddMenuEntry("Reset", 4);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Enter the main loop
    glutMainLoop();

    return 0;
}        
