//g++ double_pendulum.cpp -lglut -lGLU -lGL -std=c++17

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include "Vector.h"

#include <GL/glut.h>

const double shift = 1.118926; //Shift between pixel_y and mouse_y coordinates 

const int size = 800; //Size of square shaped window
const double circle_r = 15; //Radius of pendulum balls
const double maxvel = 15; //Max. velocity when moving balls

const double l1 = 1.0; //Length of first rod
const double l2 = 1.0; //Length of second rod
const double m1 = 1.0; //Mass of first ball
const double m2 = 1.0; //Mass of second ball
const double M = m1+ m2; 
const double g = 0.0;
const double eta = 0.0; //Drag coefficient

const double h = 1e-4, t1 = 1000000; //Time from t to t1 with h step

struct Pendulum{
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;

    int i = 0; //Will be used to only show every 7th step
    int menuchoice = 1, prev_menuchoice = 1;

    double mouse_x, mouse_y;
    bool mouseleftdown;
    bool prev_mouseleftdown;
    bool fmove_entry;
    bool smove_entry;

    Vector<double> y{0, 0, 0, 0}; //(angle1, angle2, angle1 velocity, angle2 velocity), pl. y{M_PI/2, -M_PI/8, 4, -1}
    Vector<double> y_click{0, 0, 0, 0}; //To save state if left botton clicked
    double maxvel1 = 0.05, maxvel2 = 0.05; //Save max. velocities to show phase space in correct velocity scale
    double t = 0;

    double fcircle_center_x, fcircle_center_y;
    double scircle_center_x, scircle_center_y;
    double fcircle_eq, scircle_eq;

    //Differential equations to be solved
    Vector<double> func (double t, Vector<double> y){
        double norm = M+m1-m2*std::cos(2*y[0]-2*y[1]);
        double anglediff1 = std::sin(y[0]-y[1]);
        double anglediff2 = std::cos(y[0]-y[1]);
        double drag1 = 2*sq(l1)*y[2]+l1*l2*y[3]*anglediff2;
        double third = (-g*(M+m1)*std::sin(y[0])-m2*g*std::sin(y[0]-2*y[1])-2*anglediff1*m2*(sq(y[2])*l1*anglediff2+sq(y[3])*l2))/(l1*norm)-eta*drag1;
        double drag2 = sq(l2)*y[3]+l1*l2*y[2]*anglediff2;
        double fourth = (2*anglediff1*(M*sq(y[2])*l1+g*M*std::cos(y[0])+sq(y[3])*l2*m2*anglediff2))/(l2*norm)-eta*drag2;
        Vector<double> yn{y[2], y[3], third, fourth};
        return yn;
    };

    //RK4 to next step
    void run_RK4(){
        if(t + h > t1) glutDestroyWindow(1);
        //Running RK4
        Vector<double> k1 = func(t, y); 
        Vector<double> k2 = func(t + h * 0.5, y + (h * 0.5) * k1);
        Vector<double> k3 = func(t + h * 0.5, y + (h * 0.5) * k2);
        Vector<double> k4 = func(t + h, y + h * k3) ;

        y = y + (k1 + k4 + 2.0 * (k2 + k3)) * (h / 6);

        t = t + h;
        i++;
    }

    //Display pendulum
    void display_pendulum(){
        //First line
        glBegin(GL_LINES);
        glVertex2d(size/2, size/2);
        glVertex2d(fcircle_center_x, fcircle_center_y);
        glEnd();
        //First circle
        glPushMatrix();
        glTranslated(fcircle_center_x, fcircle_center_y, 0.0);
        glutWireSphere(circle_r, 50, 50);
        glPopMatrix();
        //Second circle
        glPushMatrix();
        glTranslated(scircle_center_x, scircle_center_y, 0.0);
        glutWireSphere(circle_r, 50, 50);
        glPopMatrix();
        //Second line
        glBegin(GL_LINES);
        glVertex2d(fcircle_center_x, fcircle_center_y);
        glVertex2d(scircle_center_x, scircle_center_y);
        glEnd();
        glFlush();

        glClear(GL_COLOR_BUFFER_BIT);
    }

    //Display phase space
    void display_phasespace(int menuchoice){
        //Display phase point in every 8th step
        if(i == 7){
            glPushMatrix();
            //Separation based on menuchoice (first or second ball's phase space)
            if(menuchoice == 2) glTranslated(size/2 + 190 * std::asin(sin(y[0])), size/2 - (300 / maxvel1) * y[2], 0.0);
            if(menuchoice == 3) glTranslated(size/2 + 190 * std::asin(sin(y[1])), size/2 - (300 / maxvel2) * y[3], 0.0);
            if(maxvel1 < abs(y[2])) maxvel1 = std::abs(y[2]);
            if(maxvel2 < abs(y[3])) maxvel2 = std::abs(y[3]);
            glutWireSphere(0.5, 50, 50);
            glPopMatrix();
            glFlush();
            i = 0;
        }
    }
  
    void idle(){
        //If menuchoice changed clear screen
        if(menuchoice != prev_menuchoice) glClear(GL_COLOR_BUFFER_BIT);

        //Pendulum choice
        if(menuchoice == 1){
            //If left mouse button clicked
            if(mouseleftdown && prev_mouseleftdown == false){ t0 = std::chrono::high_resolution_clock::now(); y_click = y; }

            //If left mouse button released
            if(mouseleftdown == false && prev_mouseleftdown){ 
                 auto t1 = std::chrono::high_resolution_clock::now();
                 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                 if(fmove_entry){
                     y[2]  = ((y[0] - y_click[0]) / elapsed) * 400;
                     if(maxvel < y[2]) y[2] = maxvel;
                     if(- maxvel > y[2]) y[2] = - maxvel;
                 }
                 if(smove_entry){
                     y[3]  = ((y[1] - y_click[1]) / elapsed) * 400;   
                     if(maxvel < y[3]) y[3] = maxvel; 
                     if(- maxvel > y[3]) y[3] = - maxvel;
                 }              
                 fmove_entry = false; 
                 smove_entry = false; 
            }

            //Don't entry first and second in same loop
            if(smove_entry && fmove_entry) fmove_entry = false;  
       
            //Calculate variables needed to move balls
            double l1sin = 180 * l1 * std::sin(y[0]);
            double l1cos = 180 * l1 * std::cos(y[0]);
            double l2sin = 180 * l2 * std::sin(y[1]);
            double l2cos = 180 * l2 * std::cos(y[1]);
            fcircle_center_x = size/2 + l1sin;
            fcircle_center_y = size/2 + l1cos; 

            scircle_center_x = size/2 + l1sin + l2sin;
            scircle_center_y = size/2 + l1cos + l2cos; 

            fcircle_eq = sq(mouse_x - fcircle_center_x) + sq(mouse_y * shift - fcircle_center_y);
            scircle_eq = sq(mouse_x - scircle_center_x) + sq(mouse_y * shift - scircle_center_y);

            //Pendulum choice -> move first ball
            if((mouseleftdown && fcircle_eq < sq(5 * circle_r)) || (mouseleftdown && prev_mouseleftdown && fmove_entry)){
                fmove_entry = true;
                //Separation based on plane quarters
                y[0] = std::atan2(mouse_x - size/2, mouse_y * shift - size/2);
                y[2] = y[3] = 0;
                display_pendulum();
            } 

            //Don't entry first and second ball move in same loop
            if(fmove_entry) smove_entry = false;

            //Pendulum choice -> move second ball
            if((mouseleftdown && scircle_eq < sq(5 * circle_r)) || (mouseleftdown && prev_mouseleftdown && smove_entry)){
                smove_entry = true;
                //Separation based on plane quarters  
                y[1] = std::atan2(mouse_x - (size/2 + l1sin), mouse_y * shift - (size/2 + l1cos)); 
                y[2] = y[3] = 0;
                display_pendulum();
            } 
            //Pendulum choice -> balls move free
            else{
                run_RK4();
                //Display pendulum in every 8th step
                if(i == 7){
                    display_pendulum();
                    i = 0;
                }
            }
            glClear(GL_COLOR_BUFFER_BIT);
        }

        //First or second ball phase space choice
        if(menuchoice !=1){ run_RK4(); display_phasespace(menuchoice); }

        //Save previous state of left mouse button
        prev_mouseleftdown = mouseleftdown;
        //Save previous menuchoice
        prev_menuchoice =  menuchoice;
    }
};

Pendulum p;

//Mouse related functions
void mouse(int button, int state, int x, int y){
    glMatrixMode(GL_MODELVIEW);
    //Save left button state
    if(button == GLUT_LEFT_BUTTON)
    {
       p.mouseleftdown = (state == GLUT_DOWN);
    }
    //Save mouse position
    p.mouse_x = x;
    p.mouse_y = y;
}

void motion(int x, int y){
    //Save mouse position
    p.mouse_x = x;
    p.mouse_y = y;
}

void menu(int choice){
    p.menuchoice = choice;
        if(choice == 1) glutSetWindowTitle("Double Pendulum");
        if(choice == 2) glutSetWindowTitle("Double Pendulum - Firs ball phase space");
        if(choice == 3) glutSetWindowTitle("Double Pendulum - Second ball phase space");    
}

void glidle(){
    p.idle();
}


int main(int argc, char **argv)
{   
    //Initialize window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(size, size);
    glutCreateWindow("Double Pendulum");

    glViewport(0, 0, size, size);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(0, size, size, 0, -1, 1);

    glutIdleFunc(glidle);

    //Mouse related functions
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    //Menu related functions
    glutCreateMenu(menu);
    glutAddMenuEntry("Double Pendulum", 1);
    glutAddMenuEntry("First ball's phase space", 2);
    glutAddMenuEntry("Second ball's phase space", 3);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glutMainLoop();

    return 0;
}
