//sudo apt-get install libboost-all-dev
//g++ heat_flow.cpp -lboost_iostreams -lboost_system -lboost_filesystem -Wall -O3

#include <iostream>
#include <vector>
#include <functional>
#include <boost/tuple/tuple.hpp>

#include "gnuplot-iostream.h"

using namespace std;

#define PI M_PI

int main(){
  int nslice = 16; //nslice: number of time slices
  double dt = 0.00001, total_t = 1.5; //dt: length of time step, total_t: full time of simulation
  double alpha2 = 1;

  //Size of the rectangle, X_size must be bigger than Y_size
  int X_size = 17;
  int Y_size = 12;

  //Configure gnuplot related variables
  std::vector<boost::tuple<double, double, double> > vecs;
  Gnuplot gp;
  //Configure gnuplot plot properties
  gp << "set view map\n";
  gp << "set dgrid3d 20,20,2\n";
  gp << "set pm3d at b interpolate 2,2\n";
  gp << "unset key\n";
  gp << "unset surface\n";
  gp << "set cbrange [0:120]\n"; //[-30:15]
  gp << "set xrange [1:" << X_size-2 << "]\n";
  gp << "set yrange [1:" << Y_size-2 << "]\n";

  double ds = 1./(X_size-1), r = alpha2 * dt/ds/ds;
  vector<double> mat_vec(X_size*X_size, 0.0);
  for(int i = 1; i < X_size-1; i++){
    for(int j = 1; j < Y_size-1; j++){
      mat_vec[X_size*i+j] = i + j;
    }
  }
  //Creat ghost grid on all sides for boundaries
  for(int i = 0; i < X_size; i++){
    mat_vec[X_size*i+0] = 0.0;
    mat_vec[X_size*i+X_size-1] = 0.0;
  }
  for(int j = 0; j < Y_size; j++){
    mat_vec[X_size*0+j] = 0.0;
    mat_vec[X_size*(X_size-1)+j] = 0.0;
  }

  double timeslice = total_t/nslice;
  int nsteps = (int) ceil((timeslice/dt));
  dt = timeslice / nsteps;

  int counter = 0;
  for(int i = 0; i <= nslice; i++){
    for(int l = 0; l < X_size; l++){
      mat_vec[X_size*l+1] = l;
    }
    //Display
    for(int j = 1; j < X_size - 1; j++){
      for(int k = 1; k < Y_size - 1; k++){
        vecs.push_back(boost::make_tuple( j, k, mat_vec[X_size*j+k]));
        gp << "splot" << gp.file1d(vecs) << "\n";
      }
    }
    gp << "splot '-'\n"; 
    gp.send1d(vecs);
    counter++;
    
    vector<double> laplacian_vec(X_size*X_size, 0.0);
    for(int t = 0; t < nsteps; t++){
      for(int i = 1; i < X_size -1; i++){
        for(int j = 1; j < Y_size -1; j++){
          //Insulator property 
          if(j == Y_size-2){ laplacian_vec[i*X_size+j] = (4 * laplacian_vec[i*X_size+Y_size-3] - laplacian_vec[i*X_size+Y_size-4])/3; }
          else if(i == 1){ laplacian_vec[i*X_size+j] = (4 * laplacian_vec[(i+1)*X_size+j] - laplacian_vec[(i+2)*X_size+j])/3; }
          //Heat comes in
          else if(i == X_size-2){ laplacian_vec[i*X_size+j] = (4 * laplacian_vec[(X_size-3)*X_size+j] - laplacian_vec[(X_size-4)*X_size+j])/3 + 0.0001 * 2 * 1; } //- 0.001 * 2 * 1
          //General way
          else{ laplacian_vec[i*X_size+j] = r * (mat_vec[(i-1)*X_size+j] + mat_vec[(i+1)*X_size+j] + mat_vec[i*X_size+j-1] + mat_vec[i*X_size+j+1] - 4 * mat_vec[i*X_size+j]); }
        }
      }
      //Add matrices
      transform(mat_vec.begin(), mat_vec.end(), laplacian_vec.begin(), mat_vec.begin(), std::plus<double>());
    }
  }
  gp << "exit\n";
  return 0;
}






