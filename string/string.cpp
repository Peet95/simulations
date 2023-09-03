#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

int main() 
{
  std::ofstream myfile;
  myfile.open ("data.txt");

  // Parameters
  double t = 0.0;
  double rho   = 0.01;                                              
  double ten   = 40.0;                                             
  double c     = sqrt(ten/rho);                                  
  double c1    = c; // CFL criterium
  double ratio =  c*c/(c1*c1);

  // Initialization
  std::vector<std::vector<double>>  xi(101, std::vector<double>(3, 0));
  std::vector<double> xa(100), ya(100);

  for(int i = 0; i < 81; i++) {
    xi[i][0] = 0.00125*i; 
  }        
  for(int i = 81; i < 101; i++) {
    xi[i][0] = 0.1 - 0.005*(i - 80);
  }        
  for(int i = 0; i < 100; i++) {
    xa[i] = 2.0*i - 100.0;                            
    ya[i] = 300.0*xi[i][0];  
    myfile << t << " " << xa[i] << " " << ya[i] << std::endl;  
  }

  // Later time steps
  for(int i = 1; i < 99; i++) {
    xi[i][1] = xi[i][0] + 0.5*ratio*(xi[i+1][0] + xi[i-1][0] - 2.0*xi[i][0]);
  }
  while (t < 80.0){
    for(int i = 1; i < 100; i++) {
      xi[i][2] = 2.0*xi[i][1] - xi[i][0] + ratio*(xi[i+1][1] + xi[i-1][1] - 2.0*xi[i][1]);

    }
      
    for(int i = 1; i < 100; i++) {
      xa[i] = 2.0*i - 100.0;                  
      ya[i] = 300.0*xi[i][2];
      myfile << t << " " << xa[i] << " " << ya[i] << std::endl; 
    }
    
   
    for(int i = 0; i < 101; i++) {
      xi[i][0] = xi[i][1];                              
      xi[i][1] = xi[i][2];
    }
    
    t=t+1.0;
  }                                                               
  myfile.close();
  return 0;
}
