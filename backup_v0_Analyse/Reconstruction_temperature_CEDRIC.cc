#include "Reconstruction_temperature_CEDRIC.hh"

void Reconstruction_temperature_CEDRIC()
{

  float Calib_PSL = 0.695;
  float solid_angle1 = 0.0218;
  float solid_angle2 = 0.00815;

  //SHOT GSI
  Routine_Analyse("Data/Tir17_Results_QL.txt", "Tir17", Calib_PSL, solid_angle1, 21.4675);

  //Lecture_analyse_file("Data/Tir17_Results_QL.txt");
 
  

}
