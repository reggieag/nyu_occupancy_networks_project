#include <iostream>
#include <marchingCubes.h>

using namespace std;

int main(int argc, char *argv[]) {

  MarchingCubes::assembleCubes("/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/coords.txt",
                               "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/predictions.txt");
  
  return 0;
}
