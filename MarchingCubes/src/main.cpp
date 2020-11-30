#include <iostream>
#include <marchingCubes.h>

using namespace std;

int main(int argc, char *argv[]) {

  std::vector<Cube> cubes = MarchingCubes::assembleCubes("/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/coords.txt",
                                                         "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/predictions.txt");

  MarchingCubes::march(cubes, "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/MarchingCubes/mesh.off");
  
  return 0;
}
