#include <iostream>
#include <marchingCubes.h>
#include <math.h>
#include <functional>


using namespace std;


bool implicitSphere(float x, float y, float z, float r=1){
  return std::sqrt(x*x + y*y + z*z) <= r;;

}

int main(int argc, char *argv[]) {

  //std::vector<Cube> cubes = MarchingCubes::assembleCubes("/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/coords.txt",
  //                                                       "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/predictions.txt");

  // MarchingCubes::march(cubes, "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/MarchingCubes/mesh.off");

  std::vector<Cube> cubes = MarchingCubes::generateCubes();
  using namespace std::placeholders;
  MarchingCubes::computeOccupancies(cubes, std::bind(implicitSphere, _1,_2,_3,0.2));

  MarchingCubes::march(cubes, "testSphere.off");



  return 0;
}
