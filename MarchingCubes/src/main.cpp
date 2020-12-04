#include <iostream>
#include <marchingCubes.h>
#include <math.h>
#include <functional>


using namespace std;


bool implicitSphere(float x, float y, float z, float r=0.3){
  return std::sqrt(x*x + y*y + z*z) <= r;

}

int main(int argc, char *argv[]) {

  //std::vector<Cube> cubes = MarchingCubes::assembleCubes("/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/bench_ag_32_3.txt",
  //                                                       "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/bench_ag_preds_32_3.txt");



  std::vector<Cube> cubes = MarchingCubes::generateCubes();
  using namespace std::placeholders;
  MarchingCubes::computeOccupancies(cubes, std::bind(implicitSphere, _1,_2,_3,1.0));

  MarchingCubes::march(cubes, "sphere_midpoint.off");



  return 0;
}
