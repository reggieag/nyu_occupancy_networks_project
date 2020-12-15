#include <iostream>
#include <marchingCubes.h>
#include <math.h>
#include <functional>


using namespace std;



/**
   Implicit Surface Equations sourced from wikipedia https://en.wikipedia.org/wiki/Implicit_surface
 */
bool sphere(float x, float y, float z, float r=0.3){
  return std::sqrt(x*x + y*y + z*z) <= r;

}

bool sphereWithHole(float x, float y,float z, float outerRadius, float innerRadius){
  return (fabs(outerRadius-std::sqrt(x*x + y*y + z*z)) >= innerRadius);

}

bool torus(float x, float y, float z, float outerRadius, float innerRadius){

  return std::pow(x*x + y*y + z*z + outerRadius*outerRadius - (innerRadius*innerRadius),2) - 2*(outerRadius*outerRadius)*(x*x + y*y) >= 0;

}

//cut off at the edges
bool genus2(float x, float y, float z){
  return 2.0*y*(y*y - 3*x*x)*(1-z*z) + std::pow(x*x + y*y,2) - (9*z*z - 1)*(1-z*z) >= 0;

}

bool wineglass(float x,float y, float z){
  return x*x + y*y - std::pow(std::log(z + 3.2),2) - 0.02 >= 0;

}


int main(int argc, char *argv[]) {

  std::string gridPtsFile;
  std::string predsFile;
  std::vector<Cube> cubes ;
  if (argc > 1){
    if(argc > 2){
      gridPtsFile = argv[1];
      predsFile = argv[2];
      cubes =  MarchingCubes::assembleCubes(gridPtsFile, predsFile);
      if(argc > 3){
        std::string meshName = "mesh" + std::string(argv[3]) + ".off";
        MarchingCubes::march(cubes, meshName);
        }
      else
        MarchingCubes::march(cubes,"mesh.off");
    }
    else{
   gridPtsFile = argv[1];
   cubes = MarchingCubes::generateCubes(gridPtsFile);
   using namespace std::placeholders;
   MarchingCubes::computeOccupancies(cubes, std::bind(sphere, _1,_2,_3,1.0));
   MarchingCubes::march(cubes, "mesh.off");
   }
  }
  else{ //default hardcode

    //std::vector<Cube> cubes = MarchingCubes::assembleCubes("/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/bench_ag_32_3.txt",
    //                                                       "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/couch_interp/couch_ag_preds_32_zero.txt");

    //std::vector<Cube> cubes = MarchingCubes::generateCubes();
    //using namespace std::placeholders;
    //MarchingCubes::computeOccupancies(cubes, std::bind(sphere, _1,_2,_3,1.0));
    //MarchingCubes::computeOccupancies(cubes, std::bind(sphereWithHole, _1,_2,_3,1.0,0.1));
    //MarchingCubes::computeOccupancies(cubes, std::bind(torus, _1,_2,_3,0.5,0.4));
    //MarchingCubes::computeOccupancies(cubes, std::bind(genus2, _1,_2,_3));
    //MarchingCubes::computeOccupancies(cubes, std::bind(wineglass, _1,_2,_3));

    //MarchingCubes::march(cubes, "couch_zero.off");
  }
  return 1;
}
