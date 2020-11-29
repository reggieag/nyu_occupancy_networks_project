#include <marchingCubes.h>

#include <iostream>
#include <fstream>

std::vector<Cube> MarchingCubes::assembleCubes(std::string pointsFile, std::string occupanciesFile){

  std::vector<Cube> cubes;
  std::ifstream ptsFile(pointsFile);
  float x,y,z;

  while(ptsFile.good()){
    Cube cube;
    for (int i = 0; i <8; i++){
      ptsFile >> x;
      ptsFile >> y;
      ptsFile >> z;
      Vertex v(x,y,z);
      cube.vertices.push_back(v);
    }
    cubes.push_back(cube);
  }


  float threshold = 0.1;
  std::ifstream occFile(occupanciesFile);
  int cubeidx =-1;
  int vtxidx = 0;
  bool occupied;
  float pred;
  while(occFile.good()){
    occFile >> pred;
    if(pred >= threshold)
      occupied = true;
    else {
      occupied = false;
    }

    if(vtxidx % 8 == 0){
      cubeidx += 1;
      vtxidx = 0;
    }
    else {
      vtxidx += 1;
    }
    if(cubeidx < cubes.size())
      cubes[cubeidx].vertices[vtxidx].occupies = occupied;
    else{
      //why even
      break;
    }
  }

  return cubes;
}

void march(std::vector<Cube> & cubes, std::string meshFileName){


}
