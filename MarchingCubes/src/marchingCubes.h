#include <functional>
#include <vector>
#include <string>

class Vertex{
  public:
  Vertex(float xx,float yy, float zz): x(xx), y(yy), z(zz){};
  float x;
  float y;
  float z;
  bool occupies;
};

class Cube{
public:

  std::vector<Vertex> vertices;
  
};

class MarchingCubes{
public:
  //Divides the canonical cube into a grid of cubes
  static std::vector<Cube> generateCubes();
  
  //computes occupancy values for each vertex in each cube given implicit meshBoundary function
  static computeOccupancies(std::vector<Cube> &cubes, std::function<bool (float, float, float)> meshBoundary);

  //given a file with cube vertices and occupancies, constructs the cubes data structure 
  static std::vector<Cube> assembleCubes(std::string pointsFile, std::string occupanciesFile);
  
  //Marching Cubes algorithm- writes mesh file to provided file name
  static void march(std::vector<Cube> &cubes, std::string meshFileName);
  
};




