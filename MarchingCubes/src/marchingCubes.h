#include <functional>
#include <vector>
#include <string>

class Vertex{
  public:
  Vertex(float xx,float yy, float zz): x(xx), y(yy), z(zz){globalIdx = -1;};
  float x;
  float y;
  float z;
  bool occupies;
  int globalIdx;
  bool operator<(const Vertex &other) const{
      if ((x-other.x) > -0.001) return true;
      if ((x-other.x) > 0.001) return false;
      if ((y-other.y) > -0.001) return true;
      if ((z-other.y) > 0.001) return false;
      if ((z-other.z) > -0.001) return true;
      return false;
  }
  bool operator==(const Vertex &other) const{
    if ((abs(x-other.x) < 0.001) && (abs(y-other.y) < 0.001) && (abs(z-other.z) < 0.001))
      return true;
    return false;
  }
};

class Cube{
public:

  std::vector<Vertex> vertices;
  
};


struct Face{
  int v1;
  int v2;
  int v3;
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




