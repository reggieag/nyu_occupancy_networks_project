#include <functional>
#include <vector>
#include <string>
#include <cmath>

class Vertex{
  public:
  Vertex(float xx,float yy, float zz): x(xx), y(yy), z(zz){globalIdx = -1; occupies=0;};
  float x;
  float y;
  float z;
  bool occupies;
  int globalIdx;
  bool operator<(const Vertex &other) const{
    float dist = std::sqrt((x-other.x)*(x-other.x) + (y-other.y)*(y-other.y) +
                             (z-other.z)*(z-other.z));

      if (dist < 0.0000001)
        return false;

      if (fabs(x - other.x) > 0.0000001)
        return x < other.x;
      if (fabs(y - other.y) > 0.0000001)
        return y < other.y;
      if (fabs(z - other.z) > 0.0000001)
        return z < other.z;
      return false;

      }

  /*
  bool operator==(const Vertex &other) const{
    if ((fabs(x-other.x) < 0.001) && (fabs(y-other.y) < 0.001) && (fabs(z-other.z) < 0.001))
      return true;
    return false;
  }*/
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
  static std::vector<Cube> generateCubes(std::string gridPtsFile="");
  
  //computes occupancy values for each vertex in each cube given implicit meshBoundary function
  static void computeOccupancies(std::vector<Cube> &cubes, std::function<bool (float, float, float)> meshBoundary);

  //given a file with cube vertices and occupancies, constructs the cubes data structure 
  static std::vector<Cube> assembleCubes(std::string pointsFile, std::string occupanciesFile);
  
  //Marching Cubes algorithm- writes mesh file to provided file name
  static void march(std::vector<Cube> &cubes, std::string meshFileName);
  
};




