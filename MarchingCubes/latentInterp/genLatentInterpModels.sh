#!/bin/bash


MARCHING_CUBES=/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/MarchingCubes/bld/MarchingCubes
GRID_FILE=/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/bench_ag_32_3.txt

echo "START" ;
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
    echo $i;
    $MARCHING_CUBES $GRID_FILE /home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/evaluation/couch_interp/couch_ag_preds_32_${i}.txt $i ;
done
