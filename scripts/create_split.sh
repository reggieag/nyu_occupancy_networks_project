OUTPUT_PATH=/scratch/rag551/occupancy_networks/ShapeNet

declare -a CLASSES=(
#03001627
02958343
#04256520
#02691156
#03636649
#04401088
#04530566
#03691459
#02933112
#04379243
#03211117
#02828884
#04090263
)

for c in ${CLASSES[@]}; do
    OUTPUT_PATH_C=$OUTPUT_PATH/$c

  echo "Creating split"
  python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.2
done