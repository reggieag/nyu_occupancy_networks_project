#!/bin/env/python
#Reads in a .obj file coming out of the Shapenet dataset
#Writes a simplied .off file

import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert .obj to .off')
    parser.add_argument('objFile', metavar='f', type=str, nargs=1,
                        help='Specify the objfile')
    args = parser.parse_args()

    outFileName = f"{args.objFile[0].split('.')[0]}.off"
    outFile = open(outFileName, 'w')
    outFile.write("OFF\n")
    vertices = []
    faces = []
    with open(args.objFile[0], 'r') as file:
        nVertices = 0
        nFaces = 0
        for line in file:
            x = line.strip('\n').split(' ')
             #skip comments
            if x[0] == "#": continue 
            ##skip material lines
            if x[0] == "mtlib": continue 
            if x[0] == "vt": continue
            if x[0] == "usemtl" : continue
            #skip smooth shading
            if x[0] == "s":continue 
            #skip object naming
            if x[0] == "o": continue
            if x[0] == "v":
                nVertices = nVertices + 1
                vertices.append(list(map(float,x[1:4])))
            if x[0] == "f": 
                nFaces = nFaces + 1
                #ignore texture vertex
                verts = list(map(lambda x: int(x.split('/')[0]), x[1:4]))
                verts.sort()
                faces.append(verts)

    #sort the faces by the minimum vertex in the face
    faces.sort(key=lambda row: min(row))
    
    outFile.write(f"{nVertices} {nFaces} 0\n")
    for v in vertices:
        outFile.write(f"{v[0]} {v[1]} {v[2]} \n")
    for f in faces:
        outFile.write(f"3 {f[0]} {f[1]} {f[2]}\n")
    outFile.close()

if __name__ == '__main__':
    main()
