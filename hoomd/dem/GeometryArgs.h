
#include "hoomd/hoomd_config.h"

#ifndef __GEOMETRYARGS_H__
#define __GEOMETRYARGS_H__

class GeometryArgs
{
public:
    GeometryArgs(unsigned int vertCount, Scalar3 *verts, unsigned int *starts, unsigned int *counts):
        numVertices(vertCount), vertices(verts), typeStarts(starts), typeCounts(counts) {}

    unsigned int numVertices;
    Scalar3 *vertices;
    unsigned int *typeStarts;
    unsigned int *typeCounts;
};

#endif
