#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/AABBTree.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include <iostream>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

unsigned int err_count;

/*!
 * Currently, we test convex shapes only
 */

// helper function to compute poly radius
void set_radius(poly3d_data& data)
    {
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < data.n_verts; i++)
        {
        radius_sq = std::max(radius_sq, dot(data.verts[i],data.verts[i]));
        }

    // set the diameter
    data.convex_hull_verts.diameter = 2*(sqrt(radius_sq)+data.sweep_radius);
    }

GPUTree build_tree(poly3d_data &data)
    {
    OBBTree tree;
    hpmc::detail::OBB *obbs;
    int retval = posix_memalign((void**)&obbs, 32, sizeof(hpmc::detail::OBB)*data.n_faces);
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned OBB memory.");
        }

    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;
    // construct bounding box tree
    for (unsigned int i = 0; i < data.n_faces; ++i)
        {
        std::vector< vec3<OverlapReal> > face_vec;

        unsigned int n_vert = 0;
        for (unsigned int j = data.face_offs[i]; j < data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v = data.verts[data.face_verts[j]];
            face_vec.push_back(v);
            n_vert++;
            }

        std::vector<OverlapReal> vertex_radii(n_vert,data.sweep_radius);
        obbs[i] = hpmc::detail::compute_obb(face_vec, vertex_radii, false);
        internal_coordinates.push_back(face_vec);
        }
    unsigned int capacity = 4;
    tree.buildTree(obbs, internal_coordinates, data.sweep_radius, data.n_faces, capacity, false);
    GPUTree gpu_tree(tree);
    free(obbs);
    return gpu_tree;
    }

void initialize_convex_hull(poly3d_data &data)
    {
    auto x_handle = data.convex_hull_verts.x.requestWriteAccess();
    auto y_handle = data.convex_hull_verts.y.requestWriteAccess();
    auto z_handle = data.convex_hull_verts.z.requestWriteAccess();
    // for simplicity, use all vertices instead of convex hull
    for (unsigned int i = 0; i < data.n_verts; ++i)
        {
        x_handle[i] = data.verts[i].x;
        y_handle[i] = data.verts[i].y;
        z_handle[i] = data.verts[i].z;
        }
    }

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    poly3d_data data(4,1,4,4,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.0f;
    auto verts_handle = data.verts.requestWriteAccess();
    auto face_verts_handle = data.face_verts.requestWriteAccess();
    auto face_offs_handle = data.face_offs.requestWriteAccess();
    verts_handle[0] = vec3<OverlapReal>(0,0,0);
    verts_handle[1] = vec3<OverlapReal>(1,0,0);
    verts_handle[2] = vec3<OverlapReal>(0,1.25,0);
    verts_handle[3] = vec3<OverlapReal>(0,0,1.1);
    face_verts_handle[0] = 0;
    face_verts_handle[1] = 1;
    face_verts_handle[2] = 2;
    face_verts_handle[3] = 3;
    face_offs_handle[0] = 0;
    face_offs_handle[1] = 4;
    data.ignore = 0;
    set_radius(data);
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o, p);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.data.n_verts, data.n_verts);
    for (unsigned int i = 0; i < data.n_verts; i++)
        {
        MY_CHECK_CLOSE(a.data.verts[i].x, data.verts[i].x, tol);
        MY_CHECK_CLOSE(a.data.verts[i].y, data.verts[i].y, tol);
        MY_CHECK_CLOSE(a.data.verts[i].z, data.verts[i].z, tol);
        }

    UP_ASSERT_EQUAL(a.data.n_faces, data.n_faces);
    for (unsigned int i = 0; i < data.n_faces; i++)
        {
        UP_ASSERT_EQUAL(a.data.face_offs[i], data.face_offs[i]);
        unsigned int offs = a.data.face_offs[i];
        for (unsigned int j = offs; j < a.data.face_offs[i+1]; ++j)
            UP_ASSERT_EQUAL(a.data.face_verts[j], data.face_verts[j]);
        }

    UP_ASSERT_EQUAL(a.data.face_offs[data.n_faces], data.face_offs[data.n_faces]);
    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST( overlap_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data(6,8,24,6,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.0f;

    auto verts_handle = data.verts.requestWriteAccess();
    auto face_verts_handle = data.face_verts.requestWriteAccess();
    auto face_offs_handle = data.face_offs.requestWriteAccess();

    verts_handle[0] = vec3<OverlapReal>(-0.5,-0.5,0);
    verts_handle[1] = vec3<OverlapReal>(0.5,-0.5,0);
    verts_handle[2] = vec3<OverlapReal>(0.5,0.5,0);
    verts_handle[3] = vec3<OverlapReal>(-0.5,0.5,0);
    verts_handle[4] = vec3<OverlapReal>(0,0,0.707106781186548);
    verts_handle[5] = vec3<OverlapReal>(0,0,-0.707106781186548);
    face_offs_handle[0] = 0;
    face_verts_handle[0] = 0; face_verts_handle[1] = 4; face_verts_handle[2] = 1;
    face_offs_handle[1] = 3;
    face_verts_handle[3] = 1; face_verts_handle[4] = 4; face_verts_handle[5] = 2;
    face_offs_handle[2] = 6;
    face_verts_handle[6] = 2; face_verts_handle[7] = 4; face_verts_handle[8] = 3;
    face_offs_handle[3] = 9;
    face_verts_handle[9] = 3; face_verts_handle[10] = 4; face_verts_handle[11] = 0;
    face_offs_handle[4] = 12;
    face_verts_handle[12] = 0; face_verts_handle[13] = 5; face_verts_handle[14] = 1;
    face_offs_handle[5] = 15;
    face_verts_handle[15] = 1; face_verts_handle[16] = 5; face_verts_handle[17] = 2;
    face_offs_handle[6] = 18;
    face_verts_handle[18] = 2; face_verts_handle[19] = 5; face_verts_handle[20] = 3;
    face_offs_handle[7] = 21;
    face_verts_handle[21] = 3; face_verts_handle[22] = 5; face_verts_handle[23] = 0;
    face_offs_handle[8] = 24;
    data.ignore = 0;
    set_radius(data);
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
    p.tree = build_tree(data);

    ShapePolyhedron a(o, p);
    ShapePolyhedron b(o, p);

    // zeroth test: exactly overlapping shapes
    r_ij =  vec3<Scalar>(0.0, 0.0, 0.0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate octahedrons by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij =  vec3<Scalar>(1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.0,0.2,0);

    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }


UP_TEST( overlap_sphero_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data(6,8,24,6,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.1f;

    auto verts_handle = data.verts.requestWriteAccess();
    auto face_verts_handle = data.face_verts.requestWriteAccess();
    auto face_offs_handle = data.face_offs.requestWriteAccess();


    verts_handle[0] = vec3<OverlapReal>(-0.5,-0.5,0);
    verts_handle[1] = vec3<OverlapReal>(0.5,-0.5,0);
    verts_handle[2] = vec3<OverlapReal>(0.5,0.5,0);
    verts_handle[3] = vec3<OverlapReal>(-0.5,0.5,0);
    verts_handle[4] = vec3<OverlapReal>(0,0,0.707106781186548);
    verts_handle[5] = vec3<OverlapReal>(0,0,-0.707106781186548);
    face_offs_handle[0] = 0;
    face_verts_handle[0] = 0; face_verts_handle[1] = 4; face_verts_handle[2] = 1;
    face_offs_handle[1] = 3;
    face_verts_handle[3] = 1; face_verts_handle[4] = 4; face_verts_handle[5] = 2;
    face_offs_handle[2] = 6;
    face_verts_handle[6] = 2; face_verts_handle[7] = 4; face_verts_handle[8] = 3;
    face_offs_handle[3] = 9;
    face_verts_handle[9] = 3; face_verts_handle[10] = 4; face_verts_handle[11] = 0;
    face_offs_handle[4] = 12;
    face_verts_handle[12] = 0; face_verts_handle[13] = 5; face_verts_handle[14] = 1;
    face_offs_handle[5] = 15;
    face_verts_handle[15] = 1; face_verts_handle[16] = 5; face_verts_handle[17] = 2;
    face_offs_handle[6] = 18;
    face_verts_handle[18] = 2; face_verts_handle[19] = 5; face_verts_handle[20] = 3;
    face_offs_handle[7] = 21;
    face_verts_handle[21] = 3; face_verts_handle[22] = 5; face_verts_handle[23] = 0;
    face_offs_handle[8] = 24;
    data.ignore = 0;
    set_radius(data);
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
    p.tree = build_tree(data);

    ShapePolyhedron a(o, p);
    ShapePolyhedron b(o, p);

    // zeroth test: exactly overlapping shapes
    r_ij =  vec3<Scalar>(0.0, 0.0, 0.0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate octahedrons by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij =  vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.1,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.1999,0.2,0);

    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.20001,0.2,0);

    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
    }

