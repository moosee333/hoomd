import hoomd
from hoomd import hpmc
import unittest
hoomd.context.initialize()
print(hoomd.__file__)

class test_type_shapes(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()

    def test_type_shapes_convex_polygon(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_polygon(seed=10);
        test_verts = [(1, 0), (0, 1), (-1, -1)]
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Polygon')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))

    def test_type_shapes_simple_polygon(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.simple_polygon(seed=10);
        test_verts = [(1, 0), (0, 1), (-1, -1)]
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Polygon')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))

    def test_type_shapes_disks(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.sphere(seed=10);
        test_diam = 1
        self.mc.shape_param.set('A', diameter = test_diam)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Disk')
        self.assertEqual(shape_types[0]['diameter'], test_diam)
        self.assertNotIn('vertices', shape_types[0])

    def test_type_shapes_spheres(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.sphere(seed=10);
        test_diam = 1
        self.mc.shape_param.set('A', diameter = test_diam)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Sphere')
        self.assertEqual(shape_types[0]['diameter'], test_diam)
        self.assertNotIn('vertices', shape_types[0])

    def test_type_shapes_convex_polyhedron(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        test_verts = [(1, 0, 0), (0, 1, 0), (-1, -1, 0)]
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'ConvexPolyhedron')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))

    def tearDown(self):
        del self.mc
        del self.system
        hoomd.context.initialize()
