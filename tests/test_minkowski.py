"""Analytic moments, covariance, fabric conventions, and weighted periodic integration."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from Minkowski3D.main import polyhedron_moments, fabric_statistics, load_particles, analyze, save_results


VERTICES = np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
FACES = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]


class MomentTests(unittest.TestCase):
    def test_box_and_reference_shift(self):
        vertices=VERTICES*np.array([1.,2.,3.])
        volume, center, tensor, _=polyhedron_moments(vertices,FACES)
        self.assertAlmostEqual(volume,48)
        np.testing.assert_allclose(center,0,atol=1e-15)
        np.testing.assert_allclose(tensor,np.diag([16.,64.,144.]),atol=1e-13)
        ref=np.array([4.,-2.,1.])
        shifted=polyhedron_moments(vertices,FACES,ref)[2]
        np.testing.assert_allclose(shifted,tensor+volume*np.outer(ref,ref))

    def test_unit_tetrahedron(self):
        vertices=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
        volume,center,tensor,_=polyhedron_moments(vertices,[[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.zeros(3))
        self.assertAlmostEqual(volume,1/6)
        np.testing.assert_allclose(center,[.25,.25,.25])
        np.testing.assert_allclose(tensor,(np.ones((3,3))+np.eye(3))/120)

    def test_rotation_translation_scaling_and_winding(self):
        angle=.71
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        vertices=VERTICES*np.array([1.,2.,3.]); shift=np.array([120.,-90.,22.])
        volume,center,tensor,_=polyhedron_moments(vertices,FACES)
        newvol,newcenter,newtensor,_=polyhedron_moments(2*vertices@rotation.T+shift,
                                                        [face[::-1] for face in FACES])
        self.assertAlmostEqual(newvol,8*volume)
        np.testing.assert_allclose(newcenter,shift,atol=1e-13)
        np.testing.assert_allclose(newtensor,32*rotation@tensor@rotation.T,rtol=1e-12,atol=1e-11)

    def test_open_surface_rejected(self):
        with self.assertRaisesRegex(ValueError,'closed'):
            polyhedron_moments(VERTICES,FACES[:-1])

    def test_fabric_limits_and_ambiguous_axes(self):
        aligned=np.tile(np.diag([1.,2.,4.]),(10,1,1))
        summary,*_=fabric_statistics(aligned)
        self.assertAlmostEqual(summary['Ed'],.75)
        self.assertAlmostEqual(summary['mean_beta'],.25)
        self.assertAlmostEqual(summary['orientation_all_cells']['Ad'],7.5)
        isotropic_axes=np.array([np.diag([4.,1.,1.]),np.diag([1.,4.,1.]),np.diag([1.,1.,4.])])
        s,*_=fabric_statistics(isotropic_axes)
        self.assertAlmostEqual(s['orientation_all_cells']['Ad'],0)
        s,*_=fabric_statistics(np.eye(3)[None])
        self.assertEqual(s['ambiguous_axes'],1)
        self.assertIsNone(s['orientation_resolved_axes']['Ad'])

    def test_loader_requires_radii_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'particles.txt'
            path.write_text('id x y z\n81 1 2 3\n4 2 3 4\n')
            with self.assertRaisesRegex(ValueError,'no radii'):
                load_particles(path)
            ids,_,radii=load_particles(path,uniform_radius=.4)
            np.testing.assert_array_equal(ids,[81,4])
            np.testing.assert_allclose(radii,.4)


@unittest.skipUnless(importlib.util.find_spec('pyvoro2'),'Optional pyvoro2 backend not installed')
class TessellationTests(unittest.TestCase):
    def test_weighted_two_particle_split(self):
        # Power equality gives x=(d^2+r0^2-r1^2)/(2d)=0.85.
        points=np.array([[0.,0.,0.],[2.,0.,0.]])
        config=dict(bounds=[[-1,3],[-1,1],[-1,1]],periodic=[False]*3,units='test')
        r=analyze(np.array([91,7]),points,np.array([.5,np.sqrt(.85)]),config)
        np.testing.assert_allclose(r['volumes'],[7.4,8.6])
        np.testing.assert_array_equal(r['ids'],[91,7])
        self.assertAlmostEqual(np.trace(r['summary']['orientation_all_cells']['V']),1)
        with tempfile.TemporaryDirectory() as directory:
            save_results(r,directory)
            with np.load(Path(directory)/'tensors.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['ids'],[91,7])

    def test_periodic_single_cell_analytic_cube(self):
        config=dict(lattice_vectors=[[2.,0,0],[0,2.,0],[0,0,2.]],units='test')
        r=analyze(np.array([35]),np.array([[.2,.3,.4]]),np.array([.2]),config)
        self.assertAlmostEqual(r['volumes'][0],8)
        np.testing.assert_allclose(r['tensors'][0],np.eye(3)*8/3,atol=1e-13)
        self.assertEqual(r['summary']['ambiguous_axes'],1)

    def test_triclinic_periodic_translation(self):
        lattice=np.array([[3.,0,0],[-.4,2.5,0],[.2,.1,2.]])
        points=(np.indices((2,2,2)).reshape(3,-1).T+.5)/2 @ lattice
        ids=np.arange(8)*3+10; radii=np.linspace(.12,.2,8)
        config=dict(lattice_vectors=lattice.tolist(),units='test')
        a=analyze(ids,points,radii,config)
        b=analyze(ids,points+lattice[1],radii,config)
        np.testing.assert_allclose(a['volumes'],b['volumes'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a['tensors'],b['tensors'],rtol=1e-11,atol=1e-12)


if __name__=='__main__':
    unittest.main()
