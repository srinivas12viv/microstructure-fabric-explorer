"""Numerical regression tests: conservation, axes, sparse bins and validation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from demviz import ROOT, load_normals, polar_stats, spherical_stats, force_plot


class HistogramTests(unittest.TestCase):
    def test_supplied_data_conserves_all_rows(self):
        for name, expected in [('Iso',21169),('Aniso',20650)]:
            vectors, forces = load_normals(ROOT/f'PolarHistogram2D/normals_{name}.txt')
            self.assertEqual(len(vectors),expected)
            for plane in ['xy','xz','yz']:
                stats = polar_stats(vectors,forces,plane)
                self.assertEqual(int(stats['counts'].sum()),2*expected)
            counts, _, _ = spherical_stats(vectors)
            self.assertEqual(int(counts.sum()),2*expected)

    def test_poles_and_axes_are_counted(self):
        counts, _, _ = spherical_stats(np.array([[0,0,1],[0,0,-1],[1,0,0],[0,-1,0]]))
        self.assertEqual(int(counts.sum()),8)
        self.assertEqual(int(counts[:,-1].sum()),2)
        self.assertEqual(int(counts[:,0].sum()),2)

    def test_sparse_bins_and_mean_force(self):
        stats = polar_stats(np.array([[1.,0,0],[1.,0,0]]),np.array([2.,4.]))
        self.assertEqual(int(stats['counts'].sum()),4)
        self.assertTrue(np.isfinite(stats['mean_force']).all())
        np.testing.assert_allclose(stats['mean_force'][stats['counts'] > 0],3)

    def test_second_harmonic_quadrant(self):
        theta = np.radians(125)
        vectors = np.tile([np.cos(theta),np.sin(theta),0],(100,1))
        stats = polar_stats(vectors,np.ones(100))
        self.assertAlmostEqual(stats['direction'],125,places=8)

    def test_zero_projection_is_excluded(self):
        stats = polar_stats(np.array([[0.,0,1],[1,0,0]]),np.ones(2))
        self.assertEqual(stats['excluded'],1)
        self.assertEqual(int(stats['counts'].sum()),2)

    def test_isotropic_second_harmonic(self):
        theta = np.linspace(0,2*np.pi,3600,endpoint=False)+0.00001
        vectors = np.column_stack([np.cos(theta),np.sin(theta),np.zeros(len(theta))])
        stats = polar_stats(vectors,np.ones(len(theta)))
        self.assertLess(stats['amplitude'],1e-10)

    def test_force_validation_and_zero_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positions,contacts = root/'pos.txt',root/'contacts.txt'
            positions.write_text('id x y z\n0 0 0 0\n1 0.01 0.01 0\n')
            contacts.write_text('id1 id2 nx ny nz f\n0 1 1 0 0 0\n')
            result = force_plot(positions,contacts,root/'figure.png',dpi=30)
            self.assertEqual(result['plotted'],1)
            contacts.write_text('id1 id2 nx ny nz f\n0 99 1 0 0 1\n')
            with self.assertRaisesRegex(ValueError,'Unknown particle'):
                force_plot(positions,contacts,root/'bad.png')


if __name__ == '__main__':
    unittest.main()
