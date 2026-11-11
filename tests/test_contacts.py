"""Synthetic geometry and integration checks for image-based contacts."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from ContactDetection3D.config import Config
from ContactDetection3D.main import run
from ContactDetection3D.contact_detection import compute_shell
from ContactDetection3D.io_utils import load_label_tiff
from demviz import load_normals, polar_stats, spherical_stats


class ContactTests(unittest.TestCase):
    def analyze(self, labels, root, **options):
        path = root / "labels.tif"
        tifffile.imwrite(path, labels, photometric="minisblack")
        with contextlib.redirect_stdout(io.StringIO()):
            return run(Config(tiff_path=str(path), out_dir=str(root / "results"),
                              plots=False, **options))

    def test_planar_contact_and_toolkit_export(self):
        labels = np.zeros((8, 8, 8), dtype=np.uint16)
        labels[1:5, 1:5, 1:3] = 7
        labels[1:5, 1:5, 3:5] = 1000
        labels[7, 7, 7] = 42  # isolated particle, sparse IDs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.analyze(labels, root, voxel_size_mm=0.25, connectivity=6)
            self.assertEqual(result["n_contacts"], 1)
            self.assertAlmostEqual(result["mean_coordination_all"], 2/3)
            self.assertEqual(result["mean_coordination_contacting"], 1)
            with np.load(root / "results/contact_results.npz") as data:
                np.testing.assert_array_equal(data["pairs"], [[7, 1000]])
                np.testing.assert_allclose(data["centers_mm"], [[0.625, 0.625, 0.75]])
                np.testing.assert_allclose(np.abs(data["normals"]), [[0, 0, 1]])
                self.assertAlmostEqual(np.trace(data["fabric_tensor"]), 1)
            vectors, weights = load_normals(root / "results/normals.txt")
            np.testing.assert_allclose(np.abs(vectors), [[1, 0, 0]])
            np.testing.assert_allclose(weights, [1])
            self.assertEqual(polar_stats(vectors, weights)["observations"], 2)
            self.assertEqual(spherical_stats(vectors)[0].sum(), 2)

    def test_no_contacts_and_background_only(self):
        for labels in [np.zeros((4, 4, 4), dtype=np.uint8), np.ones((4, 4, 4), dtype=np.uint8)]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.analyze(labels, root)
                self.assertEqual(result["n_contacts"], 0)
                with np.load(root / "results/contact_results.npz") as data:
                    self.assertEqual(data["pairs"].shape, (0, 2))
                    self.assertEqual(data["normals"].shape, (0, 3))
                    np.testing.assert_array_equal(data["fabric_tensor"], np.zeros((3, 3)))
                self.assertEqual(len((root / "results/contacts_summary.csv").read_text().splitlines()), 1)

    def test_collinear_patch_is_discarded(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[1:4, 2, 1] = 1
        labels[1:4, 2, 2] = 2
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(labels, Path(directory), connectivity=6)
            self.assertEqual(result["candidate_pairs"], 1)
            self.assertEqual(result["n_contacts"], 0)

    def test_connectivity_and_validation(self):
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        for connectivity in (6, 18, 26):
            self.assertEqual(compute_shell(mask, 1, connectivity).sum(), connectivity)
        for kwargs in [dict(shell_thickness_vox=0), dict(voxel_size_mm=float("nan")),
                       dict(min_contact_voxels=2), dict(connectivity=9)]:
            with self.assertRaises(ValueError):
                Config(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.tif"
            tifffile.imwrite(path, np.zeros((3, 3, 3), dtype=float), photometric="minisblack")
            with self.assertRaisesRegex(ValueError, "integer"):
                load_label_tiff(path)


if __name__ == "__main__":
    unittest.main()
