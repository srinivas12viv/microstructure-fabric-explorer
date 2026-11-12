"""Render supplied pore-space thesis figures and package the MATLAB module.

Does not run MATLAB or regenerate segmentation. Install the module's
requirements-pore-showcase.txt only when rebuilding this presentation.
"""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / 'PoreSpaceSegmentation3D'


FIGURES = {
    'VEP': ('geometry', 'workflow', 'refinement-short', 'refinement-long',
            'sizes', 'orientations', 'sphericity'),
    'SCP': ('segmentation', 'sizes', 'sphericity'),
}
CORE_FUNCTIONS = (
    'apply_refinement_proposals', 'export_pore_features', 'step0_voxel_coordinates',
    'step1_local_maxima', 'step1_separate_porevolumes', 'step2_neighbours',
    'step3_throatcenters', 'step4_throat_body', 'step4_throat_body_surface',
    'step5b_bestfit', 'step5c_bestfit2D_robust', 'step5d_boundary', 'step5ef_filter',
    'step5ef_filtered_combined', 'step5gh_correct_inflexion_c1',
    'step5gh_correct_inflexion_robust', 'step5g_inflexion',
    'step5g_inflexion_combined_enhanced', 'step5h_lsi',
    'step5i_correct_division3d_proposal', 'step5i_inflexion3d_fromT', 'step5_refine',
)


def public_files():
    """Explicit release contents; unrelated files are never picked up implicitly."""
    relative = ['.gitignore', 'LICENSE.upstream', 'README.md', 'REFERENCES.bib',
                'THIRD_PARTY_NOTICES.md', 'poreSpaceSegmentation3D.m',
                'run_example.m', 'docs/MODIFICATIONS.md']
    relative += [f'private/{name}.m' for name in CORE_FUNCTIONS]
    for example, figures in FIGURES.items():
        relative += [f'examples/{example}/input.mat', f'examples/{example}/README.md']
        relative += [f'examples/{example}/figures/{name}.pdf' for name in figures]
    paths = [MODULE / name for name in relative]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing release file: {path}')
    return paths


def verify_example(example):
    import numpy as np
    from scipy.io import loadmat, whosmat
    folder = MODULE / 'examples' / example
    if [name for name, _, _ in whosmat(folder / 'input.mat')] != ['vox3d']:
        raise ValueError(f'{example}: example MAT must contain only vox3d')
    voxels = loadmat(folder / 'input.mat')['vox3d']
    if voxels.ndim != 3 or not np.isin(voxels, [0, 1]).all():
        raise ValueError(f'{example}: expected binary 3D vox3d')
    return dict(shape_xyz=list(voxels.shape), binary_input=True)


def write_bundle():
    target = ROOT / 'downloads' / 'pore-space-segmentation3d.zip'
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in public_files():
            archive.write(path, Path(MODULE.name) / path.relative_to(MODULE))


def build(update_gallery=True, package=True):
    import pymupdf
    output = ROOT / 'outputs' / 'pore-space'
    metadata = {'runtime': 'MATLAB', 'presentation_source': 'supplied thesis figures',
                'segmentation_regenerated': False, 'examples': {}}
    for name in ['VEP', 'SCP']:
        metadata['examples'][name] = verify_example(name)
        destination = output / name.lower()
        destination.mkdir(parents=True, exist_ok=True)
        for figure in FIGURES[name]:
            pdf = MODULE / 'examples' / name / 'figures' / (figure + '.pdf')
            with pymupdf.open(pdf) as document:
                if len(document) != 1:
                    raise ValueError(f'Expected a single-page figure: {pdf}')
                page = document[0]
                scale = min(2.5, 2000 / page.rect.width)
                page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(
                    destination / (pdf.stem + '.png'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    if update_gallery:
        path = ROOT / 'outputs' / 'metadata.json'
        gallery = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        gallery['pore_space'] = metadata
        path.write_text(json.dumps(gallery, indent=2) + '\n', encoding='utf-8')
        (ROOT / 'gallery-data.js').write_text('window.MFE_DATA = ' + json.dumps(gallery) + ';\n', encoding='utf-8')
    write_bundle()
    if package:
        from build_minkowski_showcase import refresh_full_bundle
        refresh_full_bundle()
    print('Curated pore-space code, example inputs and figures packaged. MATLAB was not run.')
    return metadata


if __name__ == '__main__':
    build()
