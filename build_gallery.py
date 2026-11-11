"""Regenerate all showcase outputs and downloadable example bundles."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ContactDetection3D.main import run as run_contacts
from ContactDetection3D.config import Config
from demviz import (ROOT, load_normals, polar_stats, spherical_stats, force_plot,
                    polar_figure, spherical_figure, write_plot)


def build():
    output = ROOT/'outputs'
    output.mkdir(exist_ok=True)
    contact_metadata = run_contacts(Config())
    metadata = {'contacts3d': contact_metadata, 'force': force_plot(ROOT/'ForceChain2D/pos.txt', ROOT/'ForceChain2D/normals.txt', output/'force-chain.png')}
    for dataset in ['Iso', 'Aniso']:
        vectors, forces = load_normals(ROOT/f'PolarHistogram2D/normals_{dataset}.txt')
        metadata[dataset] = {'contacts':len(vectors), 'planes':{}}
        for plane in ['xy','xz','yz']:
            stats = polar_stats(vectors,forces,plane)
            write_plot(polar_figure(stats,f'{dataset} dataset · {plane.upper()} projection'),output/f'polar-{dataset.lower()}-{plane}.html')
            metadata[dataset]['planes'][plane] = {key:stats[key] for key in ['amplitude','direction','excluded','observations']}
            np.savetxt(output/f'polar-{dataset.lower()}-{plane}.csv',
                np.column_stack([np.degrees(stats['centers']),stats['counts'],stats['mean_force']]),
                delimiter=',', header='angle_degrees,antipodal_count,mean_force',comments='')
        counts,azimuth,mu = spherical_stats(vectors)
        write_plot(spherical_figure(counts,azimuth,mu,f'{dataset} dataset · equal solid angle'),output/f'sphere-{dataset.lower()}.html')
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(projection='3d')
        p,m = np.meshgrid((azimuth[1:]+azimuth[:-1])/2,(mu[1:]+mu[:-1])/2,indexing='ij')
        s = np.sqrt(1-m*m)
        ax.plot_surface(counts*s*np.cos(p),counts*s*np.sin(p),counts*m,cmap='viridis',linewidth=0.2)
        ax.set(xlabel='X · count',ylabel='Y · count',zlabel='Z · count',title=f'{dataset} · 3D contact orientation')
        fig.tight_layout()
        fig.savefig(output/f'sphere-{dataset.lower()}.png',dpi=160)
        plt.close(fig)
        np.savetxt(output/f'sphere-{dataset.lower()}.csv',
            np.array([[azimuth[i],azimuth[i+1],mu[j],mu[j+1],counts[i,j]] for i in range(36) for j in range(18)]),
            delimiter=',',header='azimuth_start_rad,azimuth_end_rad,cos_theta_start,cos_theta_end,antipodal_count',comments='')
        metadata[dataset]['spherical_observations'] = int(counts.sum())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (ROOT/'gallery-data.js').write_text('window.MFE_DATA = '+json.dumps(metadata)+';\n',encoding='utf-8')
    downloads = ROOT/'downloads'
    downloads.mkdir(exist_ok=True)
    common = ['demviz.py','requirements.txt','README.md','LICENSE-NOTE.md']
    for name,folder in [('force-chain','ForceChain2D'),('polar-histogram','PolarHistogram2D'),('spherical-histogram','SphericalHistogram3D'),('contact-detection','ContactDetection3D')]:
        with ZipFile(downloads/f'{name}.zip','w',ZIP_DEFLATED) as bundle:
            for relative in common:
                bundle.write(ROOT/relative,relative)
            for path in (ROOT/folder).iterdir():
                if path.suffix in ['.py','.txt','.md','.tif']:
                    bundle.write(path,path.relative_to(ROOT))
            if folder == 'ContactDetection3D':
                for path in (ROOT/folder/'out').iterdir():
                    if path.suffix in {'.svg', '.pdf', '.csv', '.npz'}:
                        bundle.write(path,path.relative_to(ROOT))
    with ZipFile(downloads/'microstructure-fabric-explorer.zip','w',ZIP_DEFLATED) as bundle:
        # Include example downloads without recursively including this archive.
        for name in ['force-chain.zip', 'polar-histogram.zip', 'spherical-histogram.zip', 'contact-detection.zip']:
            bundle.write(downloads/name, f'downloads/{name}')
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and not any(part in {'downloads','__pycache__','.git','.venv'} for part in path.relative_to(ROOT).parts):
                bundle.write(path,path.relative_to(ROOT))
    print(json.dumps(metadata,indent=2))
    print('Built gallery outputs and five download bundles.')


if __name__ == '__main__':
    build()
