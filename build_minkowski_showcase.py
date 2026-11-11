"""Regenerate webpage illustrations from the actual Minkowski sample computation."""
from pathlib import Path
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Minkowski3D.main import run_particles

ROOT = Path(__file__).resolve().parent
TEAL, INK, GOLD = '#087e83', '#183039', '#d99445'


def style_3d(ax, title):
    ax.set_title(title, loc='left', color=INK, fontsize=13, pad=18)
    ax.set(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    ax.tick_params(labelsize=8, colors='#5b6c70')
    ax.view_init(elev=23, azim=-58)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis._axinfo['grid']['color'] = '#e1e8e5'


def draw_packing(ax, result):
    p = result['sites']
    ax.scatter(*p.T, s=2, c=result['radii'], cmap='viridis', alpha=.35, linewidths=0,
               rasterized=True)
    lattice = np.asarray(result['summary']['domain']['lattice_vectors'])
    corners = np.indices((2,2,2)).reshape(3,-1).T
    for i,a in enumerate(corners):
        for j,b in enumerate(corners):
            if j>i and np.sum(abs(a-b))==1:
                ax.plot(*np.array([a@lattice,b@lattice]).T,color=INK,lw=.8)
    ax.set_box_aspect(np.ptp(p,axis=0))
    style_3d(ax, '01  Particle positions + radii')


def draw_cell(ax, result):
    idx = int(np.argmax(result['ed']))
    cell = result['cells'][idx]
    center = result['centroids'][idx]
    vertices = cell['vertices']-center
    mesh = Poly3DCollection(vertices[cell['triangles']], facecolor=TEAL,
                            edgecolor='#3a7779', linewidth=.35, alpha=.14)
    ax.add_collection3d(mesh)
    theta = np.linspace(0,2*np.pi,36)
    phi = np.linspace(0,np.pi,22)
    x = np.outer(np.cos(theta),np.sin(phi))
    y = np.outer(np.sin(theta),np.sin(phi))
    z = np.outer(np.ones_like(theta),np.cos(phi))
    sphere = np.stack([x,y,z],axis=-1)
    particle = sphere*result['radii'][idx]+result['sites'][idx]-center
    ax.plot_surface(*particle.transpose(2,0,1), color=GOLD, alpha=.9, linewidth=0, shade=True)
    # In the centroid frame the solid ellipsoid has identical volume-normalized
    # second moments; semiaxes are sqrt(5 * lambda / cell volume).
    values, vectors = np.linalg.eigh(result['tensors'][idx])
    semiaxes = np.sqrt(5*values/result['volumes'][idx])
    ellipsoid = (sphere*semiaxes)@vectors.T
    ax.plot_wireframe(*ellipsoid.transpose(2,0,1),rstride=4,cstride=3,color=TEAL,lw=.6,alpha=.8)
    axis = result['axes'][idx]
    ends = np.array([-1,1])[:,None]*semiaxes[-1]*1.2*axis
    ax.plot(*ends.T,color='#bf5060',lw=2.5)
    limit = max(np.max(abs(vertices)),np.max(semiaxes))*1.1
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit))
    ax.set_box_aspect([1,1,1])
    style_3d(ax, '02  Weighted cell + moment ellipsoid')
    ax.set(xlabel='X offset (mm)',ylabel='Y offset (mm)',zlabel='Z offset (mm)')


def draw_fabric(ax, result):
    ax.hist(result['ed'],bins=np.linspace(0,1,41),color=TEAL,edgecolor='white',lw=.7)
    ax.axvline(result['summary']['Ed'],color=GOLD,lw=2)
    ax.set(xlim=(0,.8),xlabel='Local elongation, ed = 1 − β',ylabel='Cell count')
    ax.set_title('03  From local shape to assembly fabric',loc='left',color=INK,fontsize=13,pad=18)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    s = result['summary']
    ax.text(.97,.95,f"Ed  {s['Ed']:.4f}\nAd  {s['orientation_all_cells']['Ad']:.4f}",
            ha='right',va='top',transform=ax.transAxes,fontsize=13,color=INK,
            bbox=dict(facecolor='#eef4ef',edgecolor='none',pad=12))


def write_minkowski_bundle():
    downloads=ROOT/'downloads'; downloads.mkdir(exist_ok=True)
    with ZipFile(downloads/'minkowski3d.zip','w',ZIP_DEFLATED) as bundle:
        for path in sorted((ROOT/'Minkowski3D').rglob('*')):
            relative=path.relative_to(ROOT/'Minkowski3D')
            if path.is_file() and not any(part in {'__pycache__','outputs','.venv'} for part in relative.parts):
                bundle.write(path,Path('Minkowski3D')/relative)


def refresh_full_bundle():
    downloads=ROOT/'downloads'; downloads.mkdir(exist_ok=True)
    with ZipFile(downloads/'microstructure-fabric-explorer.zip','w',ZIP_DEFLATED) as bundle:
        for name in ['force-chain','polar-histogram','spherical-histogram','contact-detection','minkowski3d']:
            source=downloads/f'{name}.zip'
            if source.exists():
                bundle.write(source,f'downloads/{name}.zip')
        for path in sorted(ROOT.rglob('*')):
            parts=path.relative_to(ROOT).parts
            if path.is_file() and not any(p in {'downloads','__pycache__','.git','.venv'} for p in parts):
                if parts[:2] != ('Minkowski3D','outputs'):
                    bundle.write(path,path.relative_to(ROOT))


def build(update_gallery=True, package=True):
    output=ROOT/'outputs'/'minkowski'
    print('Computing 10,000 weighted cells for the Minkowski showcase...',flush=True)
    result=run_particles(ROOT/'Minkowski3D/example/particles.txt',ROOT/'Minkowski3D/example/domain.json',
                         output,reference='centroid')
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK,
                         'xtick.color':INK,'ytick.color':INK,'font.size':10,'savefig.facecolor':'white'})
    panels=[('packing',draw_packing,True),('cell',draw_cell,True),('fabric',draw_fabric,False)]
    for name,draw,three_d in panels:
        fig=plt.figure(figsize=(8,6),layout='constrained')
        ax=fig.add_subplot(projection='3d' if three_d else None)
        draw(ax,result)
        if name=='packing':
            fig.colorbar(ScalarMappable(norm=Normalize(result['radii'].min(),result['radii'].max()),cmap='viridis'),
                         ax=ax,shrink=.55,label='Particle radius (mm)',pad=.08)
        fig.savefig(output/f'{name}.png',dpi=200)
        plt.close(fig)
    fig=plt.figure(figsize=(18,5.7),layout='constrained')
    for i,(_,draw,three_d) in enumerate(panels):
        draw(fig.add_subplot(1,3,i+1,projection='3d' if three_d else None),result)
    fig.suptitle('Particle geometry → Voronoi cells → Minkowski fabric',fontsize=19,color=INK)
    fig.text(.5,.01,'10,000 polydisperse spheres · periodic sheared domain · centroid moments · all cells used in statistics',
             ha='center',fontsize=10,color='#5b6c70')
    fig.savefig(output/'workflow.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    if update_gallery:
        path=ROOT/'outputs'/'metadata.json'
        metadata=json.loads(path.read_text()) if path.exists() else {}
        metadata['minkowski']=result['summary']
        path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        (ROOT/'gallery-data.js').write_text('window.MFE_DATA = '+json.dumps(metadata)+';\n',encoding='utf-8')
    write_minkowski_bundle()
    if package:
        refresh_full_bundle()
    print(f'Minkowski figures and results written to {output}',flush=True)
    return result['summary']


if __name__=='__main__':
    build()
