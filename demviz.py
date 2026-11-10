"""Reusable, offline microstructure and fabric visualization tools."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent


def load_table(path, columns):
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if not data.size or data.shape[1] != columns or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected finite data with {columns} columns')
    return data


def load_normals(path):
    data = load_table(path, 4)
    lengths = np.linalg.norm(data[:, :3], axis=1)
    if np.any(lengths == 0) or np.any(data[:, 3] < 0):
        raise ValueError('Normals must be nonzero and force magnitudes nonnegative')
    return data[:, :3]/lengths[:, None], data[:, 3]


def polar_stats(vectors, forces, plane='xy', bins=36):
    """Antipodal counts, mean force and least-squares second harmonic."""
    if bins < 4:
        raise ValueError('Use at least four bins')
    projected = vectors[:, {'xy': (0,1), 'xz': (0,2), 'yz': (1,2)}[plane]]
    valid = np.linalg.norm(projected, axis=1) > 1e-12
    if not valid.any():
        raise ValueError('No nonzero normals in selected projection')
    theta = np.mod(np.arctan2(projected[valid,1], projected[valid,0]), 2*np.pi)
    theta = np.concatenate([theta, (theta+np.pi) % (2*np.pi)])
    edges = np.linspace(0, 2*np.pi, bins+1)
    counts, _ = np.histogram(theta, edges)
    sums, _ = np.histogram(theta, edges, weights=np.tile(forces[valid], 2))
    mean = np.divide(sums, counts, out=np.zeros(bins), where=counts > 0)
    centers = (edges[1:]+edges[:-1])/2
    width = edges[1]-edges[0]
    design = np.column_stack([np.ones(bins), np.cos(2*centers), np.sin(2*centers)])
    coeff = np.linalg.lstsq(design, counts/(len(theta)*width), rcond=None)[0]
    return dict(counts=counts, mean_force=mean, centers=centers, coeff=coeff,
        amplitude=float(2*np.pi*np.hypot(coeff[1], coeff[2])),
        direction=float(np.degrees(0.5*np.arctan2(coeff[2], coeff[1])) % 180),
        observations=len(theta), excluded=int((~valid).sum()), width=width)


def spherical_stats(vectors, longitude_bins=36, latitude_bins=18):
    """Equal solid angle bins: uniform azimuth and cos(colatitude)."""
    if longitude_bins < 4 or latitude_bins < 2:
        raise ValueError('Use at least 4 longitude and 2 latitude bins')
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths == 0) or not np.isfinite(vectors).all():
        raise ValueError('Normals must be finite and nonzero')
    unit = vectors/lengths[:,None]
    unit = np.concatenate([unit, -unit])
    phi = np.mod(np.arctan2(unit[:,1], unit[:,0]), 2*np.pi)
    azimuth = np.linspace(0, 2*np.pi, longitude_bins+1)
    mu = np.linspace(-1, 1, latitude_bins+1)
    counts, _, _ = np.histogram2d(phi, np.clip(unit[:,2], -1, 1), bins=(azimuth, mu))
    return counts.astype(int), azimuth, mu


def style_figure(fig, title):
    fig.update_layout(title=dict(text=title, font=dict(size=18)),
        font=dict(family='Arial, sans-serif', color='#172c36'), paper_bgcolor='white',
        plot_bgcolor='white', margin=dict(l=55,r=55,t=65,b=45), autosize=True)
    return fig


def write_plot(fig, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(destination, include_plotlyjs='directory', full_html=True,
        config=dict(responsive=True, displaylogo=False, scrollZoom=False))


def polar_figure(stats, title):
    fig = go.Figure(go.Barpolar(r=stats['counts'], theta=np.degrees(stats['centers']),
        width=np.degrees(stats['width']), customdata=stats['mean_force'],
        marker=dict(color=stats['mean_force'], colorscale='Teal',
            line=dict(color='white',width=0.6), colorbar=dict(title='Mean force<br>(input units)')),
        name='Contact count', hovertemplate='%{theta:.0f}°<br>Count: %{r}<br>Mean force: %{customdata:.3g}<extra></extra>'))
    angles = np.linspace(0, 2*np.pi, 361)
    c = stats['coeff']
    fit = (c[0]+c[1]*np.cos(2*angles)+c[2]*np.sin(2*angles))*stats['observations']*stats['width']
    fig.add_trace(go.Scatterpolar(r=fit, theta=np.degrees(angles), mode='lines',
        line=dict(color='#db7047',width=3), name='Second harmonic fit'))
    fig.update_layout(legend=dict(orientation='h',y=-0.12),
        polar=dict(radialaxis=dict(title='Count'), angularaxis=dict(rotation=0,direction='counterclockwise')))
    return style_figure(fig, title)


def spherical_figure(counts, azimuth, mu, title):
    vertices, triangles, intensity = [], [], []
    for i,j in zip(*np.nonzero(counts)):
        r = float(counts[i,j])
        base = len(vertices)
        vertices.append([0.,0.,0.])
        for phi,z in [(azimuth[i],mu[j]),(azimuth[i+1],mu[j]),
                      (azimuth[i+1],mu[j+1]),(azimuth[i],mu[j+1])]:
            s = np.sqrt(max(0, 1-z*z))
            vertices.append([r*s*np.cos(phi),r*s*np.sin(phi),r*z])
        for face in [(1,2,3),(1,3,4),(0,1,2),(0,2,3),(0,3,4),(0,4,1)]:
            triangles.append([base+k for k in face])
            intensity.append(r)
    xyz, ijk = np.array(vertices).T, np.array(triangles).T
    fig = go.Figure(go.Mesh3d(x=xyz[0],y=xyz[1],z=xyz[2],i=ijk[0],j=ijk[1],k=ijk[2],
        intensity=intensity,intensitymode='cell',colorscale='Teal',cmin=0,cmax=int(counts.max()),
        colorbar=dict(title='Contact count'),flatshading=True,
        hovertemplate='Count: %{intensity}<extra></extra>'))
    fig.update_layout(scene=dict(aspectmode='data',xaxis=dict(title='X · count'),
        yaxis=dict(title='Y · count'),zaxis=dict(title='Z · count'),
        camera=dict(eye=dict(x=1.55,y=1.55,z=1.1))))
    return style_figure(fig, title)


def force_plot(positions_path, contacts_path, destination, max_component_gap=0.1, dpi=220):
    positions, contacts = load_table(positions_path,4), load_table(contacts_path,6)
    ids = positions[:,0]
    if np.any(ids != np.floor(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError('Particle IDs must be unique integers')
    if np.any(contacts[:,:2] != np.floor(contacts[:,:2])) or np.any(contacts[:,5] < 0):
        raise ValueError('Contact IDs must be integers; forces must be nonnegative')
    lookup = {int(row[0]):row[1:3] for row in positions}
    try:
        segments = np.array([[lookup[int(a)],lookup[int(b)]] for a,b in contacts[:,:2]])
    except KeyError as exc:
        raise ValueError(f'Unknown particle {exc.args[0]}') from exc
    keep = np.ones(len(contacts),dtype=bool)
    if max_component_gap is not None:
        if max_component_gap <= 0:
            raise ValueError('Maximum component gap must be positive')
        keep = np.all(np.abs(segments[:,1]-segments[:,0]) < max_component_gap,axis=1)
    segments, forces = segments[keep], contacts[keep,5]
    if not len(forces):
        raise ValueError('No contacts remain after filtering')
    peak = max(float(forces.max()),np.finfo(float).eps)
    fig,ax = plt.subplots(figsize=(8,7),facecolor='#102a35')
    ax.set_facecolor('#102a35')
    collection = LineCollection(segments,cmap='inferno',norm=plt.Normalize(0,peak),
        linewidths=0.15+2.5*forces/peak)
    collection.set_array(forces)
    ax.add_collection(collection)
    ax.autoscale()
    ax.margins(0.025)
    ax.set_aspect('equal')
    ax.set_axis_off()
    bar = fig.colorbar(collection,ax=ax,shrink=0.72,pad=0.02)
    bar.set_label('Normal force (input units)',color='white')
    bar.ax.tick_params(colors='white')
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=dpi,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(particles=len(positions),contacts=len(contacts),plotted=int(keep.sum()),
        excluded=int((~keep).sum()),maximum_force=float(forces.max()))


def main(kind):
    parser = argparse.ArgumentParser(description=f'Microstructure & Fabric Explorer: {kind} visualization')
    if kind == 'force':
        parser.add_argument('--positions',type=Path,default=ROOT/'ForceChain2D/pos.txt')
        parser.add_argument('--contacts',type=Path,default=ROOT/'ForceChain2D/normals.txt')
        parser.add_argument('--max-gap',type=float,default=0.1,help='Legacy distance cutoff; 0 disables')
        parser.add_argument('--output',type=Path,default=ROOT/'outputs/force-chain.png')
    else:
        folder = 'PolarHistogram2D' if kind == 'polar' else 'SphericalHistogram3D'
        parser.add_argument('--input',type=Path,default=ROOT/folder/'normals_Aniso.txt')
        parser.add_argument('--output',type=Path,default=ROOT/'outputs'/f'{kind}.html')
        if kind == 'polar':
            parser.add_argument('--plane',choices=['xy','xz','yz'],default='xy')
            parser.add_argument('--bins',type=int,default=36)
        else:
            parser.add_argument('--longitude-bins',type=int,default=36)
            parser.add_argument('--latitude-bins',type=int,default=18)
    args = parser.parse_args()
    try:
        if kind == 'force':
            print(json.dumps(force_plot(args.positions,args.contacts,args.output,args.max_gap or None),indent=2))
        else:
            vectors,forces = load_normals(args.input)
            if kind == 'polar':
                fig = polar_figure(polar_stats(vectors,forces,args.plane,args.bins),f'{args.input.stem} · {args.plane.upper()}')
            else:
                fig = spherical_figure(*spherical_stats(vectors,args.longitude_bins,args.latitude_bins),args.input.stem)
            write_plot(fig,args.output)
        print(f'Saved {args.output.resolve()}')
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
