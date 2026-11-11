"""Standalone positions -> Voronoi -> W020 -> assembly fabric workflow.

Only tessellation uses pyvoro2/Voro++; moment integration is implemented here.
Lattice vectors in JSON are rows (a, b, c), in particle coordinate units.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import numpy as np


def polyhedron_moments(vertices, faces, reference=None, geometry_tolerance=1e-8):
    """Integrate a closed convex polyhedron, repairing either face winding.

    Return positive volume, volume centroid, W020 about reference (default:
    centroid), and outward triangle indices. Integration uses exact simplex
    second moments; no numerical volume quadrature is performed.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError('Expected at least four finite 3D vertices')
    q = v.mean(axis=0)
    scale = np.max(np.linalg.norm(v-q, axis=1))
    if scale <= 0:
        raise ValueError('Degenerate cell')
    triangles, edges = [], {}
    for face in faces:
        raw = np.asarray(face)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Face indices must be integers')
        ids = raw.astype(int)
        if len(ids) < 3 or len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(v)):
            raise ValueError('Invalid polygon indices')
        for a, b in zip(ids, np.roll(ids, -1)):
            edge = tuple(sorted((int(a), int(b))))
            edges[edge] = edges.get(edge, 0) + 1
        polygon = v[ids]
        normal = np.cross(polygon-polygon[0], np.roll(polygon,-1,axis=0)-polygon[0]).sum(axis=0)
        norm = np.linalg.norm(normal)
        if norm <= np.finfo(float).eps * scale**2 * 32:
            raise ValueError('Degenerate polygon')
        if np.dot(normal, polygon[0]-q) < 0:
            ids = ids[::-1]
            normal = -normal
        if (np.max((v-polygon[0]) @ normal) > geometry_tolerance * scale * norm
                or np.max(np.abs((polygon-polygon[0]) @ normal)) > geometry_tolerance * scale * norm):
            raise ValueError('Cell is nonconvex or a polygon is nonplanar')
        for j in range(1, len(ids)-1):
            tri = [int(ids[0]), int(ids[j]), int(ids[j+1])]
            a, b, c = v[tri]
            triangle_normal = np.cross(b-a, c-a)
            # Collinear vertices occasionally occur on polygonal boundaries.
            if np.linalg.norm(triangle_normal) <= np.finfo(float).eps * scale**2 * 32:
                continue
            triangles.append(tri)
    if not edges or any(n != 2 for n in edges.values()):
        raise ValueError('Cell surface must be closed with two faces per edge')
    indices = np.asarray(triangles, dtype=int)
    t = v[indices] - q
    volumes = np.linalg.det(t) / 6
    volume = volumes.sum()
    if volume <= np.finfo(float).eps * scale**3:
        raise ValueError('Cell has zero or numerically unresolved volume')
    sums = t.sum(axis=1)
    first = np.einsum('t,ti->i', volumes, sums) / 4
    second = (np.einsum('t,ti,tj->ij', volumes, sums, sums)
              + np.einsum('t,tki,tkj->ij', volumes, t, t)) / 20
    centroid = q + first / volume
    central = second - np.outer(first, first) / volume
    origin = centroid if reference is None else np.asarray(reference, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Reference must be a finite 3D point')
    delta = centroid-origin
    tensor = central + volume * np.outer(delta, delta)
    tensor = (tensor+tensor.T)/2
    if np.linalg.eigvalsh(tensor)[0] <= 0:
        raise ValueError('Cell moment tensor is not positive definite')
    return float(volume), centroid, tensor, indices


def fabric_statistics(tensors, gap_tolerance=1e-6):
    """Thesis equations 3.48–3.54, with beta retained for legacy comparison.

    The all-cell orientation estimator includes ambiguous eigendirections for
    reproducibility. A second, explicitly named estimator excludes them.
    """
    w = np.asarray(tensors, dtype=float)
    if w.ndim != 3 or w.shape[1:] != (3, 3) or len(w) == 0 or not np.isfinite(w).all():
        raise ValueError('Expected a nonempty finite (N,3,3) tensor array')
    if not np.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError('Eigenvalue gap tolerance must be finite and nonnegative')
    if not np.allclose(w, w.swapaxes(1, 2), rtol=1e-12, atol=0):
        raise ValueError('Moment tensors must be symmetric')
    eigenvalues, eigenvectors = np.linalg.eigh(w)
    if np.any(eigenvalues[:, 0] <= 0):
        raise ValueError('Moment tensors must be positive definite')
    axes = eigenvectors[:, :, -1].copy()
    # Stable sign for exports; dyadic fabric is invariant under this choice.
    largest_component = np.argmax(np.abs(axes), axis=1)
    axes *= np.where(axes[np.arange(len(axes)), largest_component] < 0, -1, 1)[:, None]
    beta = eigenvalues[:, 0] / eigenvalues[:, -1]
    gap = (eigenvalues[:, -1]-eigenvalues[:, -2])/eigenvalues[:, -1]
    valid = gap > gap_tolerance

    def orientation(a):
        if len(a) == 0:
            return {'V': None, 'D': None, 'Ad': None, 'count': 0}
        fabric = a.T @ a / len(a)
        deviator = 7.5 * (fabric-np.eye(3)/3)
        return {'V': fabric.tolist(), 'D': deviator.tolist(),
                'Ad': float(np.sqrt(1.5*np.sum(deviator**2))), 'count': len(a)}

    summary = {'count': len(w), 'Ed': float(np.mean(1-beta)),
               'mean_beta': float(beta.mean()), 'ambiguous_axes': int((~valid).sum()),
               'axis_gap_tolerance': gap_tolerance,
               'orientation_all_cells': orientation(axes),
               'orientation_resolved_axes': orientation(axes[valid]),
               'Ad_convention': 'sqrt(3/2 * D:D), D=15/2*(V-I/3); aligned limit 7.5',
               'orientation_note': 'All-cell axes are arbitrary when the largest eigenvalue is degenerate.'}
    return summary, eigenvalues, axes, beta, gap, valid


def load_particles(path, uniform_radius=None):
    """Header optional: id x y z radius, or id x y z plus explicit radius."""
    path = Path(path)
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    first = next((line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')), '')
    header = first.split() in [['id', 'x', 'y', 'z', 'radius'], ['id', 'x', 'y', 'z']]
    header_line = next((i for i, line in enumerate(lines) if line.strip() == first), 0)
    data = np.loadtxt(path, skiprows=header_line+1 if header else 0, ndmin=2)
    if data.shape[1] not in (4, 5) or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError('Particle table must contain finite id x y z [radius] rows')
    ids = data[:, 0].astype(np.int64)
    if not np.array_equal(ids, data[:, 0]) or len(np.unique(ids)) != len(ids):
        raise ValueError('Particle IDs must be unique integers')
    if data.shape[1] == 4:
        if uniform_radius is None:
            raise ValueError('Four-column positions have no radii: provide --uniform-radius explicitly')
        radii = np.full(len(data), uniform_radius)
    else:
        if uniform_radius is not None:
            raise ValueError('Do not override radii already present in the particle file')
        radii = data[:, 4]
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Radii must be finite and positive')
    return ids, data[:, 1:4], radii


def analyze(ids, positions, radii, config, reference='particle', gap_tolerance=1e-6):
    """Compute every cell and fabric. Missing/invalid cells fail the run."""
    try:
        import pyvoro2 as pv
    except ImportError as error:
        raise ImportError('Install Minkowski3D/requirements.txt to enable tessellation') from error
    ids = np.asarray(ids)
    positions, radii = np.asarray(positions, dtype=float), np.asarray(radii, dtype=float)
    if (positions.shape != (len(ids), 3) or radii.shape != (len(ids),) or len(ids) == 0
            or not np.isfinite(positions).all() or not np.isfinite(radii).all() or np.any(radii <= 0)
            or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Supply unique integer IDs, finite positions, and positive radii')
    if reference not in ('particle', 'centroid'):
        raise ValueError('Reference must be particle or centroid')
    if not isinstance(config.get('units'), str) or not config['units'].strip():
        raise ValueError('Domain configuration must explicitly specify coordinate units')
    if 'lattice_vectors' in config:
        if 'bounds' in config or 'periodic' in config:
            raise ValueError('Lattice configuration is fully periodic; omit bounds/periodic')
        domain = pv.PeriodicCell(tuple(map(tuple, config['lattice_vectors'])),
                                 origin=tuple(config.get('origin', [0, 0, 0])))
        domain_volume = abs(np.linalg.det(config['lattice_vectors']))
    else:
        periodic = config.get('periodic')
        if not isinstance(periodic, list) or len(periodic) != 3 or any(type(x) is not bool for x in periodic):
            raise ValueError('Box configuration requires three explicit periodic booleans')
        domain = pv.OrthorhombicCell(tuple(map(tuple, config['bounds'])), periodic=tuple(periodic))
        domain_volume = np.prod(np.diff(config['bounds'], axis=1))
    mode = config.get('mode', 'power')
    if mode not in ('power', 'standard'):
        raise ValueError('Mode must be power or standard')
    result = pv.compute(positions, ids=ids, domain=domain, mode=mode,
                        **({'weights': radii**2} if mode == 'power' else {}))
    by_id = {c['id']: c for c in result.cells}
    if len(by_id) != len(ids) or set(by_id) != set(ids.tolist()):
        raise ValueError('Tessellation omitted cells; check sites and power weights')
    cells, tensors, volumes, centroids, sites = [], [], [], [], []
    for pid, radius in zip(ids, radii):
        cell = by_id[int(pid)]
        # pyvoro2 returns each complete cell in the continuous frame of its site.
        site = np.asarray(cell['site'])
        vertices = np.asarray(cell['vertices'])
        volume, centroid, tensor, triangles = polyhedron_moments(
            vertices, [f['vertices'] for f in cell['faces']], site if reference == 'particle' else None)
        if not np.isclose(volume, cell['volume'], rtol=1e-7, atol=1e-12*domain_volume):
            raise ValueError(f'Cell {pid}: integrated volume disagrees with Voro++')
        cells.append({'id': int(pid), 'vertices': vertices, 'triangles': triangles})
        volumes.append(volume); centroids.append(centroid); tensors.append(tensor); sites.append(site)
    closure = abs(sum(volumes)-domain_volume)/domain_volume
    if closure > 1e-7:
        raise ValueError(f'Cell volumes do not fill domain: relative error {closure:g}')
    tensors = np.asarray(tensors)
    normalized = tensors / radii[:, None, None]**5
    summary, eigenvalues, axes, beta, gap, valid = fabric_statistics(normalized, gap_tolerance)
    summary.update({'reference': reference, 'tensor_definition': 'volume second moment of full cell; not pore-only',
                    'coordinate_units': config['units'], 'physical_tensor_units': f"({config['units']})^5",
                    'normalized_tensor_units': 'dimensionless', 'normalization': 'W020 / radius^5',
                    'domain': config, 'domain_volume': float(domain_volume), 'cell_volume_sum': float(sum(volumes)),
                    'relative_volume_closure_error': float(closure),
                    'backend': 'pyvoro2', 'backend_version': getattr(pv, '__version__', 'unknown'),
                    'python_version': platform.python_version(), 'numpy_version': np.__version__})
    return dict(ids=ids, positions=positions, sites=np.asarray(sites), radii=radii,
                volumes=np.asarray(volumes), centroids=np.asarray(centroids), tensors=tensors,
                normalized_tensors=normalized, eigenvalues=eigenvalues, axes=axes, beta=beta,
                ed=1-beta, axis_gap=gap, resolved_axis=valid, cells=cells, summary=summary)


def save_results(result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/'cells.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id','x','y','z','radius','volume','cx','cy','cz','beta','ed',
                         'lambda_min_normalized','lambda_mid_normalized','lambda_max_normalized',
                         'vx','vy','vz','axis_gap','resolved_axis'])
        for i, pid in enumerate(result['ids']):
            writer.writerow([int(pid), *result['sites'][i], result['radii'][i], result['volumes'][i],
                             *result['centroids'][i], result['beta'][i], result['ed'][i],
                             *result['eigenvalues'][i], *result['axes'][i], result['axis_gap'][i],
                             int(result['resolved_axis'][i])])
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output/'tensors.npz', **arrays,
                        metadata_json=json.dumps(result['summary'], allow_nan=False))
    (output/'fabric.json').write_text(json.dumps(result['summary'], indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savetxt(output/'principal_axes.txt', np.column_stack([result['axes'], np.ones(len(result['axes']))]),
               header='nx ny nz f', comments='')


def write_preview(result, path, max_particles=200):
    """Self-contained offline HTML; geometry is display-sampled, statistics are not."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    summary = result['summary']; n = len(result['ids'])
    selected = np.linspace(0, n-1, min(n, max_particles), dtype=int)
    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scene'}, {'type':'scene'}],
                         [{'type':'scene'}, {'type':'xy'}]], subplot_titles=[
                         f'Packing ({len(selected)} spheres; {n} centers)',
                         'Most elongated cell: geometry and principal axis',
                         'Principal axes (antipodal; all cells)', 'Local elongation (all cells)'])
    # One batched sphere mesh; radii and positions retain coordinate units.
    from scipy.spatial import ConvexHull
    phi = np.linspace(0, 2*np.pi, 14, endpoint=False)
    mu = np.linspace(-.95, .95, 8)
    pp, mm = np.meshgrid(phi, mu)
    sphere = np.column_stack([(np.sqrt(1-mm**2)*np.cos(pp)).ravel(),
                             (np.sqrt(1-mm**2)*np.sin(pp)).ravel(), mm.ravel()])
    sphere = np.vstack([sphere, [0,0,1], [0,0,-1]])
    sf = ConvexHull(sphere).simplices
    vertices = np.concatenate([sphere*result['radii'][i]+result['sites'][i] for i in selected])
    triangles = np.concatenate([sf+j*len(sphere) for j in range(len(selected))])
    fig.add_trace(go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                  i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
                  intensity=np.repeat(result['ed'][selected], len(sphere)),
                  colorscale='Viridis', cmin=0, cmax=1, colorbar=dict(title='ed',len=.4,y=.8),
                  name='Spheres', hoverinfo='skip'), row=1, col=1)
    if n > max_particles:
        p = result['sites']
        fig.add_trace(go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers',
                      marker=dict(size=1, color='#597180', opacity=.25),
                      name='All particle centers', hoverinfo='skip'), row=1, col=1)
    config = summary['domain']
    if 'lattice_vectors' in config:
        lattice = np.asarray(config['lattice_vectors'])
        box_origin = np.asarray(config.get('origin',[0,0,0]))
    else:
        bounds = np.asarray(config['bounds'])
        lattice = np.diag(bounds[:,1]-bounds[:,0]); box_origin = bounds[:,0]
    corners = np.indices((2,2,2)).reshape(3,-1).T
    lines = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j > i and np.sum(abs(a-b)) == 1:
                lines.extend([a@lattice+box_origin,b@lattice+box_origin,[None,None,None]])
    xyz = np.asarray(lines,dtype=object)
    fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',
                  line=dict(color='#243b53',width=3),hoverinfo='skip'),row=1,col=1)
    idx = int(np.argmax(result['ed'])); cell = result['cells'][idx]
    v, t = cell['vertices'], cell['triangles']
    fig.add_trace(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=t[:,0], j=t[:,1], k=t[:,2],
                           opacity=.25, color='#169c9c', name=f"Cell {cell['id']}"), row=1, col=2)
    site = result['sites'][idx]; radius = result['radii'][idx]
    s = sphere*radius+site
    fig.add_trace(go.Mesh3d(x=s[:,0], y=s[:,1], z=s[:,2], i=sf[:,0], j=sf[:,1], k=sf[:,2],
                           color='#e0a04b', opacity=.7, name='Particle'), row=1, col=2)
    origin = site if summary['reference'] == 'particle' else result['centroids'][idx]
    fig.add_trace(go.Scatter3d(x=[origin[0]],y=[origin[1]],z=[origin[2]],mode='markers',
                  marker=dict(size=5,color='#be4255'),name='Moment reference'),row=1,col=2)
    length = np.max(np.linalg.norm(v-origin, axis=1))
    ends = origin+np.array([-1,1])[:,None]*length*result['axes'][idx]
    fig.add_trace(go.Scatter3d(x=ends[:,0], y=ends[:,1], z=ends[:,2], mode='lines',
                             line=dict(width=7,color='#be4255'), name='Principal axis'), row=1, col=2)
    axes = np.concatenate([result['axes'], -result['axes']])
    fig.add_trace(go.Scatter3d(x=axes[:,0], y=axes[:,1], z=axes[:,2], mode='markers',
                             marker=dict(size=2,opacity=.35,color='#169c9c'), name='Axes'), row=2, col=1)
    fig.add_trace(go.Histogram(x=result['ed'], xbins=dict(start=0,end=1,size=.025),
                              marker_color='#169c9c', name='Cell count'), row=2, col=2)
    for key in ['scene','scene2','scene3']:
        fig.update_layout(**{key:dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z')})
    orientation = summary['orientation_all_cells']
    fig.update_layout(height=950, showlegend=False, template='plotly_white',
        title=dict(text=f"Minkowski3D · {n} cells · Ed={summary['Ed']:.4f} · Ad={orientation['Ad']:.4f}"
                   f"<br><sup>{summary.get('example_label','Particle environment fabric')} · reference: {summary['reference']}"
                   f" · ambiguous axes: {summary['ambiguous_axes']} · geometry view is sampled; statistics use all cells</sup>"))
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def run_particles(input_path, domain_path, output, reference='particle', uniform_radius=None, preview=True):
    ids, positions, radii = load_particles(input_path, uniform_radius)
    config = json.loads(Path(domain_path).read_text(encoding='utf-8'))
    result = analyze(ids, positions, radii, config, reference)
    result['summary']['input_sha256'] = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    result['summary']['example_label'] = config.get('description', Path(input_path).name)
    result['summary']['uniform_radius_override'] = uniform_radius
    save_results(result, output)
    if preview:
        write_preview(result, Path(output)/'preview.html')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--domain', type=Path, required=True, help='JSON units and domain geometry')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/minkowski'))
    parser.add_argument('--reference', choices=['particle','centroid'], default='particle')
    parser.add_argument('--uniform-radius', type=float, help='Explicit assumption for four-column files only')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    try:
        result = run_particles(args.input, args.domain, args.output_dir, args.reference,
                               args.uniform_radius, not args.no_preview)
    except (ValueError, ImportError, OSError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
