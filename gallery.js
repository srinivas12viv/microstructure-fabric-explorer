'use strict';
const byId = id => document.getElementById(id);
function updatePolar() {
  const dataset = byId('polar-dataset').value;
  const plane = byId('polar-plane').value;
  const base = `outputs/polar-${dataset}-${plane}`;
  byId('polar-frame').src = `${base}.html`;
  byId('polar-open').href = `${base}.html`;
  byId('polar-csv').href = `${base}.csv`;
  const stats = window.MFE_DATA?.[dataset === 'iso' ? 'Iso' : 'Aniso']?.planes[plane];
  if (stats) byId('polar-caption').textContent = `${stats.observations.toLocaleString()} antipodal observations · Harmonic amplitude ${stats.amplitude.toFixed(3)} · Preferred axis ${stats.direction.toFixed(1)}° · ${stats.excluded} excluded projections. Color: mean force in input units.`;
}
function updateSphere() {
  const dataset = byId('sphere-dataset').value;
  const base = `outputs/sphere-${dataset}`;
  byId('sphere-frame').src = `${base}.html`;
  byId('sphere-open').href = `${base}.html`;
  byId('sphere-csv').href = `${base}.csv`;
  byId('sphere-png').href = `${base}.png`;
  const stats = window.MFE_DATA?.[dataset === 'iso' ? 'Iso' : 'Aniso'];
  if (stats) byId('sphere-caption').textContent = `${stats.contacts.toLocaleString()} contacts · ${stats.spherical_observations.toLocaleString()} antipodal observations · 648 equal-solid-angle bins. Radius and color: count.`;
}
byId('polar-dataset').addEventListener('change', updatePolar);
byId('polar-plane').addEventListener('change', updatePolar);
byId('sphere-dataset').addEventListener('change', updateSphere);
byId('copy-command').addEventListener('click', async () => {
  const commands = byId('quickstart').textContent;
  try {
    await navigator.clipboard.writeText(commands);
    byId('copy-status').textContent = 'Commands copied.';
  } catch {
    byId('copy-status').textContent = 'Select and copy the commands above; clipboard access is unavailable.';
  }
});
const force = window.MFE_DATA?.force;
if (force) byId('force-caption').textContent = `${force.particles.toLocaleString()} particles · ${force.plotted.toLocaleString()} contacts plotted · ${force.excluded} contacts excluded by the 0.1 component-gap filter. Force units are inherited from the input.`;
updatePolar();
updateSphere();

const contacts = window.MFE_DATA?.contacts3d;
if (contacts) byId('contact-summary').textContent = `${contacts.n_particles.toLocaleString()} particles; ${contacts.n_contacts.toLocaleString()} contacts; mean coordination ${contacts.mean_coordination_all.toFixed(2)}.`;

const contactViews = {
  volume: {
    file: 'segmented-volume.html', png: 'segmented-volume.png',
    caption: 'Original segmented TIFF: colors distinguish particle labels. Drag to rotate and zoom. Surface display samples every third voxel; contact detection uses the full image.'
  },
  patch: {
    file: 'contact-voxels.png',
    caption: 'Contact voxel patch (retained contact index 0): blue and green show the neighboring particle surfaces; red highlights the contact voxels within the dilation shell.'
  },
  normals: {
    file: 'centroids-normals.png',
    caption: 'Contact centroids and PCA normals: points locate contacts; symmetric line segments show the axial normal directions used to compute fabric. Normal signs are arbitrary.'
  },
  coordination: {
    file: 'coordination.png',
    caption: 'Coordination number distribution: the number of retained neighbors per particle. Counts include isolated particles with zero contacts.'
  }
};
function updateContact() {
  const key = byId('contact-view').value;
  const view = contactViews[key];
  const base = 'outputs/contact-detection/';
  byId('contact-frame').hidden = key !== 'volume';
  byId('contact-image').hidden = key === 'volume';
  if (key !== 'volume') {
    byId('contact-image').src = base + view.file;
    byId('contact-image').alt = view.caption;
  }
  byId('contact-open').href = base + view.file;
  byId('contact-save').href = base + (view.png || view.file);
  byId('contact-caption').textContent = view.caption;
}
byId('contact-view').addEventListener('change', updateContact);
updateContact();

const minkowskiViews = {
  cell: {file: 'cell.png', caption: 'Actual sample cell: gold shows the particle, teal shows its Voronoi boundary and moment-equivalent ellipsoid, and red marks the principal axis. Coordinates are offsets from the cell centroid.'},
  packing: {file: 'packing.png', caption: 'All 10,000 particle centers from the sample, colored by radius in mm. The outline shows the sheared periodic domain. Markers represent centers, not rendered sphere surfaces.'},
  fabric: {file: 'fabric.png', caption: 'Elongation distribution over all 10,000 cells. The gold line marks mean elongation Ed; Ad summarizes alignment of the principal axes.'},
  workflow: {file: 'workflow.png', caption: 'The complete computation: spherical particle geometry, a weighted Voronoi cell and its moment ellipsoid, then assembly fabric statistics. Every panel comes from the same sample computation.'},
  interactive: {file: 'preview.html', caption: 'Drag to rotate the three-dimensional views. The packing shows 200 sphere meshes and all 10,000 centers; fabric statistics include all cells.'}
};
function updateMinkowski() {
  const key = byId('minkowski-view').value;
  const view = minkowskiViews[key];
  const interactive = key === 'interactive';
  const base = 'outputs/minkowski/';
  byId('minkowski-image').hidden = interactive;
  byId('minkowski-frame').hidden = !interactive;
  if (!interactive) {
    byId('minkowski-image').src = base + view.file;
    byId('minkowski-image').alt = view.caption;
  }
  byId('minkowski-open').href = base + view.file;
  byId('minkowski-save').href = base + (interactive ? 'workflow.png' : view.file);
  byId('minkowski-caption').textContent = view.caption;
}
byId('minkowski-view').addEventListener('change', updateMinkowski);
updateMinkowski();
const minkowski = window.MFE_DATA?.minkowski;
if (minkowski) {
  byId('minkowski-count').textContent = minkowski.count.toLocaleString();
  byId('minkowski-ed').textContent = minkowski.Ed.toFixed(4);
  byId('minkowski-ad').textContent = minkowski.orientation_all_cells.Ad.toFixed(4);
}
