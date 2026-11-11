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
  const commands = 'python -m pip install -r requirements.txt\npython build_gallery.py\npython ContactDetection3D/main.py\npython PolarHistogram2D/PolarHistogram.py --plane xz --output outputs/my-polar.html';
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
