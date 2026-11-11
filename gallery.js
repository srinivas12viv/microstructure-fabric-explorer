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

const contactViews = {
  patch: ['typicalContact.svg', 'Supplied contact-patch example: particle surfaces in blue and green; detected contact voxels in red.'],
  normals: ['centroids_normals.svg', 'Supplied 3D distribution of contact centroids and axial PCA normals.'],
  coordination: ['histogram.svg', 'Supplied coordination distribution. The original figure reports mean Z = 4.80; see the documentation for averaging conventions.']
};
byId('contact-view').addEventListener('change', () => {
  const [file, caption] = contactViews[byId('contact-view').value];
  byId('contact-image').src = `ContactDetection3D/out/${file}`;
  byId('contact-image').alt = caption;
  byId('contact-open').href = `ContactDetection3D/out/${file}`;
  byId('contact-caption').textContent = caption;
});
