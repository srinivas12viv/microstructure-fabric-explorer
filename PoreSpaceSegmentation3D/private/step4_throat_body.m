% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [throat_body, throat_center, neighbour, locals] = step4_throat_body( ...
    locals, voxels, throat_center, vox3d, neighbour, num_pore_body)
% This function discretizes the pore structure into pore bodies and pore
% throats based on MEDA, updates body and throat centers along with
% their neighbours. It is step 4 of the algorithm.
%
% Optimized version:
% - Uses voxels(:,1:3) directly (no x_voxel, y_voxel, z_voxel copies)
% - Skips pdist2 calls on solid voxels
% - Uses accumarray to count voxels per peak (instead of find(...) in a loop)
%
% ------ Original Author: Nimisha Roy
% e-mail: nroy9@gatech.edu
% ------

% -------------------------------------------------------------------------
% Flatten voxel info and preallocate throat_body
% -------------------------------------------------------------------------
voxelInfo = reshape(vox3d, [], 1);                 % 0 = solid, 1 = pore
N_voxel   = size(voxels, 1);
throat_body = zeros(N_voxel, 1, 'uint32');         % labels 0,1,...,num_peaks

% Number of voxels to process per block (tune if needed)
npp = 20;
counter = 0:npp:N_voxel;
if counter(end) ~= N_voxel
    counter = [counter, N_voxel];
end

% -------------------------------------------------------------------------
% Construct peak_voxels: body centers (via locals) + throat centers (via 4th col)
% -------------------------------------------------------------------------
body_peaks   = voxels(locals, 1:3);                % Pore-body peak voxels
throat_peaks = voxels(throat_center(:,4), 1:3);    % Throat-center voxels
peak_voxels  = [body_peaks; throat_peaks];         % (#bodies + #throats) x 3

num_peaks = size(peak_voxels, 1);

% -------------------------------------------------------------------------
% Assign each pore voxel to nearest peak (blockwise to avoid huge pdist2)
% -------------------------------------------------------------------------
for j = 1:(numel(counter)-1)
    idx_start = counter(j)   + 1;
    idx_end   = counter(j+1);
    idx_range = idx_start:idx_end;

    % Portion of voxels and corresponding binary mask
    voxels_block = voxels(idx_range, 1:3);          % (block_size x 3)
    isPore       = voxelInfo(idx_range) ~= 0;       % logical, pore vs solid

    if any(isPore)
        % Only compute distances for pore voxels in this block
        voxels_pore_block = voxels_block(isPore, :);

        % Distance between peak voxels and this subset of pore voxels
        % peak_voxels: (num_peaks x 3), voxels_pore_block: (Np x 3)
        DIST = pdist2(peak_voxels, voxels_pore_block);   % (num_peaks x Np)

        % Each pore voxel -> index of nearest peak (1..num_peaks)
        [~, ind_min] = min(DIST, [], 1);                 % 1 x Np

        % Build label array for the whole block (including solids)
        labels_block = zeros(numel(idx_range), 1, 'uint32');
        labels_block(isPore) = uint32(ind_min(:));

        % Store into throat_body
        throat_body(idx_range) = labels_block;
    else
        % all solid in this block -> labels stay 0
        % (throat_body already initialized to 0)
    end
end

% -------------------------------------------------------------------------
% If some peak (body or throat) gets no voxel, remove it:
% - for bodies: remove from locals
% - for throats: remove from throat_center and neighbour
% -------------------------------------------------------------------------
% Count voxels per peak (ignore label 0)
nonzero_labels = throat_body(throat_body > 0);
if isempty(nonzero_labels)
    % No voxels were assigned at all -> nothing to remove
    return;
end

counts = accumarray(double(nonzero_labels), 1, [num_peaks, 1]);

% Which peaks are empty?
empty_peaks = find(counts == 0);

% Bodies are peaks 1..num_pore_body, throats are num_pore_body+1..num_peaks
empty_bodies  = empty_peaks(empty_peaks <= num_pore_body);
empty_throats = empty_peaks(empty_peaks  > num_pore_body);

% ----- Remove empty bodies from locals (keeping same logic as original) ---
if ~isempty(empty_bodies)
    % Set those entries to 0, then filter out zeros
    locals(empty_bodies, 1) = 0;
    locals = locals(locals(:,1) ~= 0, :);
end

% ----- Remove empty throats from throat_center and neighbour -------------
if ~isempty(empty_throats)
    % throat index in throat_center/neighbour is (peak_idx - num_pore_body)
    throat_rows = empty_throats - num_pore_body;

    % Set first column to 0 for those throat rows (as in original code),
    % then filter out rows with first column == 0
    throat_center(throat_rows, 1) = 0;
    neighbour(throat_rows, 1)     = 0;

    throat_center = throat_center(throat_center(:,1) ~= 0, :);
    neighbour     = neighbour(neighbour(:,1) ~= 0, :);
end

end
