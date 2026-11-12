% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [throat_center, neighbour, constriction_length1] = step3_throatcenters( ...
    separator, voxels, ste, locals, num_pore_bodies, neighbour)
% This function finds the individual constriction surfaces, their length
% and throat center on the constriction surfaces. This is part of Steps 2
% and 3 of the algorithm.
%
% Optimized version:
% - Precomputes voxel indices per pore label
% - Builds one KDTree per pore volume and reuses it
% - Uses vectorized distance formulas instead of repeated pdist2
% - Preallocates throat_center and constriction_length1
%
% ------ Original Author: Nimisha Roy
% e-mail: nroy9@gatech.edu
% ------

% -------------------------------------------------------------------------
% Preprocessing: label -> voxel indices, KD-trees, pore centers
% -------------------------------------------------------------------------
labels  = separator(:);
mask    = labels > 0;
linIdx  = find(mask);
labelVals = labels(mask);

if isempty(labelVals)
    throat_center        = [];
    constriction_length1 = [];
    neighbour            = [neighbour zeros(size(neighbour,1),1)];
    neighbour            = neighbour([],:); % no throats
    return;
end

% Unique nonzero labels (pore volumes)
indices   = unique(labelVals);           % same role as original "indices"
numLabels = max(indices);                % assumes labels are sequential 1..num_pore

% bodyVox{L} = linear indices of voxels with label L in 'separator'
bodyVox = accumarray(labelVals, linIdx, [numLabels 1], @(v){v});

% Precompute pore body centers: centerByIndex(k,:) corresponds to pore "k"
centerByIndex = zeros(numLabels, 3);
for L = 1:numLabels
    centerByIndex(L, :) = voxels(locals(L), :);
end

% Build one KDTree per pore volume (by label), reuse inside the loop
kdt = cell(numLabels, 1);
for L = 1:numLabels
    voxIdx = bodyVox{L};
    if ~isempty(voxIdx)
        kdt{L} = KDTreeSearcher(voxels(voxIdx, :));
    else
        kdt{L} = [];
    end
end

% -------------------------------------------------------------------------
% Main loop over neighbour pairs
% -------------------------------------------------------------------------
numPairs = size(neighbour, 1);

% Add a 3rd column for throat labels (as in original code)
neighbour = [neighbour zeros(numPairs,1)];

% Preallocate throat_center and constriction_length1
% each row of throat_center: [cx cy cz surface_voxel_idx body_label1 body_label2] -> 6 columns
throat_center_tmp        = zeros(numPairs, 6);
constriction_length_tmp  = zeros(numPairs, 1);
tCount = 0;  % number of accepted throats

sqrt3ste = sqrt(3) * ste;   % used in contact criterion

for i = 1:numPairs
    index = neighbour(i, 1);   % pore index 1 (1..num_pore_body)
    j     = neighbour(i, 2);   % pore index 2

    % Voxel indices for these two pore bodies (labels in separator)
    label1 = indices(index);
    label2 = indices(j);

    loc1 = bodyVox{label1};
    loc2 = bodyVox{label2};

    if isempty(loc1) || isempty(loc2)
        neighbour(i,3) = 0;
        continue;
    end

    % Coordinates of voxels in each pore
    coords1 = voxels(loc1, :);
    coords2 = voxels(loc2, :);

    % ---------------------------------------------------------------------
    % Find facing surface voxels via KD-tree (same logic as original)
    % ---------------------------------------------------------------------
    kdt_1 = kdt{label1};
    if isempty(kdt_1)
        neighbour(i,3) = 0;
        continue;
    end

    % nearest in pore 1 to each voxel in pore 2
    [nearestIdx, val1] = knnsearch(kdt_1, coords2, 'K', 1);

    mask_close          = (val1 <= sqrt3ste);
    surfaces_ind2       = unique(loc2(mask_close));           % global indices in pore 2
    surfaces_ind1_local = nearestIdx(mask_close);             % indices in coords1
    surfaces_ind1       = unique(loc1(surfaces_ind1_local));  % global indices in pore 1

    surfaces1 = voxels(surfaces_ind1, :);
    surfaces2 = voxels(surfaces_ind2, :);

    if isempty(surfaces1) && isempty(surfaces2)
        % Not neighbours in practice
        neighbour(i,3) = 0;
        continue;
    end

    % Pore body centers in 3D
    c1 = centerByIndex(index, :);  % center of pore "index"
    c2 = centerByIndex(j, :);      % center of pore "j"

    % ---------------------------------------------------------------------
    % Find constriction centers in each surface set (same logic as original)
    % ---------------------------------------------------------------------
    haveCenter1 = false;
    haveCenter2 = false;

    % ---- surfaces1 -> center1 -------------------------------------------
    if ~isempty(surfaces1)
        % Sum of distances to the two pore centers
        d1 = sqrt(sum((surfaces1 - c1).^2, 2));
        d2 = sqrt(sum((surfaces1 - c2).^2, 2));
        dist_sum1 = round(d1 + d2, 3);   % roundn(...,-3) -> round(...,3)

        minDist_sum1 = min(dist_sum1);
        ind_sum1     = find(dist_sum1 == minDist_sum1);   % indices in surfaces1

        pts1   = surfaces1(ind_sum1, :);                  % candidates
        d1p    = sqrt(sum((pts1 - c1).^2, 2));
        d2p    = sqrt(sum((pts1 - c2).^2, 2));
        dist_diff1 = round(abs(d1p - d2p), 3);            % roundn(...,-3)

        [minDist_diff1, ind_diff1] = min(dist_diff1); %#ok<ASGLU>
        ind_diff3 = find(dist_diff1 == minDist_diff1);

        if numel(ind_diff3) == 1
            center1 = surfaces1(ind_sum1(ind_diff1), :);
        else
            center1 = surfaces1(ind_sum1(ind_diff1(1)), :);
        end

        haveCenter1 = true;
    end

    % ---- surfaces2 -> center2 -------------------------------------------
    if ~isempty(surfaces2)
        d1_2 = sqrt(sum((surfaces2 - c1).^2, 2));
        d2_2 = sqrt(sum((surfaces2 - c2).^2, 2));
        dist_sum2 = round(d1_2 + d2_2, 3);

        minDist_sum2 = min(dist_sum2);
        ind_sum2     = find(dist_sum2 == minDist_sum2);

        pts2   = surfaces2(ind_sum2, :);
        d1p_2  = sqrt(sum((pts2 - c1).^2, 2));
        d2p_2  = sqrt(sum((pts2 - c2).^2, 2));
        dist_diff2 = round(abs(d1p_2 - d2p_2), 3);

        [minDist_diff2, ind_diff2] = min(dist_diff2); %#ok<ASGLU>
        ind_diff4 = find(dist_diff2 == minDist_diff2);

        if numel(ind_diff4) == 1
            center2 = surfaces2(ind_sum2(ind_diff2), :);
        else
            center2 = surfaces2(ind_sum2(ind_diff2(1)), :);
        end

        haveCenter2 = true;
    end

    % ---------------------------------------------------------------------
    % Choose final constriction center & surface set
    % (same decision tree as original, just with 'all' for equality)
    % ---------------------------------------------------------------------
    if haveCenter1 && haveCenter2
        if all(center1(:) == center2(:))
            center       = center1;
            surfaces     = surfaces1;
            surfaces_ind = surfaces_ind1;
            ind_sum      = ind_sum1;
            ind_diff     = ind_diff1;
        else
            % Compare symmetry wrt pore centers
            diff1 = abs( ...
                sqrt(sum((center1 - c1).^2, 2)) - ...
                sqrt(sum((center1 - c2).^2, 2)) );
            diff2 = abs( ...
                sqrt(sum((center2 - c1).^2, 2)) - ...
                sqrt(sum((center2 - c2).^2, 2)) );
            if diff1 < diff2
                center       = center1;
                surfaces     = surfaces1;
                surfaces_ind = surfaces_ind1;
                ind_sum      = ind_sum1;
                ind_diff     = ind_diff1;
            elseif diff1 > diff2
                center       = center2;
                surfaces     = surfaces2;
                surfaces_ind = surfaces_ind2;
                ind_sum      = ind_sum2;
                ind_diff     = ind_diff2;
            else
                center       = center1;
                surfaces     = surfaces1;
                surfaces_ind = surfaces_ind1;
                ind_sum      = ind_sum1;
                ind_diff     = ind_diff1;
            end
        end
    elseif haveCenter1
        center       = center1;
        surfaces     = surfaces1;
        surfaces_ind = surfaces_ind1;
        ind_sum      = ind_sum1;
        ind_diff     = ind_diff1;
    else
        center       = center2;
        surfaces     = surfaces2;
        surfaces_ind = surfaces_ind2;
        ind_sum      = ind_sum2;
        ind_diff     = ind_diff2;
    end

    % ---------------------------------------------------------------------
    % Constriction length + throat acceptance (same criteria as original)
    % ---------------------------------------------------------------------
    % Find the surface voxel farthest from 'center'
    d_center = sqrt(sum((surfaces - center).^2, 2));
    [~, idx_anchor] = max(d_center);
    anchor = surfaces(idx_anchor, :);

    % Constriction length = maximum distance from that anchor
    d_anchor = sqrt(sum((surfaces - anchor).^2, 2));
    length1  = max(d_anchor);

    % Distances from center to pore body centers
    dc1 = sqrt(sum((center - c1).^2, 2));
    dc2 = sqrt(sum((center - c2).^2, 2));

    if (length1 / 2 < min(dc1, dc2)) && (length1 > 5 * ste)
        % accept this throat
        tCount = tCount + 1;

        % [center(1:3), surface voxel idx, body_label1, body_label2]
        throat_center_tmp(tCount, :) = [ ...
            center, ...
            surfaces_ind(ind_sum(ind_diff)), ...
            indices(index), indices(j)];

        constriction_length_tmp(tCount) = length1;

        neighbour(i, 3) = num_pore_bodies + 1;
        num_pore_bodies = num_pore_bodies + 1;
    else
        neighbour(i, 3) = 0;
    end
end

% Trim preallocated arrays to actual number of throats
if tCount > 0
    throat_center        = throat_center_tmp(1:tCount, :);
    constriction_length1 = constriction_length_tmp(1:tCount, :);
else
    throat_center        = zeros(0, 6);
    constriction_length1 = zeros(0, 1);
end

% Keep only valid throats in neighbour list
neighbour = neighbour(neighbour(:,3) > 0, :);

end
