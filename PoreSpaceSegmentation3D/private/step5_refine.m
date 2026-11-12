% Adapted from the submitted parallel Step 5 by Srinivas Vivek Bokkisa.
% Framework: Nimisha Roy (MIT); see docs/MODIFICATIONS.md for integration changes.
function [throat_body_new, throat_center_new, locals_new, diagnostics] = ...
    step5_refine(throat_body, voxels, num_pore_body, throat_center, ste, locals, throat_body_surface, sig, ~, ~, useParallel)

% NOTE:
%   This version reduces "Out of Memory during deserialization" by avoiding
%   broadcasting large arrays into the parfor loop. Large shared inputs are
%   stored as parallel.pool.Constant and accessed via .Value inside workers.

maxWorkers = 0;
if useParallel, maxWorkers = Inf; end
makeConstant = @(x) struct('Value', {x});
if useParallel, makeConstant = @(x) parallel.pool.Constant(x); end

indices = unique(nonzeros(throat_body));
throat_body_ref = throat_body;

Nthroat = size(throat_center,1);

if numel(indices) < (num_pore_body + Nthroat)
    error('Mismatch: indices does not contain enough throat labels for throat_center rows.');
end
throat_lbl_by_x1 = indices(num_pore_body + (1:Nthroat));

% Output init (write only outside parfor)
throat_center_new = zeros(Nthroat,8);
throat_center_new(:,1:6) = throat_center;

% PASS-1 per-throat results (sliced OK)
proposals      = cell(Nthroat, 1);
proposal_valid = false(Nthroat, 1);
errors = cell(Nthroat, 1);

tc7_vec = nan(Nthroat,1);   % throat_center_new(:,7)
tc8_vec = nan(Nthroat,1);   % throat_center_new(:,8)

% ------------------- PRECOMPUTE label -> voxel indices -------------------
labels = throat_body_ref(:);
mask   = labels > 0;
linIdx = find(mask);
labelVals = labels(mask);

maxLabel = max(labelVals);
voxIdxByLabel = accumarray(labelVals, linIdx, [maxLabel 1], @(v){v});

% ------------------- PRECOMPUTE throat_label -> surface point cloud -------------------
surfaceByLabel = cell(maxLabel, 1);
if ~isempty(throat_body_surface)
    tl = throat_body_surface(:,4);
    ok = tl > 0 & tl <= maxLabel;
    tbs = throat_body_surface(ok,:);
    tl  = tbs(:,4);

    uL = unique(tl(:)).';
    for L = uL
        surfaceByLabel{L} = tbs(tl == L, 1:3);
    end
end

% =========================================================================
%  Wrap large read-only data in parallel.pool.Constant (reduces deserialization)
% =========================================================================
tbConst      = makeConstant(throat_body_ref);
voxConst     = makeConstant(voxels);
localsConst  = makeConstant(locals);
tcConst      = makeConstant(throat_center);
idxConst     = makeConstant(indices);
lblVecConst  = makeConstant(throat_lbl_by_x1);

voxIdxConst  = makeConstant(voxIdxByLabel);
surfByConst  = makeConstant(surfaceByLabel);

% ------------------- PASS 1: PARALLEL -------------------
parfor (x1 = 1:Nthroat, maxWorkers)

    % Pull constant values into worker-local variables
    throat_body_ref_w = tbConst.Value;
    voxels_w          = voxConst.Value;
    locals_w          = localsConst.Value;
    throat_center_w   = tcConst.Value;
    indices_w         = idxConst.Value;
    throat_lbl_by_x1_w= lblVecConst.Value;

    voxIdxByLabel_w   = voxIdxConst.Value;
    surfaceByLabel_w  = surfByConst.Value;

    thrLbl = throat_lbl_by_x1_w(x1);
    if thrLbl <= 0 || thrLbl > maxLabel
        continue;
    end

    locations = voxIdxByLabel_w{thrLbl};
    if isempty(locations)
        continue;
    end

    b1Lbl = throat_center_w(x1,5);
    b2Lbl = throat_center_w(x1,6);

    if b1Lbl <= 0 || b2Lbl <= 0 || b1Lbl > maxLabel || b2Lbl > maxLabel
        continue;
    end

    locations_body1 = voxIdxByLabel_w{b1Lbl};
    locations_body2 = voxIdxByLabel_w{b2Lbl};
    if isempty(locations_body1) || isempty(locations_body2)
        continue;
    end

    set_throat = voxels_w(locations, :);
    set_body1  = voxels_w(locations_body1, :);
    set_body2  = voxels_w(locations_body2, :);

    set2 = surfaceByLabel_w{thrLbl};
    if ~isempty(set2)
        % keep only surface points that belong to the throat voxel set
        % (this can be expensive; OK for now, but could be optimized)
        set2 = set2(ismember(set2, set_throat, 'rows'), :);
    end

    vis_local = 0;

    try
        % Step 5b
        [mid_profile, surface_profile, unit_norm, body1_profile, body2_profile] = ...
            step5b_bestfit(set_body1, set_body2, set2, ste, x1, set_throat, ...
                           voxels_w, locations, locals_w, throat_center_w, vis_local);

        % Step 5c robust
        [mid_uv, body_uv, throat_uv, surf_uv, b1_uv, b2_uv, T] = ...
            step5c_bestfit2D_robust(locals_w, throat_center_w, unit_norm, x1, voxels_w, ...
                                    mid_profile, surface_profile, body1_profile, body2_profile, vis_local);

        if size(mid_uv,1) <= 10
            continue;
        end

        % Step 5d boundary
        store_new = step5d_boundary(mid_uv, sig, throat_uv, ste, vis_local);

        % Step 5e/f throat filter
        thint = (round(pi/360 * 10^(sig))) / 10^(sig); % 0.5 degrees

        twr_local = [];
        tc_local  = zeros(1,8);
        tc_local(1,1:6) = throat_center_w(x1,1:6);

        [store_new, twr_local, tc_local] = ...
            step5ef_filter(1, store_new, thint, twr_local, tc_local, throat_uv, vis_local);

        tc7_raw = tc_local(1,7);

        % Throat knee points
        [firstqti,secondqti,thirdqti,fourthqti, ...
         firstqtall,secondqtall,thirdqtall,fourthqtall, throat_axis_center] = ...
            step5g_inflexion(store_new, surf_uv, vis_local);

        % Combined boundary + filtering
        combined_profile = [mid_uv; b1_uv; b2_uv];
        store_new_combined = step5ef_filtered_combined(combined_profile, ste, sig, thint, vis_local);

        sizy = max(store_new(:,2)) - min(store_new(:,2));
        sizx = max(store_new(:,1)) - min(store_new(:,1));

        body1_center_uv = mean(b1_uv(:,1:2), 1);
        body2_center_uv = mean(b2_uv(:,1:2), 1);

        [firstqci,secondqci,thirdqci,fourthqci, ...
         firstqcall,secondqcall,thirdqcall,fourthqcall] = ...
            step5g_inflexion_combined_enhanced(throat_axis_center, store_new_combined, ...
                                               throat_uv, body1_center_uv, body2_center_uv, vis_local);

        % Correct selection
        [p1,p2,p3,p4,~] = ...
            step5gh_correct_inflexion_robust(sizy, sizx, firstqci, secondqci, thirdqci, fourthqci, ...
                                             firstqtall, secondqtall, thirdqtall, fourthqtall, ...
                                             firstqti, secondqti, thirdqti, fourthqti, ...
                                             firstqcall, secondqcall, thirdqcall, fourthqcall, ...
                                             store_new, throat_axis_center, vis_local);

        points = [p1; p2; p3; p4];

        % Normalize tc7 by L; compute LSI -> store in tc7_vec / tc8_vec
        L = max(max(pdist2(points(:,1:2), points(:,1:2), 'euclidean')));
        if isfinite(L) && L > 0
            tc7_vec(x1) = tc7_raw * 100 / L;
        else
            tc7_vec(x1) = NaN;
        end

        [ang1, ang2] = step5h_lsi(store_new, store_new_combined, points, vis_local);
        tc8_vec(x1)  = 0.5 * (ang1 + ang2);

        % Back-project to 3D (new)
        [points3d, combined_profile_3d] = ...
            step5i_inflexion3d_fromT(points(:,1:2), points(:,3), T, ...
                                     mid_profile, body1_profile, body2_profile);

        center3 = mean(combined_profile_3d, 1);
        body1_center = mean(body1_profile(:,1:3), 1);
        body2_center = mean(body2_profile(:,1:3), 1);

        % Proposal for final relabeling
        % NOTE: proposal currently receives the full volume (throat_body_ref_w).
        % That is now sourced from a Constant to avoid broadcast deserialization blow-up.
        p = step5i_correct_division3d_proposal( ...
            x1, throat_body_ref_w, indices_w, (num_pore_body + x1), throat_center_w, body_uv, ...
            points3d, body1_center, body2_center, center3, unit_norm, combined_profile_3d);

        proposals{x1} = p;
        proposal_valid(x1) = ~isempty(p) && isfield(p,'valid') && p.valid;

    catch exception
        errors{x1} = sprintf('%s: %s', exception.identifier, exception.message);
        continue;
    end
end

% Write sliced results into throat_center_new outside parfor
throat_center_new(:,7) = tc7_vec;
throat_center_new(:,8) = tc8_vec;

% PASS 2: Apply proposals once
throat_body_new = apply_refinement_proposals(throat_body_ref, proposals, proposal_valid);

diagnostics = struct('total_throats', Nthroat, ...
    'accepted_proposals', nnz(proposal_valid), 'errors', {errors}, ...
    'merge_policy', 'ascending throat row; later proposal wins');

% Update throat centers (after apply)
for x1 = 1:Nthroat
    tLbl = throat_lbl_by_x1(x1);
    if tLbl <= 0, continue; end
    ind = find(throat_body_new == tLbl);
    if ~isempty(ind)
        throat_center_new(x1,1:3) = mean(voxels(ind,1:3), 1);

        B = voxels(ind,1:3);
        distances = sqrt(sum((B - mean(B,1)).^2, 2));
        throat_center_new(x1,4) = ind(find(distances == min(distances), 1, 'first'));
    end
end

% Update body centers
locals_new = zeros(num_pore_body,1);
indices_b = unique(nonzeros(throat_body));
for index = 1:num_pore_body
    locations_body = find(throat_body_new == indices_b(index));
    if ~isempty(locations_body)
        B = voxels(locations_body,1:3);
        distances = sqrt(sum(bsxfun(@minus, B, mean(B,1)).^2, 2));
        locals_new(index) = locations_body(find(distances==min(distances),1,'first'));
    else
        locals_new(index) = 0;
    end
end

end


