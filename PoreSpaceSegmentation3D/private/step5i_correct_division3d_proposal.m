% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function proposal = step5i_correct_division3d_proposal( ...
    x1, throat_body_ref, indices, index, throat_center, body_center_xy1, ...
    points1, body1_center, body2_center, center, unit_norm, combined_profile_i)
% step5i_correct_division3d_proposal
% -------------------------------------------------------------------------
% Two-pass compatible version of step5i_correct_division3d:
%   PASS 1: Compute a relabeling proposal (ind_all -> lab_all) using the
%           frozen reference segmentation throat_body_ref (Step-4 output).
%   PASS 2: Apply all proposals at the end to produce throat_body_new.
%
% This function does NOT modify any label volume in-place.
%
% Based on original step5i_correct_division3d code. 
% -------------------------------------------------------------------------

% Output struct
proposal = struct('valid', false, 'ind', [], 'lab', []);

% -------------------------
% Build voxel index sets (REFERENCE ONLY)
% -------------------------
thrLbl = indices(index);

ind    = find(throat_body_ref == thrLbl);                % throat voxels
b1Lbl  = throat_center(x1,5);
b2Lbl  = throat_center(x1,6);
ind_b1 = find(throat_body_ref == b1Lbl);                 % body1 voxels
ind_b2 = find(throat_body_ref == b2Lbl);                 % body2 voxels
ind_all = [ind; ind_b1; ind_b2];

if isempty(ind_all)
    return;
end

% combined_profile_i must correspond to ind_all rows (as in the original)
if size(combined_profile_i,1) ~= numel(ind_all)
    error('step5i_correct_division3d_proposal:SizeMismatch', ...
        'combined_profile_i has %d rows but ind_all has %d elements.', ...
        size(combined_profile_i,1), numel(ind_all));
end

% Default: keep original labels (acts as "no change" where logic does not assign)
lab_all = throat_body_ref(ind_all);

% -------------------------
% Robust sign helper (reduces symmetry-breaking due to sign(0))
% -------------------------
tol = 1e-12;
sgn = @(x) (x > tol) - (x < -tol);   % returns -1, 0, +1

[~, ind1] = min(body_center_xy1(:,1));

% -------------------------
% MAIN LOGIC (identical structure to original, but writing into lab_all)
% -------------------------
if ind1 == 2
    % ---- For points 1 and 4 ----
    AB  = points1(4,:) - points1(1,:);
    AC2 = body2_center - points1(1,:);
    db2 = sgn( sind(atan2d(dot(unit_norm.', cross(AB,AC2)), dot(AB,AC2))) );

    ACt = center - points1(1,:);
    dbt = sgn( sind(atan2d(dot(unit_norm.', cross(AB,ACt)), dot(AB,ACt))) );

    AC1 = body1_center - points1(1,:);
    db1 = sgn( sind(atan2d(dot(unit_norm.', cross(AB,AC1)), dot(AB,AC1))) );

    AB1 = repmat(points1(4,:), numel(ind_all), 1) - repmat(points1(1,:), numel(ind_all), 1);
    AP  = combined_profile_i - repmat(points1(1,:), numel(ind_all), 1);
    d1  = sgn( sind(atan2d( dot(repmat(unit_norm.',numel(ind_all),1), cross(AB1,AP),2), dot(AB1,AP,2) )) );

    % ---- For points 2 and 3 ----
    ABi  = points1(3,:) - points1(2,:);
    % NOTE: the original code used points1(1,:) here for AC2i; that looks
    % inconsistent with the else branch (which uses points1(2,:)).
    % Retain the original behavior for compatibility.
    AC2i = body2_center - points1(1,:);
    db2i = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,AC2i)), dot(ABi,AC2i))) );

    ACti = center - points1(2,:);
    dbti = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,ACti)), dot(ABi,ACti))) );

    AC1i = body1_center - points1(2,:);
    db1i = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,AC1i)), dot(ABi,AC1i))) );

    AB1i = repmat(points1(3,:), numel(ind_all), 1) - repmat(points1(2,:), numel(ind_all), 1);
    APi  = combined_profile_i - repmat(points1(2,:), numel(ind_all), 1);
    d1i  = sgn( sind(atan2d( dot(repmat(unit_norm.',numel(ind_all),1), cross(AB1i,APi),2), dot(AB1i,APi,2) )) );

    % --- Branches: same as original, but set lab_all(...) instead of throat_body_new(...)
    if (db1 ~= db2i) && (dbt ~= db1 || dbti ~= db2i)   % .|.|.  .| |..  ..||.
        lab_all(d1 == db1) = b2Lbl;
        lab_all(d1i == db2i) = b1Lbl;
        lab_all(d1 ~= db1 & d1i ~= db2i) = thrLbl;

    else
        if (db1 ~= db2i) && (dbt == db1 && dbti == db2i) % |...|
            lab_all(d1 ~= db1)  = b2Lbl;
            lab_all(d1i ~= db2i)= b1Lbl;
            lab_all(d1 == db1 & d1i == db2i) = thrLbl;

        else
            if (db1 == db2i)

                if (dbti == db2i) && (dbt ~= db1)      % |..|.
                    lab_all(d1 == db1) = b2Lbl;
                    lab_all(d1i ~= db2i) = b1Lbl;
                    lab_all(d1 ~= db1 & d1i == db2i) = thrLbl;

                else
                    if (dbti ~= db2i) && (dbt == db1)  % .|..|
                        lab_all(d1 ~= db1) = b2Lbl;
                        lab_all(d1i == db2i) = b1Lbl;
                        lab_all(d1 == db1 & d1i ~= db2i) = thrLbl;

                    else
                        if (dbti == db2i) && (dbt == db1) && (dbt ~= db2) % |.|..
                            lab_all(d1 == db1) = b2Lbl;
                            lab_all(d1i ~= db2i) = b1Lbl;
                            lab_all(d1 ~= db1 & d1i == db2i) = thrLbl;

                        else
                            if (dbti == db2i) && (dbt == db1) && (dbt == db2) % ..|.|
                                lab_all(d1 ~= db1) = b2Lbl;
                                lab_all(d1i == db2i) = b1Lbl;
                                lab_all(d1 == db1 & d1i ~= db2i) = thrLbl;
                            end
                        end
                    end
                end
            end
        end
    end

else
    % positions 3 and 2 with body center 1; 1 and 4 with body center 2

    % ---- For points 1 and 4 ----
    AB  = points1(4,:) - points1(1,:);
    AC2 = body2_center - points1(1,:);
    db2 = sgn( sind(atan2d(dot(unit_norm.', cross(AB,AC2)), dot(AB,AC2))) );

    ACt = center - points1(1,:);
    dbt = sgn( sind(atan2d(dot(unit_norm.', cross(AB,ACt)), dot(AB,ACt))) );

    AC1 = body1_center - points1(1,:);
    db1 = sgn( sind(atan2d(dot(unit_norm.', cross(AB,AC1)), dot(AB,AC1))) );

    AB1 = repmat(points1(4,:), numel(ind_all), 1) - repmat(points1(1,:), numel(ind_all), 1);
    AP  = combined_profile_i - repmat(points1(1,:), numel(ind_all), 1);
    d1  = sgn( sind(atan2d( dot(repmat(unit_norm.',numel(ind_all),1), cross(AB1,AP),2), dot(AB1,AP,2) )) );

    % ---- For points 2 and 3 ----
    ABi  = points1(3,:) - points1(2,:);
    AC2i = body2_center - points1(2,:);
    db2i = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,AC2i)), dot(ABi,AC2i))) );

    ACti = center - points1(2,:);
    dbti = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,ACti)), dot(ABi,ACti))) );

    AC1i = body1_center - points1(2,:);
    db1i = sgn( sind(atan2d(dot(unit_norm.', cross(ABi,AC1i)), dot(ABi,AC1i))) );

    AB1i = repmat(points1(3,:), numel(ind_all), 1) - repmat(points1(2,:), numel(ind_all), 1);
    APi  = combined_profile_i - repmat(points1(2,:), numel(ind_all), 1);
    d1i  = sgn( sind(atan2d( dot(repmat(unit_norm.',numel(ind_all),1), cross(AB1i,APi),2), dot(AB1i,APi,2) )) );

    if (db2 ~= db1i) && (dbt ~= db2 || dbti ~= db1i)   % .|.|.  .| |..  ..||.
        lab_all(d1 == db2) = b2Lbl;
        lab_all(d1i == db1i) = b1Lbl;
        lab_all(d1 ~= db2 & d1i ~= db1i) = thrLbl;

    else
        if (db2 ~= db1i) && (dbt == db2 && dbti == db1i) % |...|
            lab_all(d1 ~= db2)  = b2Lbl;
            lab_all(d1i ~= db1i)= b1Lbl;
            lab_all(d1 == db2 & d1i == db1i) = thrLbl;

        else
            if (db2 == db1i)

                if (dbti == db1i) && (dbt ~= db2)      % |..|.
                    lab_all(d1 == db2) = b2Lbl;
                    lab_all(d1i ~= db1i) = b1Lbl;
                    lab_all(d1 ~= db2 & d1i == db1i) = thrLbl;

                else
                    if (dbti ~= db1i) && (dbt == db2)  % .|..|
                        lab_all(d1 ~= db2) = b2Lbl;
                        lab_all(d1i == db1i) = b1Lbl;
                        lab_all(d1 == db2 & d1i ~= db1i) = thrLbl;

                    else
                        if (dbti == db1i) && (dbt == db2) && (dbt ~= db1) % |.|..
                            lab_all(d1 == db2) = b2Lbl;
                            lab_all(d1i ~= db1i) = b1Lbl;
                            lab_all(d1 ~= db2 & d1i == db1i) = thrLbl;

                        else
                            if (dbti == db1i) && (dbt == db2) && (dbt == db1) % ..|.|
                                lab_all(d1 ~= db2) = b2Lbl;
                                lab_all(d1i == db1i) = b1Lbl;
                                lab_all(d1 == db2 & d1i ~= db1i) = thrLbl;
                            end
                        end
                    end
                end
            end
        end
    end
end

%{
% -------------------------
% POST-FILTER 1: Distance gate to prevent far-field throat assignment
% -------------------------
% Compute robust gate radius from ORIGINAL throat voxels (Step-4)
thrPts = combined_profile_i(1:numel(ind), :);   % NOTE: ind is first in ind_all by construction
d_thr  = sqrt(sum((thrPts - center).^2, 2));

% Robust scale (median + k*MAD); tune k depending on aggressiveness
d_med = median(d_thr);
d_mad = mad(d_thr, 1);          % robust deviation
k_gate = 6;                     % typical 4–8
r_gate = d_med + k_gate * d_mad;

% Apply to ALL voxels currently proposed as throat
d_all = sqrt(sum((combined_profile_i - center).^2, 2));
far_throat = (lab_all == thrLbl) & (d_all > r_gate);

if any(far_throat)
    % Reassign far-field "throat" voxels to the closer body center
    d_to_b1 = sqrt(sum((combined_profile_i - body1_center).^2, 2));
    d_to_b2 = sqrt(sum((combined_profile_i - body2_center).^2, 2));

    % Only change those far_throat voxels; choose nearer body
    lab_all(far_throat & (d_to_b1 <= d_to_b2)) = b1Lbl;
    lab_all(far_throat & (d_to_b2 <  d_to_b1)) = b2Lbl;
end

% -------------------------
% POST-FILTER 2: Enforce contiguity of throat (keep only component connected to original throat)
% -------------------------
% Binary mask over FULL volume, but restricted to local labels for efficiency
volSize = size(throat_body_ref);

% Candidate throat voxels after plane-based assignment
cand_thr_lin = ind_all(lab_all == thrLbl);

if ~isempty(cand_thr_lin)
    % Seeds: original throat voxels in this local set (Step-4 throat)
    seed_lin = ind;   % original throat indices (already linear indices in full volume)

    % Intersect seeds with candidate set (in case everything got reassigned)
    seed_lin = intersect(seed_lin, cand_thr_lin);
    if ~isempty(seed_lin)
        % Keep only voxels in cand_thr_lin that are 6-connected to any seed_lin
        keep_thr = keep_connected_to_seeds(volSize, cand_thr_lin, seed_lin);

        % Any candidate throat voxels NOT connected -> reassign to nearest body
        drop_lin = setdiff(cand_thr_lin, keep_thr);
        if ~isempty(drop_lin)
            % Map drop_lin back to rows in ind_all/lab_all
            [isDrop, loc] = ismember(drop_lin, ind_all);
            loc = loc(isDrop);

            d_to_b1 = sqrt(sum((combined_profile_i(loc,:) - body1_center).^2, 2));
            d_to_b2 = sqrt(sum((combined_profile_i(loc,:) - body2_center).^2, 2));

            lab_all(loc(d_to_b1 <= d_to_b2)) = b1Lbl;
            lab_all(loc(d_to_b2 <  d_to_b1)) = b2Lbl;
        end
    end
end
%}

% Finalize proposal (SPARSE: store only changes vs reference)
refLab = throat_body_ref(ind_all);
keep   = (lab_all ~= refLab);

proposal.valid = any(keep);
proposal.ind   = ind_all(keep);
proposal.lab   = lab_all(keep);

% Optional: store throat/body labels for faster visualization without find()
proposal.thrLbl = thrLbl;
proposal.b1Lbl  = b1Lbl;
proposal.b2Lbl  = b2Lbl;

end

function keep_lin = keep_connected_to_seeds(volSize, cand_lin, seed_lin)
%KEEP_CONNECTED_TO_SEEDS  Keep candidate voxels connected to seeds.
% Supports 2D (4-neighborhood) and 3D (6-neighborhood).
% Memory-light: BFS only over candidate set.

volSize = volSize(:).';
nd = numel(volSize);
if nd < 2
    error('volSize must have at least 2 dimensions.');
end

nx = volSize(1);
ny = volSize(2);
if nd >= 3
    nz = volSize(3);
else
    nz = 1;
end

% Unique, column vectors
cand_lin = unique(cand_lin(:));
seed_lin = unique(seed_lin(:));

% Restrict seeds to candidates
seed_lin = intersect(seed_lin, cand_lin);
if isempty(seed_lin)
    keep_lin = cand_lin([]); % empty
    return;
end

% Map candidate linear indices -> local indices 1..M
M = numel(cand_lin);
[~, locSeed] = ismember(seed_lin, cand_lin);

visited = false(M,1);
q = locSeed(:);
visited(q) = true;

% Subscripts for ALL candidates (define iz always)
if nz == 1
    [ix, iy] = ind2sub([nx, ny], cand_lin);
    iz = ones(size(ix));  % <-- FIX: define iz for 2D
else
    [ix, iy, iz] = ind2sub([nx, ny, nz], cand_lin);
end

% Build a unique integer key for each candidate voxel
key = ix + (iy-1)*nx + (iz-1)*nx*ny;

% Sort keys for fast binary search
[ks, order] = sort(key);
locOfKey = order;  % locOfKey(p) is the local index whose key is ks(p)

    function loc = findLocByKey(kq)
        % Binary search in ks; returns local index (0 if not found)
        lo = 1; hi = numel(ks);
        loc = 0;
        while lo <= hi
            mid = floor((lo+hi)/2);
            if ks(mid) == kq
                loc = locOfKey(mid);
                return;
            elseif ks(mid) < kq
                lo = mid + 1;
            else
                hi = mid - 1;
            end
        end
    end

% BFS
head = 1;
while head <= numel(q)
    v = q(head); head = head + 1;

    i = ix(v); j = iy(v); k = iz(v);

    % x-1, x+1
    if i > 1
        loc = findLocByKey((i-1) + (j-1)*nx + (k-1)*nx*ny);
        if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
    end
    if i < nx
        loc = findLocByKey((i+1) + (j-1)*nx + (k-1)*nx*ny);
        if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
    end

    % y-1, y+1
    if j > 1
        loc = findLocByKey(i + (j-2)*nx + (k-1)*nx*ny);
        if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
    end
    if j < ny
        loc = findLocByKey(i + (j)*nx + (k-1)*nx*ny);
        if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
    end

    % z-1, z+1 (3D only)
    if nz > 1
        if k > 1
            loc = findLocByKey(i + (j-1)*nx + (k-2)*nx*ny);
            if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
        end
        if k < nz
            loc = findLocByKey(i + (j-1)*nx + (k)*nx*ny);
            if loc>0 && ~visited(loc), visited(loc)=true; q(end+1)=loc; end %#ok<AGROW>
        end
    end
end

keep_lin = cand_lin(visited);

end


