% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function locals = step1_local_maxima(vox3d, ste, mean_input_radius, porosity)
%STEP1_LOCAL_MAXIMA Compute centers of pore volumes (local EDT maxima).
% Optimized but numerically equivalent to the original triple-loop version.
%
% 1) Compute EDT of pore phase
% 2) Compute neighbourhood size from (1.5 * radius * porosity / ste)
% 3) Use morphological dilation to find *candidate* local maxima
% 4) Apply the original "scan + zero-out ties" logic ONLY on candidates
%
% Inputs:
%   vox3d            : logical / 0-1 3D array (1 = pore, 0 = solid)
%   ste              : voxel size
%   mean_input_radius: mean particle radius (mpd/2)
%   porosity         : scalar porosity
%
% Output:
%   locals : column vector of linear indices into vox3d

% --- 1) Euclidean distance transform (same as original) ---
vox3d1 = bwdist(~vox3d);   % distance to nearest solid voxel
sb     = size(vox3d1);

% --- 2) Local maxima stability criterion for neighbourhood size ---
val = (1.5 * mean_input_radius * porosity) / ste;
if val > 1
    ii = round(val);
    % clamp for safety, but if val<=50 this matches original exactly
    ii = max(1, min(ii, 50));
else
    ii = 5;
end

nn   = 1:50;
r    = nn(ii);        % “extension rate”
step = 2*r + 1;       % cubic window size
bound = (step - 1)/2; % half-width

% --- 3) Candidate local maxima via morphological dilation ---
% A voxel is a candidate if:
%   - dist > 0 (void)
%   - dist == max over its step×step×step neighbourhood (w.r.t *original* EDT)
se        = true(step, step, step);
localMax  = imdilate(vox3d1, se);
cand_mask = (vox3d1 > 0) & (vox3d1 == localMax);

cand_idx = find(cand_mask);          % linear indices in vox3d1
numCand  = numel(cand_idx);

% If no candidates, return empty
if numCand == 0
    locals = zeros(0,1);
    return;
end

% --- 4) Build padded "main" like original code ---
main = zeros(sb(1)+2*bound, sb(2)+2*bound, sb(3)+2*bound, 'like', vox3d1);
main(bound+1:bound+sb(1), ...
     bound+1:bound+sb(2), ...
     bound+1:bound+sb(3)) = vox3d1;

% Precompute candidate coordinates in the padded array
[I, J, K] = ind2sub(sb, cand_idx);
I = I + bound;
J = J + bound;
K = K + bound;

% --- 5) Apply the original logic, but only on candidates ---
locals  = zeros(numCand,1);
counter = 1;

for n = 1:numCand
    i = I(n);
    j = J(n);
    k = K(n);

    % This voxel may already have been zeroed out by an earlier tie
    if main(i,j,k) == 0
        continue;
    end

    % Extract the same step×step×step neighbourhood as in the original
    ii0 = i - bound;
    jj0 = j - bound;
    kk0 = k - bound;
    temp = main(ii0:ii0+step-1, ...
                jj0:jj0+step-1, ...
                kk0:kk0+step-1);

    nonzerosTemp = temp(temp ~= 0);
    if isempty(nonzerosTemp)
        continue;
    end

    MAX       = max(nonzerosTemp);
    centerVal = temp(bound+1, bound+1, bound+1);
    NumberFZ  = sum(nonzerosTemp == MAX);

    if centerVal == MAX && NumberFZ == 1
        % Unique maximum in its neighbourhood: keep as local center
        % Convert padded coords back to original sb indexing:
        ii_orig = i - bound;
        jj_orig = j - bound;
        kk_orig = k - bound;
        locals(counter,1) = sub2ind(sb, ii_orig, jj_orig, kk_orig);
        counter = counter + 1;

    elseif centerVal == MAX && NumberFZ ~= 1
        % Plateau tie: zero out this voxel, exactly like original
        main(i,j,k) = 0;
    end
end

% Trim unused entries and sort (same as original)
locals = locals(1:counter-1);
locals = sort(locals);

end
