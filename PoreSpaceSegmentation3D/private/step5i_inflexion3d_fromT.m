% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [points3d, combined_profile_3d, throat_axis_center_3d] = ...
    step5i_inflexion3d_fromT(points_uv, points_flag, T, ...
                             mid_profile, body1_profile, body2_profile)
%STEP5I_INFLEXION3D_FROMT  Back-project 2D inflexion points to 3D using transform T.
%
% Inputs:
%   points_uv   : (N x 2) 2D points in the canonical uv frame (post-rotation if T.rot.active)
%   points_flag : (N x 1) selector (0 => snap to mid only, else snap to combined)
%   T           : struct returned by step5c_bestfit2D_robust (contains p0,e1,e2 and 2D rotation)
%   mid_profile, body1_profile, body2_profile : 3D point clouds on the best-fit plane (from Step 5b)
%
% Outputs:
%   points3d            : (N x 3) snapped 3D voxel points (or exact plane points if you disable snapping below)
%   combined_profile_3d : concatenated 3D cloud used for snapping
%   throat_axis_center_3d : 3D point for throat axis center (centroid of mid_profile in uv back-mapped)

if size(points_uv,2) ~= 2
    error('points_uv must be N x 2.');
end
if nargin < 2 || isempty(points_flag)
    points_flag = zeros(size(points_uv,1),1);
end

% Build combined cloud
combined_profile_3d = [mid_profile; body1_profile; body2_profile];

% --- 1) Undo canonical rotation in 2D if applied ---
uv_raw = points_uv;
if isfield(T,'rot') && isfield(T.rot,'active') && T.rot.active
    phi = T.rot.phi;        % forward rotation was phi
    c   = T.rot.c(:).';     % 1x2
    Rinv = [cos(-phi) -sin(-phi); sin(-phi) cos(-phi)];  % inverse of R(phi)
    uv_raw = (uv_raw - c) * Rinv + c;
end

% --- 2) Analytical back-map to 3D plane ---
p0 = T.p0(:).';
e1 = T.e1(:).';
e2 = T.e2(:).';

Xplane = p0 + uv_raw(:,1).*e1 + uv_raw(:,2).*e2;  % N x 3

% --- 3) Snap to nearest voxel point cloud in 3D (recommended if you need voxel outputs) ---
points3d = zeros(size(Xplane));
for i = 1:size(Xplane,1)
    if points_flag(i) == 0
        idx = knnsearch(mid_profile, Xplane(i,:));  % 3D kNN
        points3d(i,:) = mid_profile(idx,:);
    else
        idx = knnsearch(combined_profile_3d, Xplane(i,:));
        points3d(i,:) = combined_profile_3d(idx,:);
    end
end

% --- 4) Throat axis center (centroid of mid_profile in uv, back-mapped) ---
% Compute centroid in uv frame:
mid_uv_raw = (mid_profile - p0) * [e1(:), e2(:)];  % Nx2 (raw uv, unrotated)
throat_axis_center_uv = mean(mid_uv_raw, 1);       % 1x2
throat_axis_center_3d = p0 + throat_axis_center_uv(1).*e1 + throat_axis_center_uv(2).*e2;

% Optional: snap throat axis center to mid_profile in 3D
idx = knnsearch(mid_profile, throat_axis_center_3d);
throat_axis_center_3d = mid_profile(idx,:);

end
