% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [mid_profile, surface_profile, unit_norm, body1_profile, body2_profile] = ...
    step5b_bestfit(set_body1, set_body2, set2, ste, x1, set, voxels, locations, locals, throat_center, vis)
% Robust best-fit plane for Step 5b (Option A):
% - Plane contains the body-center axis (C1-C2 line)
% - Among all such planes, choose the one whose normal is closest to PCA normal of throat cloud
% - Translation/rotation robust, deterministic sign choice (relative to PCA normal)

% ---------------------------
% 1) Collect data
% ---------------------------
A = double(voxels(locations, :));
if size(A,1) < 3
    error('step5b_bestfit: not enough points in locations to fit a plane.');
end

C1 = double(voxels(locals(throat_center(x1,5)), :));
C2 = double(voxels(locals(throat_center(x1,6)), :));

% Anchor point on plane: midpoint of body centers (lies on the required plane)
p0 = 0.5*(C1 + C2);

% ---------------------------
% 2) Axis direction v (must lie in plane)
% ---------------------------
v = (C2 - C1);
nv = norm(v);

if nv < 1e-12
    % Degenerate case: body centers coincide (rare but possible in mislabeled data)
    % Fallback: use dominant direction from the throat cloud (1st PCA direction)
    A0tmp = A - mean(A,1);
    [~,~,Vtmp] = svd(A0tmp, 0);
    v = Vtmp(:,1).';  % row
    nv = norm(v);
    if nv < 1e-12
        v = [1,0,0];
    else
        v = v / nv;
    end
else
    v = v / nv;
end
v = v(:); % 3x1

% ---------------------------
% 3) Compute PCA plane normal (unconstrained)
% ---------------------------
% Center to remove translation sensitivity
mu = mean(A,1);
A0 = A - mu;

% Optional: downweight over-dense sampling (kept OFF by default)
% If you want, set useWeights=true and choose kNN-based weights.
useWeights = false;

if ~useWeights
    [~,S,V] = svd(A0, 0);
    n_pca = V(:,end);            % 3x1 (smallest singular vector)
else
    % Simple isotropic weights: inverse local density via random subsampling (lightweight)
    % You can replace with kNN density if needed.
    idx = randperm(size(A0,1), min(size(A0,1), 5000));
    A0s = A0(idx,:);
    [~,S,V] = svd(A0s, 0);
    n_pca = V(:,end);
end

% ---- Optional diagnostics (uncomment if needed)
 s = diag(S);
 degeneracy = s(end-1)/max(s(end),eps);  % if ~1 => near-ambiguous normal
 %fprintf('Plane degeneracy indicator (s2/s3): %.4f\n', degeneracy);

% ---------------------------
% 4) Option A: enforce n·v = 0 while staying closest to n_pca
% ---------------------------
% Primary candidate: project n_pca onto subspace orthogonal to v
n1 = n_pca - (dot(n_pca, v))*v;
n1n = norm(n1);

% If nearly zero, PCA normal is parallel to v => ambiguity
if n1n < 1e-12
    % Build two deterministic candidates orthogonal to v using reference axes,
    % then choose the one closest to n_pca (Option A tie-break).
    refA = [0;0;1];
    if norm(cross(v, refA)) < 1e-12
        refA = [0;1;0];
    end
    nA = cross(v, refA); nA = nA / max(norm(nA), eps);

    refB = [1;0;0];
    if norm(cross(v, refB)) < 1e-12
        refB = [0;1;0];
    end
    nB = cross(v, refB); nB = nB / max(norm(nB), eps);

    % Choose the candidate that aligns best with n_pca (max absolute dot)
    if abs(dot(nA, n_pca)) >= abs(dot(nB, n_pca))
        n = nA;
    else
        n = nB;
    end
else
    n = n1 / n1n;
end

% ---------------------------
% 5) Deterministic sign convention (avoid random flips)
% ---------------------------
% Make n point in the same hemisphere as n_pca
if dot(n, n_pca) < 0
    n = -n;
end

unit_norm = n;  % 3x1 unit normal

% ---------------------------
% 6) Project points onto plane
% ---------------------------
mid_profile     = local_project_points(double(voxels(locations,:)), p0, unit_norm);
surface_profile = local_project_points(double(set2),               p0, unit_norm);
body1_profile   = local_project_points(double(set_body1),          p0, unit_norm);
body2_profile   = local_project_points(double(set_body2),          p0, unit_norm);

% ---------------------------
% 7) Visualization (optional)
% ---------------------------
if vis == 1
    figure;
    scatter3(set(:,1), set(:,2), set(:,3), 5, 'b'); hold on;
    scatter3(set2(:,1), set2(:,2), set2(:,3), 5, 'c');
    scatter3(mid_profile(:,1), mid_profile(:,2), mid_profile(:,3), 10, 'g', 'filled');

    scatter3(C1(1), C1(2), C1(3), 60, 'r', 'filled');
    scatter3(C2(1), C2(2), C2(3), 60, 'r', 'filled');
    scatter3(throat_center(x1,1), throat_center(x1,2), throat_center(x1,3), 60, 'm', 'filled');

    % Draw a line along v through p0 for sanity
    t = linspace(-50*ste, 50*ste, 20).';
    linePts = p0 + t.*(v.');
    plot3(linePts(:,1), linePts(:,2), linePts(:,3), 'g-', 'LineWidth', 2);

    axis equal; grid on;
    title('projection of throat and body voxels on best fit 3D plane (Option A)');
end

end

% ============================================================
% Local helper: orthogonal projection of points to plane
% ============================================================
function Xproj = local_project_points(X, p0, n)
% X: Nx3
% p0: 1x3 (point on plane)
% n:  3x1 unit normal

if isempty(X)
    Xproj = X;
    return;
end

p0 = p0(:).';   % 1x3
n  = n(:);      % 3x1

% Signed distance along normal: (X - p0)·n
d = (X - p0) * n;       % Nx1

% Projection: X - d*n'
Xproj = X - d * (n.');

end

