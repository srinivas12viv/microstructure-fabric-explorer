% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [mid_uv, body_uv, throat_uv, surf_uv, body1_uv, body2_uv, T] = ...
    step5c_bestfit2D_robust(locals, throat_center, unit_norm, x1, voxels, ...
                            mid_profile, surface_profile, body1_profile, body2_profile, vis, doRotate)
%STEP5C_BESTFIT2D_ROBUST  Robust 3D->2D mapping on best-fit plane, with optional canonical rotation.
%
% doRotate (optional): true/false (default true) to rotate 2D so body centers lie on +u axis.

if nargin < 12 || isempty(doRotate)
    doRotate = true;
end

% ---------------------------
% 0) Basic checks
% ---------------------------
n = unit_norm(:);
nn = norm(n);
if nn < 1e-12
    error('step5c_bestfit2D_robust: unit_norm is near zero.');
end
n = n / nn;

% Body centers in 3D
C1 = double(voxels(locals(throat_center(x1,5)), :));
C2 = double(voxels(locals(throat_center(x1,6)), :));
p0 = 0.5*(C1 + C2);
p0 = p0(:).';

% Direction along body axis
v = (C2 - C1);
nv = norm(v);
if nv < 1e-12
    v = [1 0 0];
else
    v = v / nv;
end
v = v(:);

% ---------------------------
% 1) Build orthonormal basis on plane
% ---------------------------
v_proj = v - (dot(v,n))*n;
if norm(v_proj) < 1e-12
    ref = [0;0;1];
    if norm(cross(n, ref)) < 1e-12
        ref = [0;1;0];
    end
    e1 = cross(ref, n);
    e1 = e1 / max(norm(e1), eps);
else
    e1 = v_proj / norm(v_proj);
end

e2 = cross(n, e1);
e2 = e2 / max(norm(e2), eps);

% Re-orthogonalize (numerical hygiene)
e1 = e1 - dot(e1,n)*n;
e1 = e1 / max(norm(e1), eps);
e2 = cross(n,e1);
e2 = e2 / max(norm(e2), eps);

% ---------------------------
% 2) Map to (u,v)
% ---------------------------
mid_uv   = map_to_uv(mid_profile,     p0, e1, e2);
surf_uv  = map_to_uv(surface_profile, p0, e1, e2);
body1_uv = map_to_uv(body1_profile,   p0, e1, e2);
body2_uv = map_to_uv(body2_profile,   p0, e1, e2);

% Body centers (project to plane first for consistency)
C1p = project_to_plane(C1, p0, n);
C2p = project_to_plane(C2, p0, n);
body_uv = map_to_uv([C1p; C2p], p0, e1, e2);

% Throat 2D center: centroid of throat mid-profile in uv
if isempty(mid_uv)
    throat_uv = [NaN NaN];
else
    throat_uv = [mean(mid_uv(:,1)), mean(mid_uv(:,2))];
end

% ---------------------------
% 3) Optional canonical rotation: align body centers with +u
% ---------------------------
T.rot.active = false;
T.rot.phi    = 0;           % applied rotation angle (radians)
T.rot.c      = throat_uv;   % 2D rotation center

if doRotate && all(isfinite(throat_uv)) && size(body_uv,1) == 2
    dx = body_uv(2,1) - body_uv(1,1);
    dy = body_uv(2,2) - body_uv(1,2);

    if hypot(dx,dy) >= 1e-12
        theta = atan2(dy, dx);   % signed angle of body-axis in uv
        phi   = -theta;          % rotate so axis aligns with +u

        R = [cos(phi) -sin(phi); sin(phi) cos(phi)];
        c = throat_uv(:).';

        mid_uv   = (mid_uv   - c) * R + c;
        surf_uv  = (surf_uv  - c) * R + c;
        body1_uv = (body1_uv - c) * R + c;
        body2_uv = (body2_uv - c) * R + c;
        body_uv  = (body_uv  - c) * R + c;

        throat_uv = [mean(mid_uv(:,1)), mean(mid_uv(:,2))];

        T.rot.active = true;
        T.rot.phi    = phi;
        T.rot.c      = c;
    end
end

% ---------------------------
% 4) Store transform
% ---------------------------
T.p0 = p0;       % 1x3
T.e1 = e1(:).';  % 1x3
T.e2 = e2(:).';  % 1x3
T.n  = n(:).';   % 1x3

% ---------------------------
% 5) Visualization
% ---------------------------
if vis == 1
    figure;
    scatter(mid_uv(:,1),   mid_uv(:,2),   8, 'b'); hold on;
    scatter(surf_uv(:,1),  surf_uv(:,2),  8, 'y');
    scatter(body1_uv(:,1), body1_uv(:,2), 8, 'g');
    scatter(body2_uv(:,1), body2_uv(:,2), 8, 'g');
    scatter(body_uv(:,1),  body_uv(:,2), 40, 'r', 'filled');
    scatter(throat_uv(1),  throat_uv(2), 40, 'm', 'filled');
    line([body_uv(1,1), body_uv(2,1)], [body_uv(1,2), body_uv(2,2)], 'Color','k');
    axis equal; grid on;
    title('Best-fit plane mapped to 2D (uv), canonicalized by body-axis rotation');
    xlabel('u'); ylabel('v');
end

end

% ---------- helpers ----------
function uv = map_to_uv(X, p0, e1, e2)
if isempty(X), uv = zeros(0,2); return; end
X = double(X);
p0 = p0(:).';
e1 = e1(:); e2 = e2(:);
Y = X - p0;
uv = [Y*e1, Y*e2];
end

function Xp = project_to_plane(X, p0, n)
X  = double(X(:).');
p0 = p0(:).';
n  = n(:);
d  = (X - p0) * n;
Xp = X - d * (n.');
end
