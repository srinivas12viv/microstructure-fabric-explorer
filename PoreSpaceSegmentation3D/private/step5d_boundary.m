% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function store_new = step5d_boundary(mid_profile_xy1, sig, throat_center_xy, ste, vis)
%STEP5D_BOUNDARY  Ordered XY boundary for throat cloud using alphaShape + fallbacks.
% Output store_new keeps the expected columns:
%   [x y r theta dtheta idx]
%
% Strategy:
%   1) Clean XY (round + stable unique)
%   2) Try alphaShape concave hull -> extract boundary edges -> order into loop
%      - pick largest loop by perimeter
%      - if fails, inflate alpha and retry a few times
%   3) Fallback to boundary() with shrink, then convhull()
%
% Notes:
% - Does NOT unique(boundary) because it breaks order.
% - alphaShape parameterized by a length scale, typically easier to tune than shrink.
% - If Stats toolbox exists, uses knnsearch for NN spacing; otherwise uses pdist2 fallback.

% ----------------------------
% 0) Clean XY cloud
% ----------------------------
XY = double(mid_profile_xy1(:,1:2));
XY = round(XY * 10^sig) / 10^sig;
XY = unique(XY, 'rows', 'stable');

if size(XY,1) < 20
    store_new = [XY, zeros(size(XY,1),4)];
    return;
end

% ----------------------------
% 1) alphaShape boundary (preferred)
% ----------------------------
B = [];
alpha_used = NaN;

% Robust spacing estimate
h = estimate_median_nn_spacing(XY);

% Throat boundaries often need tighter alpha than combined
% Start modest and inflate if it fragments.
alpha_mult_list = [1.6, 2.2, 3.0, 4.0];  % tune if needed

try
    for am = alpha_mult_list
        a = am * h;
        shp = alphaShape(XY(:,1), XY(:,2), a);

        if shp.numRegions < 1
            continue;
        end

        [E, V] = boundaryFacets(shp);  % edges + vertex coords (2D)
        if isempty(E) || isempty(V)
            continue;
        end

        loops = order_edges_into_loops(E);
        if isempty(loops)
            continue;
        end

        [Btry, ~] = pick_largest_loop_by_perimeter(V, loops);

        if ~isempty(Btry) && size(Btry,1) >= 12
            B = Btry;
            alpha_used = a;
            break;
        end
    end
catch
    B = [];
end

% ----------------------------
% 2) Fallbacks if alphaShape failed
% ----------------------------
if isempty(B) || size(B,1) < 12
    shrink = 0.75;  % throat boundary setting
    try
        idx = boundary(XY(:,1), XY(:,2), shrink);
        B = XY(idx,:);
    catch
        idx = convhull(XY(:,1), XY(:,2));
        B = XY(idx,:);
    end
end

% Clean boundary and close
B = remove_consecutive_duplicates(B);

if size(B,1) < 6
    store_new = [B, zeros(size(B,1),4)];
    return;
end

if any(B(1,:) ~= B(end,:))
    B = [B; B(1,:)];
end

% ----------------------------
% 3) Polar columns for compatibility
% ----------------------------
dx = B(:,1) - throat_center_xy(1);
dy = B(:,2) - throat_center_xy(2);

r  = hypot(dx, dy);
theta = atan2(dy, dx);
theta(theta < 0) = theta(theta < 0) + 2*pi;

% dtheta along curve order (not angle-sorted)
dtheta = [0; diff(theta)];

idxcol = (1:size(B,1)).';

store_new = [B, r, theta, dtheta, idxcol];

% ----------------------------
% 4) Visualization
% ----------------------------
if vis == 1
    figure; hold on;
    scatter(XY(:,1), XY(:,2), 4, 'k');

    plot(B(:,1), B(:,2), 'c-', 'LineWidth', 2);
    %scatter(throat_center_xy(1), throat_center_xy(2), 40, 'r', 'filled');

    axis equal; grid on;
    if isfinite(alpha_used)
        title(sprintf('step5d boundary (alphaShape, alpha=%.4g)', alpha_used));
    else
        title('step5d boundary (fallback: boundary/convhull)');
    end
end

end

% ======================================================================
% Helpers
% ======================================================================

function h = estimate_median_nn_spacing(XY)
% Robust median nearest-neighbor distance.

n = size(XY,1);

try
    % Stats & ML Toolbox
    [~, d] = knnsearch(XY, XY, 'K', 2); % 1st is itself (0), 2nd is NN
    nn = d(:,2);
    nn = nn(isfinite(nn) & nn > 0);
    h = median(nn);
catch
    % Fallback: subsample + pdist2
    m = min(n, 800);
    idx = randperm(n, m);
    Xs = XY(idx,:);
    D = pdist2(Xs, Xs);
    D(D==0) = inf;
    nn = min(D, [], 2);
    nn = nn(isfinite(nn) & nn > 0);
    h = median(nn);
end

if isempty(h) || ~isfinite(h) || h <= 0
    rngx = max(XY(:,1)) - min(XY(:,1));
    rngy = max(XY(:,2)) - min(XY(:,2));
    h = max([rngx, rngy]) / max(sqrt(n), 1);
    h = max(h, eps);
end
end

function loops = order_edges_into_loops(E)
% Convert undirected edges (Mx2) into one or more ordered vertex loops.
% Each loop is a vector of vertex indices in order (without repeated closure).

if isempty(E)
    loops = {};
    return;
end

E2 = sort(E,2);
nv = max(E2(:));

adj = cell(nv,1);
for i = 1:size(E2,1)
    a = E2(i,1); b = E2(i,2);
    adj{a} = [adj{a}, b];
    adj{b} = [adj{b}, a];
end

used = false(size(E2,1),1);
loops = {};

for e0 = 1:size(E2,1)
    if used(e0), continue; end

    vstart = E2(e0,1);
    vnext  = E2(e0,2);

    loop  = vstart;
    vprev = vstart;
    vcur  = vnext;

    used(e0) = true;

    maxSteps = size(E2,1) + 5;
    for step = 1:maxSteps
        loop(end+1) = vcur; %#ok<AGROW>

        neigh = adj{vcur};
        if isempty(neigh)
            break;
        end

        if numel(neigh) == 1
            vcand = neigh(1);
        else
            % choose neighbor not equal to vprev
            if neigh(1) == vprev
                vcand = neigh(2);
            else
                vcand = neigh(1);
            end
        end

        ee = sort([vcur, vcand]);
        k = find(E2(:,1)==ee(1) & E2(:,2)==ee(2) & ~used, 1);
        if ~isempty(k), used(k) = true; end

        vprev = vcur;
        vcur  = vcand;

        if vcur == vstart
            loop = loop(1:end-1); % drop repeated closure
            break;
        end
    end

    if numel(loop) >= 6
        loops{end+1} = loop; %#ok<AGROW>
    end
end
end

function [B, perim] = pick_largest_loop_by_perimeter(V, loops)
bestP = -inf;
bestB = [];

for i = 1:numel(loops)
    idx = loops{i};
    P = V(idx, :);

    d = hypot(diff(P(:,1)), diff(P(:,2)));
    d = [d; hypot(P(1,1)-P(end,1), P(1,2)-P(end,2))];
    L = sum(d);

    if L > bestP
        bestP = L;
        bestB = P;
    end
end

B = bestB;
perim = bestP;
end

function P = remove_consecutive_duplicates(P)
if isempty(P), return; end
d = hypot(diff(P(:,1)), diff(P(:,2)));
keep = [true; d > 0];
P = P(keep,:);
end
