% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function neighbour = step2_neighbours(separator, farthest_voxel, locals, voxels, ste)
% Optimized Step 2:
% - Work only with pore centers + farthest radii
% - Two pores are neighbours if their "spheres" intersect:
%       ||c_i - c_j|| <= r_i + r_j
%
% separator is kept only for interface compatibility (not used here).

    % Number of pore bodies
    num_pores = numel(locals);

    if num_pores <= 1
        neighbour = zeros(0, 2);
        return;
    end

    % Pore centers (P x 3)
    centers = voxels(locals, :);          % double
    % Radii (P x 1)
    r = farthest_voxel(:);                % column vector

    % Pairwise distances between centers (upper triangle only)
    % pdist returns condensed form; squareform expands to full matrix.
    D = squareform(pdist(centers));       % (P x P)
    Rsum = r + r.';                       % (P x P), r_i + r_j

    % Neighbours: spheres intersect
    tol = 2*ste;   % or 1*ste; see note below
    mask = (D <= (Rsum + tol)) & triu(true(num_pores), 1);

    % Neighbours: spheres intersect
    %mask = (D <= Rsum) & triu(true(num_pores), 1);  % upper triangle, i < j

    [i_idx, j_idx] = find(mask);
    neighbour = [i_idx, j_idx];           % each row [i, j], 1-based pore labels
end
