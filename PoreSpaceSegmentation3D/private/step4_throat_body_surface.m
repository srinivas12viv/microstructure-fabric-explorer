% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function throat_body_surface_idx = step4_throat_body_surface(throat_body_3d, neighbour)
% throat_body_surface_idx: Nx2 array [linIndex, thrLbl]
% linIndex is in the linear indexing of throat_body_3d (same as find()).
%
% This computes throat-side interface voxels:
%   throat voxel that has at least one 26-neighborhood neighbor in the body.
%
% Inputs:
%   throat_body_3d : 3D label volume (0=solid, >0=body/throat labels)
%   neighbour      : [body1_label, body2_label, throat_label] per row

sz = size(throat_body_3d);
K  = ones(3,3,3);             % 26-neighborhood kernel

throat_body_surface_idx = [];

numPairs = size(neighbour, 1);

for i = 1:numPairs
    b1  = neighbour(i, 1);
    b2  = neighbour(i, 2);
    thr = neighbour(i, 3);

    if thr <= 0
        continue;
    end

    thrMask = (throat_body_3d == thr);
    if ~any(thrMask(:))
        continue;
    end

    % body 1 adjacency -> throat-side surface
    if b1 > 0
        b1Mask = (throat_body_3d == b1);
        b1Dil  = convn(double(b1Mask), K, 'same') > 0;  % 1-voxel 26-neigh dilation
        surf1  = thrMask & b1Dil;
        if any(surf1(:))
            lin1 = find(surf1);
            throat_body_surface_idx = [throat_body_surface_idx; ...
                                      [lin1, repmat(thr, numel(lin1), 1)]];
        end
    end

    % body 2 adjacency -> throat-side surface
    if b2 > 0
        b2Mask = (throat_body_3d == b2);
        b2Dil  = convn(double(b2Mask), K, 'same') > 0;
        surf2  = thrMask & b2Dil;
        if any(surf2(:))
            lin2 = find(surf2);
            throat_body_surface_idx = [throat_body_surface_idx; ...
                                      [lin2, repmat(thr, numel(lin2), 1)]];
        end
    end
end

% Remove duplicates (same voxel may be adjacent to both bodies or repeated)
if ~isempty(throat_body_surface_idx)
    throat_body_surface_idx = unique(throat_body_surface_idx, 'rows');
end

end
