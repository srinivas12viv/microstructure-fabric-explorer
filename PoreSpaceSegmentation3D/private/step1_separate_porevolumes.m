% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [separator,farthest_voxel,locals] = step1_separate_porevolumes(locals,voxels,vox3d)
    voxelInfo = vox3d(:);
    N_voxel   = numel(voxelInfo);

    x_voxel = voxels(:,1);
    y_voxel = voxels(:,2);
    z_voxel = voxels(:,3);

    peak_voxels = [x_voxel(locals), y_voxel(locals), z_voxel(locals)];
    num_peaks   = size(peak_voxels,1);

    % ✅ keep as double, like original code
    separator = zeros(N_voxel,1);  

    kdtree = KDTreeSearcher(peak_voxels);

    chunkSize = 50000;
    startIdx  = 1;

    while startIdx <= N_voxel
        endIdx = min(startIdx + chunkSize - 1, N_voxel);
        idxRange = startIdx:endIdx;

        voxels_chunk = [x_voxel(idxRange), ...
                        y_voxel(idxRange), ...
                        z_voxel(idxRange)];
        BinaryVoxelPortion = voxelInfo(idxRange) ~= 0;

        nearestIdx = zeros(numel(idxRange),1);   % ✅ double
        if any(BinaryVoxelPortion)
            nearestIdx(BinaryVoxelPortion) = knnsearch( ...
                kdtree, voxels_chunk(BinaryVoxelPortion,:));
        end

        separator(idxRange) = nearestIdx;
        startIdx = endIdx + 1;
    end

    % -------- farthest_voxel as before --------
    labels = separator;          % 0,1,...,num_peaks
    mask   = labels > 0;
    labelsNZ = labels(mask);     % 1..num_peaks

    x_pore = x_voxel(mask);
    y_pore = y_voxel(mask);
    z_pore = z_voxel(mask);

    px = peak_voxels(labelsNZ,1);
    py = peak_voxels(labelsNZ,2);
    pz = peak_voxels(labelsNZ,3);

    dx = x_pore - px;
    dy = y_pore - py;
    dz = z_pore - pz;
    d  = sqrt(dx.^2 + dy.^2 + dz.^2);

    farthest_voxel = accumarray(labelsNZ, d, [num_peaks 1], @max, 0);

    % Empty regions differ from a valid singleton region with zero radius.
    counts = accumarray(labelsNZ, 1, [num_peaks 1], @sum, 0);
    recordi = find(counts == 0);
    if ~isempty(recordi)
        locals(recordi)         = [];
        farthest_voxel(recordi) = [];
        remap = zeros(num_peaks,1);
        retained = find(counts > 0);
        remap(retained) = 1:numel(retained);
        separator(mask) = remap(labelsNZ);
    end

    farthest_voxel = farthest_voxel(:);
end
