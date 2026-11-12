function result = poreSpaceSegmentation3D(inputFile, outputDir, options)
%PORESPACESEGMENTATION3D Segment a binary MAT volume into bodies and throats.
% Framework: Roy, Frost & Roozbahani (2023); MIT notice: LICENSE.upstream.
% Thesis refinement: Srinivas Vivek Bokkisa. See docs/MODIFICATIONS.md.
% Array dimensions are X,Y,Z; vox3d=1 is pore and vox3d=0 is solid.
arguments
    inputFile (1,1) string
    outputDir (1,1) string
    options.VoxelSize (1,1) double {mustBePositive,mustBeFinite} = 1
    options.MeanParticleDiameter (1,1) double {mustBePositive,mustBeFinite} = 40
    options.D50 (1,1) double {mustBePositive,mustBeFinite} = 40
    options.Units (1,1) string = "voxel"
    options.UseParallel (1,1) logical = false
    options.SaveIntermediate (1,1) logical = false
end
required = {'bwdist','imdilate','knnsearch','KDTreeSearcher','pdist2','sgolayfilt'};
for k = 1:numel(required)
    if exist(required{k},'file') == 0 && exist(required{k},'class') == 0
        error('PoreSpace:Dependency','Required MATLAB function unavailable: %s',required{k});
    end
end
if options.UseParallel && ~license('test','Distrib_Computing_Toolbox')
    error('PoreSpace:Parallel','UseParallel requires Parallel Computing Toolbox.');
end
S = load(inputFile,'vox3d');
if ~isfield(S,'vox3d') || ~(isnumeric(S.vox3d) || islogical(S.vox3d)) ...
        || ~isreal(S.vox3d) || ndims(S.vox3d) ~= 3 ...
        || any(~isfinite(S.vox3d(:))) || any(S.vox3d(:) ~= 0 & S.vox3d(:) ~= 1)
    error('PoreSpace:Input','Input must contain a finite binary 3D array named vox3d.');
end
vox3d = logical(S.vox3d);
sz = size(vox3d);
if any(sz ~= sz(1))
    error('PoreSpace:Domain','This release supports cubic volumes with isotropic voxels.');
end
if ~any(vox3d(:)) || all(vox3d(:))
    error('PoreSpace:Phases','Both solid and pore phases must be present.');
end
if ~isfolder(outputDir), mkdir(outputDir); end
started = tic;
res = options.VoxelSize;
voxels = step0_voxel_coordinates(0,0,sz(1)*res,sz(2)*res,0,sz(3)*res,res);
fprintf('Step 1: pore-volume candidates\n');
locals = step1_local_maxima(vox3d,res,options.MeanParticleDiameter/2,nnz(vox3d)/numel(vox3d));
if isempty(locals), error('PoreSpace:Seeds','No pore-body candidates found.'); end
[separator,farthest_voxel,locals] = step1_separate_porevolumes(locals,voxels,vox3d);
fprintf('Steps 2-3: neighbours and throat centers\n');
neighbour = step2_neighbours(separator,farthest_voxel,locals,voxels,res);
numBodyLabels = numel(locals);
[throat_center,neighbour,constriction_length] = ...
    step3_throatcenters(separator,voxels,res,locals,numBodyLabels,neighbour);
fprintf('Step 4: initial body/throat segmentation\n');
[throat_body,throat_center,neighbour,locals] = ...
    step4_throat_body(locals,voxels,throat_center,vox3d,neighbour,numBodyLabels);
present = unique(throat_body(throat_body > 0));
% Helpers assume contiguous body/throat labels and one throat row per label.
if numel(locals) ~= numBodyLabels || ...
        ~isequal(double(present(:)),(1:(numBodyLabels+size(throat_center,1)))')
    error('PoreSpace:Labels','Empty or noncontiguous Step-4 labels; refinement mapping requires review.');
end
surface_idx = step4_throat_body_surface(reshape(throat_body,sz),neighbour);
if isempty(surface_idx)
    throat_body_surface = zeros(0,4);
else
    throat_body_surface = [voxels(surface_idx(:,1),:),surface_idx(:,2)];
end
if options.SaveIntermediate
    save(fullfile(outputDir,'step4.mat'),'throat_body','throat_center','locals', ...
        'voxels','numBodyLabels','throat_body_surface','res','sz','-v7.3');
end
fprintf('Step 5: geometric refinement\n');
if isempty(throat_center)
    labels = throat_body; throat_center_new = zeros(0,8); locals_new = locals;
    diagnostics = struct('total_throats',0,'accepted_proposals',0,'errors',{{}});
else
    [labels,throat_center_new,locals_new,diagnostics] = ...
        step5_refine(throat_body,voxels,numBodyLabels,throat_center,res,locals, ...
        throat_body_surface,2,2,0,options.UseParallel);
end
labelled_output = reshape(labels,sz);
if ~isequal(labelled_output > 0,vox3d)
    error('PoreSpace:Conservation','Refinement changed the solid/pore mask.');
end
failed = find(~cellfun(@isempty,diagnostics.errors));
if ~isempty(failed)
    warning('PoreSpace:Skipped','%d throats failed refinement; see summary.json diagnostics.',numel(failed));
end
num_pore_body = numel(unique(labels(labels > 0 & labels <= numBodyLabels)));
num_throat = numel(unique(labels(labels > numBodyLabels)));
summary = struct('input_file',char(inputFile),'shape_xyz',sz, ...
    'options',options,'num_pore_body',num_pore_body,'num_throat',num_throat, ...
    'body_label_upper_bound',numBodyLabels,'pore_voxels',nnz(vox3d), ...
    'diagnostics',diagnostics,'matlab_version',version,'elapsed_seconds',toc(started));
save(fullfile(outputDir,'result.mat'),'labelled_output','num_pore_body', ...
    'num_throat','numBodyLabels','throat_center_new','locals_new', ...
    'constriction_length','summary','-v7.3');
export_pore_features(labelled_output,numBodyLabels,res,options.D50,outputDir);
fid = fopen(fullfile(outputDir,'summary.json'),'w');
if fid < 0, error('PoreSpace:Write','Cannot write summary.json'); end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(summary,PrettyPrint=true));
result = struct('labelled_output',labelled_output,'summary',summary);
end
