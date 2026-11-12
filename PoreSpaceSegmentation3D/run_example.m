function result = run_example(exampleName, options)
%RUN_EXAMPLE Run VEP or SCP through the same MATLAB implementation.
arguments
    exampleName (1,1) string {mustBeMember(exampleName,["VEP","SCP"])} = "VEP"
    options.UseParallel (1,1) logical = false
    options.SaveIntermediate (1,1) logical = false
end
root = fileparts(mfilename('fullpath'));
result = poreSpaceSegmentation3D(fullfile(root,'examples',exampleName,'input.mat'), ...
    fullfile(root,'outputs',exampleName),VoxelSize=1,MeanParticleDiameter=40, ...
    D50=40,Units="voxel",UseParallel=options.UseParallel, ...
    SaveIntermediate=options.SaveIntermediate);
end
