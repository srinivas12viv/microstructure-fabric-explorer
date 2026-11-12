function export_pore_features(labels,bodyMax,spacing,d50,outputDir)
% Toolkit integration by Srinivas Vivek Bokkisa; physical XYZ coordinates.
ids = unique(labels(labels>0));
values = zeros(numel(ids),12);
ambiguous = false(numel(ids),1);
for k = 1:numel(ids)
    idx = find(labels == ids(k));
    [x,y,z] = ind2sub(size(labels),idx);
    xyz = ([x,y,z] - 0.5)*spacing;
    c = mean(xyz,1);
    centered = xyz-c;
    [vectors,D] = eig((centered'*centered)/size(xyz,1),'vector');
    [eigenvalues,order] = sort(D,'descend');
    axis = vectors(:,order(1))';
    [~,j] = max(abs(axis));
    if axis(j)<0, axis=-axis; end
    volume = numel(idx)*spacing^3;
    diameter = (6*volume/pi)^(1/3);
    ambiguous(k) = eigenvalues(1)-eigenvalues(2) <= 1e-6*max(eigenvalues(1),eps);
    values(k,:) = [ids(k),volume,diameter,diameter/d50,c,axis,eigenvalues(1:2)'];
end
T = array2table(values,'VariableNames',{'id','volume','equivalent_diameter', ...
    'diameter_over_D50','cx','cy','cz','nx','ny','nz','eigenvalue_1','eigenvalue_2'});
T.axis_ambiguous = ambiguous;
for kind = ["bodies","throats"]
    keep = ids <= bodyMax;
    if kind == "throats", keep = ~keep; end
    writetable(T(keep,:),fullfile(outputDir,'pore_'+kind+'.csv'));
    fid = fopen(fullfile(outputDir,kind+'_orientations.txt'),'w');
    if fid < 0, error('PoreSpace:Write','Cannot write orientations'); end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'nx ny nz f\n');
    fprintf(fid,'%.12g %.12g %.12g 1\n',values(keep,8:10)');
    clear cleanup
end
end
