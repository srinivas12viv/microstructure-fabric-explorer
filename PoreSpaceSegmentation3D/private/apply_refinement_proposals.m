function vol = apply_refinement_proposals(vol_ref,props,prop_ok)
% Submitted serial merge policy: ascending throat row; later proposals win.
% Body-to-throat reassignment is permitted. Solid voxels cannot be relabeled.
vol = vol_ref;
for k = 1:numel(props)
    if ~prop_ok(k), continue; end
    p = props{k};
    if isempty(p) || ~isfield(p,'valid') || ~p.valid || ~isfield(p,'ind') || isempty(p.ind)
        continue;
    end
    ind = p.ind(:);
    lab = p.lab(:);
    if numel(ind) ~= numel(lab) || any(ind < 1 | ind > numel(vol_ref)) ...
            || any(ind ~= fix(ind)) || any(lab < 1 | lab ~= fix(lab)) ...
            || any(vol_ref(ind) == 0)
        error('PoreSpace:Proposal','Invalid proposal indices/labels or attempted solid reassignment.');
    end
    vol(ind) = lab;
end
end
