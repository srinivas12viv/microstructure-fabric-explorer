% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [p1,p2,p3,p4,dbg] = step5gh_correct_inflexion_robust( ...
    sizy, sizx, ...
    firstqci, secondqci, thirdqci, fourthqci, ...
    firstqtall, secondqtall, thirdqtall, fourthqtall, ...
    firstqti, secondqti, thirdqti, fourthqti, ...
    firstqcall, secondqcall, thirdqcall, fourthqcall, ...
    store_new, throat_axis_center, vis)
%STEP5GH_CORRECT_INFLEXION_ROBUST (DROP-IN REPLACEMENT)
%
% Chooses correct inflection/knee points per quadrant using:
%   - throat candidate (from qti / step5gh_correct_inflexion_c1)
%   - combined candidate (from qci; simple robust scan)
% Decision throat vs combined when both exist:
%   (1) if throat & combined are very close -> choose throat
%   (2) else use chord+trend bulge criterion on smoothed throat boundary store_new
%       - bulge confirmed -> choose throat
%       - no bulge confirmed -> choose combined
%       - ambiguous -> conservative throat
%       - NEW: if knee is too close to vertical reference point -> ignore bulge inference,
%              prefer combined (if available)
%
% Output flags:
%   0 -> throat point
%   1 -> combined point
%
% ASSUMPTION: store_new is a SMOOTHED, ORDERED throat boundary polyline in (u,v).
% If store_new is not ordered, the trend segment logic must be revised.

dbg = struct();

% -------------------- robust scaling --------------------
u0 = throat_axis_center(1,1);
v0 = throat_axis_center(1,2);

halfW = median(abs(store_new(:,2) - v0));
halfW = max(halfW, 1e-6);

% Proximity => "same feature" (short throat)
tolClose = 0.1*halfW; %tolClose = 0.40 * halfW

% --- Parameters for bulge decision (new) ---
params = struct();
params.du_band   = 0.25 * halfW;    % band around u0 to find vertical reference
params.pct_hi    = 95;              % robust percentile instead of max
params.pct_lo    = 5;

params.dv_min    = 0.1*halfW;    % min outward change to call chord "bulge"
params.tau       = 0.05 * halfW;    % noise floor for trend increments
params.r_bulge   = 0.35;            % fraction positive increments => bulge
params.r_nobulge = 0.1;            % fraction positive increments => no bulge
params.win_med   = 5;               % movmedian window for outward(v)
params.require_turn = true;        % optional; enables "inward dive then contraction" pattern

% --- NEW: ignore bulge inference if knee too close to vertical reference ---
params.min_du_refk = 0.10 * halfW;  % minimum |uk-u0| to trust chord/trend
params.min_d_refk  = 0.15 * halfW;  % minimum distance pv->ptT to trust bulge

% --- Combined candidate scan parameters (simple + robust) ---
u_margin      = 0.50 * halfW;       % must extend beyond this along u to be "long"
tolNearThroat = 50.0 * halfW;       % must be near throat boundary (throat-like)
vBand         = 2.5 * halfW;       % stay near axis band

% -------------------- per-quadrant selection --------------------
[p1,dbg.q1] = pick_one_long_new('first',  firstqti,  firstqci,  firstqtall,  fourthqtall, firstqcall,  store_new, throat_axis_center, ...
                                tolClose, u_margin, tolNearThroat, vBand, params);

[p2,dbg.q2] = pick_one_long_new('second', secondqti, secondqci, secondqtall, thirdqtall,  secondqcall, store_new, throat_axis_center, ...
                                tolClose, u_margin, tolNearThroat, vBand, params);

[p3,dbg.q3] = pick_one_long_new('third',  thirdqti,  thirdqci,  thirdqtall,  secondqtall, thirdqcall,  store_new, throat_axis_center, ...
                                tolClose, u_margin, tolNearThroat, vBand, params);

[p4,dbg.q4] = pick_one_long_new('fourth', fourthqti, fourthqci, fourthqtall, firstqtall,  fourthqcall, store_new, throat_axis_center, ...
                                tolClose, u_margin, tolNearThroat, vBand, params);

if vis == 1
    disp('Chosen points [u v flag]:'); disp([p1;p2;p3;p4]);
    disp('Debug (q1):'); disp(dbg.q1);
    disp('Debug (q2):'); disp(dbg.q2);
    disp('Debug (q3):'); disp(dbg.q3);
    disp('Debug (q4):'); disp(dbg.q4);
end

end


% ============================================================
% Helper: choose one quadrant (updated throat vs combined decision)
% ============================================================
function [point,info] = pick_one_long_new(name, qti, qci, qtall, qtall_neighbor, qcall, store_new, throat_axis_center, ...
                                          tolClose, u_margin, tolNearThroat, vBand, params)

info = struct();
info.haveT = ~isempty(qti);
info.haveC = ~isempty(qci);

% ---- Throat candidate ----
ptT = [];
if ~isempty(qti)
    ptT = pick_throat_candidate(qti, tolClose, name); % [u v 0]
end

% ---- Combined candidate (simple best) ----
ptC = [];
bestC = [];
if ~isempty(qci)
    [ptC, bestC] = pick_best_combined_simple(qci, store_new, throat_axis_center, name, ...
                                             u_margin, tolNearThroat, vBand);
end
info.bestCombined = bestC;

% ------------------ existence cases ------------------
if ~isempty(ptT) && isempty(ptC)
    point = ptT;
    info.picked = 'throat_only_or_no_valid_combined';
    return;
end

if isempty(ptT) && ~isempty(ptC)
    point = ptC;
    info.picked = 'combined_only';
    return;
end

if isempty(ptT) && isempty(ptC)
    point = fallback_extreme(name, qtall, qtall_neighbor);
    info.picked = 'fallback_none';
    return;
end

% ------------------ both exist: decide ------------------
% If essentially same feature -> keep throat (short throat)
dTC = hypot(ptT(1)-ptC(1), ptT(2)-ptC(2));
info.dTC = dTC;
if dTC <= tolClose
    point = ptT;
    info.picked = 'throat_close';
    return;
end

% Use chord+trend bulge criterion based on THROAT knee ptT and store_new
[pickThroat, bulgeDbg] = decide_by_chord_and_trend(store_new, throat_axis_center, ptT(1:2), params);
info.bulge = bulgeDbg;

if pickThroat
    point = ptT;
    info.picked = 'throat_bulge_confirmed';
else
    point = ptC;
    info.picked = 'combined_no_bulge_confirmed';
end

end


% ============================================================
% Pick throat candidate (unchanged intent)
% ============================================================
function pt = pick_throat_candidate(Q, tol, name)
Q = double(Q(:,1:2));
if size(Q,1) == 1
    pt = [Q(1,:) 0];
    return;
end
pt0 = step5gh_correct_inflexion_c1(Q, tol, name); % returns [u v 0] or [u v ...]
pt0(1,3) = 0;
pt = pt0;
end


% ============================================================
% Pick best combined candidate (simple: maximize axis extension, keep throat-like)
% ============================================================
function [ptC, bestInfo] = pick_best_combined_simple(qci, store_new, throat_axis_center, name, ...
                                                     u_margin, tolNearThroat, vBand)

ptC = [];
bestScore = -inf;

bestInfo = struct('candidate',[],'dToThroat',NaN,'dv',NaN,'du',NaN,'score',NaN,'reason','');

Q = double(qci(:,1:2));
if isempty(Q)
    bestInfo.reason = 'qci_empty';
    return;
end

u0 = throat_axis_center(1,1);
v0 = throat_axis_center(1,2);
isRight = ismember(name, {'first','fourth'});

for k = 1:size(Q,1)
    pc = Q(k,:);

    dv = abs(pc(2) - v0);
    if dv > vBand, continue; end

    dToTh = min(vecnorm(store_new(:,1:2) - pc, 2, 2));
    if dToTh > tolNearThroat, continue; end

    du = pc(1) - u0;
    if isRight
        if du <= u_margin, continue; end
    else
        if du >= -u_margin, continue; end
    end

    % Score: prioritize axis extension; small penalties
    score = abs(du) - 0.15*dv - 0.05*dToTh;

    if score > bestScore
        bestScore = score;
        ptC = [pc 1];

        bestInfo.candidate = pc;
        bestInfo.dToThroat = dToTh;
        bestInfo.dv = dv;
        bestInfo.du = du;
        bestInfo.score = score;
        bestInfo.reason = 'accepted_best';
    end
end

if isempty(ptC)
    bestInfo.reason = 'no_candidate_passed_filters';
end

end


% ============================================================
% Decide throat vs combined using chord + trend on store_new
% (NEW: ignore if knee too close to vertical reference)
% ============================================================
function [pickThroat, dbg] = decide_by_chord_and_trend(store_new, throat_axis_center, ptT_uv, params)
% Decide throat vs combined using DEVIATION magnitude (outward OR inward) + coherence,
% evaluated on the SHORTER periodic arc between pv and knee neighborhood.
%
% This addresses cases where a "long throat into smaller pore body" produces a strong inward dive
% (outward decreases) which should still be classified as throat-like.

u0 = throat_axis_center(1,1);
v0 = throat_axis_center(1,2);

uk = ptT_uv(1);
vk = ptT_uv(2);
isUpper = (vk >= v0);

% outward transform (so outward distance from axis is monotone w.r.t. v sign)
if isUpper
    outward = @(v) v;
else
    outward = @(v) -v;
end

% --- Ensure we work on a non-duplicated closed loop (remove repeated closure point) ---
S = store_new(:,1:2);
if size(S,1) >= 2 && hypot(S(1,1)-S(end,1), S(1,2)-S(end,2)) < 1e-12
    S = S(1:end-1,:);
end
N = size(S,1);

% ---- find vertical reference point pv on S near u0 (robust percentile) ----
du = abs(S(:,1) - u0);
near = du <= params.du_band;

if nnz(near) < 5
    [~, idxSort] = sort(du, 'ascend');
    idx = idxSort(1:min(50, numel(idxSort)));
else
    idx = find(near);
end

vCandidates = S(idx,2);
if isUpper
    vv = prctile(vCandidates, params.pct_hi);
else
    vv = prctile(vCandidates, params.pct_lo);
end
pv = [u0, vv];

% ---- baseline sanity: ignore inference if knee too close to pv ----
du_refk = abs(uk - u0);
d_refk  = hypot(uk - pv(1), vk - pv(2));
tooClose = (du_refk < params.min_du_refk) || (d_refk < params.min_d_refk);

% Instead of forcing combined, mark ambiguous and let caller decide if you prefer:
% here: keep the prior behavior (prefer combined) under the supplied selection policy.
if tooClose
    pickThroat = false;  % same behavior as the existing code
    dbg = struct();
    dbg.isUpper = isUpper;
    dbg.pv = pv;
    dbg.ptT = ptT_uv;
    dbg.du_refk = du_refk;
    dbg.d_refk  = d_refk;
    dbg.tooClose = true;
    dbg.ambiguous = true;
    dbg.reason = 'ignored_knee_too_close_to_reference_prefer_combined';
    return;
end

% ---- chord DEVIATION test (magnitude) ----
dv_signed = outward(vk) - outward(vv);
dv_mag    = abs(dv_signed);

chordDeviates = dv_mag > params.dv_min;      % "something significant happened"
chordTiny     = dv_mag < 0.5*params.dv_min;  % clearly small change (optional)

% ---- indices on S nearest to ptT and pv ----
[~, ik] = min(vecnorm(S - ptT_uv, 2, 2));
[~, ir] = min(vecnorm(S - pv,     2, 2));

% ---- choose SHORTER periodic arc between ir and ik ----
idxPath = shorter_circular_path(ir, ik, N);

segV = S(idxPath, 2);
w   = outward(segV);

% Smooth outward(v) slightly to suppress jaggedness
w_s = movmedian(w, params.win_med);

dw = diff(w_s);

% Coherence of deviation: fraction of steps with "meaningful change" (in either direction)
rStrong = mean(abs(dw) > params.tau);

% Optional: detect a clear turning point (for "inward dive then contraction" shapes)
% This checks whether the segment contains a pronounced local extremum away from both ends.
turnOK = true;
if isfield(params,'require_turn') && params.require_turn
    % A turning point exists if there is at least one sign change in dw
    % AND the extremum is not at endpoints.
    sgn = sign(dw);
    sgn(sgn==0) = 1; % treat flats as positive to avoid artificial sign changes
    nFlip = sum(sgn(1:end-1).*sgn(2:end) < 0);
    [~, iMin] = min(w_s);
    [~, iMax] = max(w_s);
    interiorMin = (iMin > 2) && (iMin < numel(w_s)-1);
    interiorMax = (iMax > 2) && (iMax < numel(w_s)-1);
    turnOK = (nFlip >= 1) && (interiorMin || interiorMax);
end

trendDeviates = (rStrong >= params.r_bulge);     % reuse r_bulge as "strong deviation fraction"
trendCalm     = (rStrong <= params.r_nobulge);   % reuse r_nobulge as "calm fraction"

% ---- combine logic (magnitude-based) ----
ambiguous = false;

if chordDeviates && trendDeviates && turnOK
    pickThroat = true;     % significant, coherent deviation (either outward or inward)
elseif chordTiny && trendCalm
    pickThroat = false;    % essentially no deviation -> likely combined
else
    % Mixed / ambiguous: conservative throat is usually safer for not losing throats
    pickThroat = true;
    ambiguous = true;
end

dbg = struct();
dbg.isUpper = isUpper;
dbg.pv = pv;
dbg.ptT = ptT_uv;
dbg.du_refk = du_refk;
dbg.d_refk  = d_refk;
dbg.tooClose = false;

dbg.dv_signed = dv_signed;
dbg.dv_mag = dv_mag;
dbg.chordDeviates = chordDeviates;
dbg.chordTiny = chordTiny;

dbg.rStrong = rStrong;
dbg.trendDeviates = trendDeviates;
dbg.trendCalm = trendCalm;

dbg.turnOK = turnOK;
dbg.ambiguous = ambiguous;
dbg.reason = 'magnitude_deviation_short_arc';

dbg.ir = ir; dbg.ik = ik;
dbg.pathLen = numel(idxPath);
end

% -------- helper: choose shorter path on circular index set 1..N --------
function idxPath = shorter_circular_path(i1, i2, N)
% Returns indices along the shorter arc from i1 to i2 (inclusive) on a circular list.

if i1 <= i2
    pathA = i1:i2;
    pathB = [i2:N, 1:i1];      % reverse wrap path (i2 -> ... -> i1)
else
    pathA = i2:i1;
    pathB = [i1:N, 1:i2];
end

% We want a path that goes from i1 to i2; build both directed paths explicitly:
idxFwd = circular_forward(i1, i2, N);
idxBwd = circular_forward(i2, i1, N); % opposite direction, then reverse to go i1->i2
idxBwd = flipud(idxBwd(:));

if numel(idxFwd) <= numel(idxBwd)
    idxPath = idxFwd(:);
else
    idxPath = idxBwd(:);
end
end

function idx = circular_forward(iFrom, iTo, N)
% forward indices from iFrom to iTo on circle
if iFrom <= iTo
    idx = (iFrom:iTo).';
else
    idx = [ (iFrom:N)'; (1:iTo)' ];
end
end



% ============================================================
% Fallback: choose an extreme point (flag=0)
% ============================================================
function point = fallback_extreme(name, qtall, qtall_neighbor)
P = qtall;
if isempty(P), P = qtall_neighbor; end
if isempty(P)
    point = [NaN NaN 0];
    return;
end

switch name
    case {'first','second'}
        [~,idx] = max(P(:,2)); % top
    case {'third','fourth'}
        [~,idx] = min(P(:,2)); % bottom
    otherwise
        [~,idx] = max(P(:,2));
end

point = [P(idx,1:2) 0];
end
