function D = combineUniqueSessions(D, D_)
% D = combineUniqueSessions(D1, D2)
% combine two super session structs

maxUnitId = max(D.spikeIds);
maxSessNum = max(D.sessNumGratings);
maxTime = max(D.spikeTimes);

fields = fieldnames(D_);

timingFields = {'GratingOnsets', 'GratingOffsets', 'eyeTime', 'frameTimes', 'spikeTimes', 'treadTime'};

D_.spikeIds = D_.spikeIds + double(maxUnitId);
D.units = [D.units D_.units];

fields = setdiff(fields, {'units'});
nfields = numel(fields);

for ifield = 1:nfields
    field = fields{ifield};
    
    if ismember(field, timingFields)
        D_.(field) = D_.(field) + maxTime;
    end

    if contains(field, 'sessNum')
        D_.(field) = D_.(field) + maxSessNum;
    end
    
    D.(field) = [D.(field); D_.(field)];
end