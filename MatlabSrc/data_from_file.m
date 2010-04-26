

function splits = data_from_file(fname)
    % Returns every 4 beats of the btchroma matrix
    
    % load data
    data = load(fname);
    btchroma = data.btchroma;
    % truncate
    truncsize = size(btchroma,2) - mod(size(btchroma,2),4);
    nsplits = truncsize / 4;
    % split
    splits = subdivide(btchroma(:,1:truncsize),12,4);
    % compress the result
    splits = reshape(splits,12,4,nsplits);
    % rehsape
    splits = reshape(splits,12 * 4,nsplits)';