


function data = load_all_data(d)
    % d is the base directory
    
    % get matfiles
    fnames = get_all_matfiles(d);
    
    % load all data
    %alldata = filefun(@data_from_file,fnames);
    alldata = cellfun(@data_from_file,fnames,'UniformOutput',false);
    
    % concatenate, done
    data = cat(1,alldata{:});