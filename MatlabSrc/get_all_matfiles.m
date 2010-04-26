

function fnames = get_all_matfiles(d)
    % Returns all matfiles filenames contains in d
    % and its subdirectories
    % EXAMPLE: get_all_matfiles('.')
    if nargin < 1
        d = '.';
    end
    pth=fullfile(d,'*.mat');
    fnames=filefun(pth,Inf);
