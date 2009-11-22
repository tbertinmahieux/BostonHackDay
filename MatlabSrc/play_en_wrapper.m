

%
% Stupid thing to read a matfile
% play it with en
% Save it to disk



function play_en_wrapper(inFile, outFile) 
%
% inFile should contain M
% outFile will contain signal
    
    if nargin < 1
        inFile = './dummy_playenwrapperpy_infile.mat'
    end
    if nargin < 2
        outFile = './dummy_playenwrapperpy_outfile.mat'
    end
    
    disp('loading')
    
    load(inFile)
    
    disp('play en')
    
    signal = play_en(M);
    
    disp('saving')
    
    %save(outFile,'signal','-ascii')
    save(outFile,'signal')
    
    disp('done')
    
    
