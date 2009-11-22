

%
% Stupid thing to read a matfile
% play it with en
% Save it to disk



function play_en_wrapper(inFile, outFile, wavfile) 
%
% inFile should contain M
% outFile will contain signal

    if nargin < 1
        inFile = './dummy_playenwrapperpy_infile.mat'
    end
    if nargin < 2
        outFile = './dummy_playenwrapperpy_outfile.mat'
    end
    if nargin < 3
        wavfile = './dummy_wavfile.wav'
    end
    
    inFile2 = './dummy_playenwrapperpy_infile2.mat'
    
    disp('loading')
    
    load(inFile)
    
    disp('play en')
    
    signal = play_en(M);
    
    disp('saving to .mat')
    
    %save(outFile,'signal','-ascii')
    save(outFile,'signal')

    disp('cutting wav file')
    load(inFile2)
    pos1 = floor(starttime * 22050) + 1;
    pos2 = ceil(stoptime * 22050)+1;
    disp(pos1)
    disp(pos2)
    if pos2 > len(signal)
        pos2 = len(signal)
    end
    signal = signal(pos1:pos2);
    
    disp('saving to .wav')
    wavwrite(signal,22050,wavfile)
    
    disp('done')
    
    
