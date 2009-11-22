

%
% Stupid thing to read a matfile
% play it with en
% Save it to disk



function play_en_wrapper(inFile, outFile) 
%
% inFile should contain M
% outFile will contain signal
    
    load -ascii inFile
    
    signal = play_en(M)
    
    save outFile signal -ascii