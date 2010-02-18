function x = play_en2(F,dur,sr)
% x = play_en2(F,dur,sr)
%    Resynthesize audio from an EN analyze structure.
%    F is a matrix fo chromas, one beat per column
%    x is returned as a waveform synthesized from that data, with
%    max duration <dur> secs (duration of song), at sampling rate
%    sr (16000 Hz).
% 2009-03-11 Dan Ellis dpwe@ee.columbia.edu
% 2010-02-18 modified T. Bertin-Mahieux tb2332@columbia.edu

if nargin < 3; sr = 16000; end

[nchr, nbeats] = size(F);
if nargin < 2; dur = nbeats; end

loudness = 1
beattimes = 0:nbeats;
% include denormalization by loudness
%C = F .* repmat(idB(loudness),nchr,1);

%if dur > 0
%  nbeats = sum(beattimes <= dur);
%  beattimes = beattimes(1:nbeats);
%  C = C(:,1:nbeats);
%end

C = F
dowt = 1;
maxnpitch = 4;

x = chromsynth2(C,beattimes,sr,dowt,maxnpitch);

%%%%% PUT ENVELOPE RECONSTRUCTION FROM TIMBRE FEATURES IN HERE %%%%%%

if nargout == 0
  soundsc(x,sr);
end
