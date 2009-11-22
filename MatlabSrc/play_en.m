function x = play_en(M,dur,sr)
% x = play_en(M,dur,sr)
%    Resynthesize audio from an EN analyze structure.
%    M is an EN segment structure e.g. from read_en_segs
%    x is returned as a waveform synthesized from that data, with
%    max duration <dur> secs (duration of song), at sampling rate
%    sr (16000 Hz).
% 2009-03-11 Dan Ellis dpwe@ee.columbia.edu

if nargin < 2; dur = 0; end
if nargin < 3; sr = 16000; end

[nchr, nbeats] = size(M.pitches);

beattimes = [0,cumsum(M.dur)];
% include denormalization by loudness
C = M.pitches .* repmat(idB(M.loudness),nchr,1);

if dur > 0
  nbeats = sum(beattimes <= dur);
  beattimes = beattimes(1:nbeats);
  C = C(:,1:nbeats);
end

dowt = 1;
maxnpitch = 4;

x = chromsynth2(C,beattimes,sr,dowt,maxnpitch);

%%%%% PUT ENVELOPE RECONSTRUCTION FROM TIMBRE FEATURES IN HERE %%%%%%

if nargout == 0
  soundsc(x,sr);
end
