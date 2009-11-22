function D = load_TK_metadata(TK,path)
% D = load_TK_metadata(TK,path)
%   Read back the data array containing the metadata for the
%   specified TK.
% 2009-05-21 DAn Ellis dpwe@ee.columbia.edu

if nargin < 2; path = 'meta/%s/%s'; end

matpath = [sprintf(path,TK(3),TK(4)),'/',TK,'.mat'];
if exist(matpath,'file')
  d = load(matpath);
else
  d.data = [];
  disp(['Warn: no metadata file for ',TK]);
end

%D.dummy = [];   % needed to use setfield !?
D.duration = 0;  % guarantee this field
D.title = '';    % and this one

%D = d; return;

% convert cell array into structure
[nr,nc] = size(d.data);

for r = 1:nr
  f = d.data{r,1};
  % remove colon
  f = f(f~=':');
  if length(f) > 0  % some null entries???
    % grab value
    v = d.data{r,2};
    % can we convert it to a number?
    [vn,c] = sscanf(v,'%f');
    if c == 0
      % it was a string
      D = setfield(D,f,v);
    else
      % we got a number
      D = setfield(D,f,vn);
    end
  else
    disp(['Warn: empty metadata field in ',TK]);
  end
end

%D = rmfield(D,'dummy');
