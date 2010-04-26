function varargout=filefun(varargin)
%FILEFUN Apply a function to files
%   usage: [A,B,...,{FLIST}] = FILEFUN({FUN,}FILES{,DEPTH})
%   usage: [A, ...] = FILEFUN(..., 'param1', value1, ...)
%
%   This function is loosely modelled after the standard CELLFUN, ARRAYFUN,
%   STRUCTFUN and SPFUN functions. It applies a given function to all files
%   in FILES and returns the concatenated output. FILES can be a char or a
%   cellstr and can have wildcards (*) in the filename and extension parts.
%   If DEPTH is greater than the default zero, FILEFUN will recurse
%   subdirectories to specified depth. By default, the last output
%   parameter holds the found filenames.
%
%   [A, B,...] = FILEFUN(FUN,FILES, ...) evaluates FUN, which is a function
%   handle to a function that returns multiple outputs, and returns arrays
%   A, B, ..., each corresponding to one of the output arguments of FUN.
%   FILEFUN calls FUN each time with as many outputs as there are in the
%   call to FILEFUN, except for a possible FLIST argument.
%
%   FLIST = FILEFUN(FILES,...) creates a list of files matching
%   FILES, without any function being called.
%
%   [...] = FILEFUN(...,DEPTH,...) traverses subdirectories matching
%   FILES recursively down to a level of DEPTH. Default is no
%   recursion, DEPTH=0.
%
%   [...] = FILEFUN(..., 'Param1', val1, ...) enables you to specify
%   optional parameter name/value pairs. Properties may be in any order,
%   and may be shortened as long as that shortening is unambiguous, down to
%   a single letter. Capitalization is ignored. Parameters are:
%
%      'Sort' -- a logical value, default true, indicating whether the file
%      list should be sorted or not. This will take place prior to any calls
%      to FUN.
%
%      'List' -- a logical value, default FALSE, indicating whether or the
%      last output parameter should be populated with the list of files
%      handled.
%
%      'LexicalSort' -- a logical value, default true, indicating whether
%      or not the file list should be sorted lexically or not. If true,
%      'FooBar' and 'foobar' will be sorted as being the same.
%
%      'Args' -- a cell array containing additional arguments to FUN.
%      FUN will be called like: FUN(FILENAME, Args{:}).
%
%      'UniformOutput' -- an integer or logical value indicating whether or
%      not the output(s) of FUN can be returned without encapsulation in a
%      cell array. If integer greater than zero, FUN must return constant
%      sized arrays, which will be concatenated along the dimension indi-
%      cated by the parameter. It is possible to specify a dimension
%      exceeding the number of dimensions of the output. If logical true,
%      the default, the output is concatenated along the first singleton
%      dimension, possibly (NDIMS+1). If the parameter is zero or false,
%      FILEFUN returns a cell array (or multiple cell arrays), where the
%      i:th cell contains the value FUN(FILES{i},...).
%
%      Note that this behaviour of 'UniformOutput' is somewhat different
%      from that of CELLSTR et.al.
%
%      'CheckExistence'  --  a logical value, specifying whether FILEFUN
%      shall check that the file exists. If CheckExistance is true and the
%      file does not exist, it is silently ignored. If false, no check is
%      made and FUN is called with the given filename. Default value is
%      true.
%
%      'ErrorHandler'	--	a function handle, specifying the function
%      MATLAB is to call if the call to FUN fails.   The error handling
%      function will be called with the following input arguments:
%          -   a structure, with the fields:  "identifier", "message",
%          respectively containing the identifier of the error that
%          occurred and the text of the error message. 
%          -   the set of input arguments at which the call to the 
%          function failed.
%
%      The error handling function should either rethrow an error, or
%      return the same number of outputs as FUN.  These outputs are then
%      returned as the outputs of FILEFUN.  If 'UniformOutput' is true,
%      the outputs of the error handler must be of the same type and size
%      as the outputs of FUN. Example:
%
%      function [A, B] = errorFunc(S, varargin)
%          warning(S.identifier, S.message); A = NaN; B = NaN;
%
%      If an error handler is not specified, the error from the call to 
%      FUN will be rethrown.
%
%   Examples:
%
%      Some of the following examples are a bit artificial, since they are
%      limited to using only functions and files included in the standard
%      MATLAB distribution. Normally, you have written a function that
%      takes a filename as a parameter and now you want to apply it to a
%      whole bunch of files.
%
%      Get a list of all m-files in current directory:
%
%         fl=filefun('*.m')
%
%      Get a list of all m-files in the matlab uitools directory and its
%      subdirectories:
%
%         pth=fullfile(matlabroot,'toolbox','matlab','uitools','*.m');
%         fl=filefun(pth,Inf)
%
%      Calculate the total size of above files:
%
%         pth=fullfile(matlabroot,'toolbox','matlab','uitools','*.m');
%         d=filefun(@dir,pth,inf);
%         sum([d.bytes])
%
%      Read the first line of every txt-file under matlabroot sys.
%
%         pth=fullfile(matlabroot,'sys','*.txt');
%         lines=filefun(@textread,pth,Inf, ...
%                         'Args',{'%s',1,'whitespace','\n\r'})
%
%      Pick a couple of image files yourself with SHIFT- or CTRL-click (in
%      Windows...), and get their mean RGB values. Keep the selected order.
%
%         pth=fullfile(matlabroot,'toolbox','matlab','demos','html',filesep);
%         [filenames,pathname]=uigetfile({'*.png'},'Pick files',pth,'MultiSelect', 'on');
%         fun=@(fn) squeeze(mean(mean(imread(fn))))';
%         [mn,flist]=filefun(fun,strcat(pathname,filenames),'Sort',false)
%
%   See also: CELLFUN, ARRAYFUN, STRUCTFUN, SPFUN, DIR, GENPATH, UIGETFILE

%   Version: 1.1 2006-01-23
%   Author: Jerker Wågberg, More Research, SWEDEN
%   email: char(hex2dec(reshape('6A65726B65722E77616762657267406D6F72652E7365',2,[])')')
%
%   Yes, I know... My parents didn't make it easy for me to do an 
%   international career with a first name like this. Sigh...
%
%   ver 1.0 Initial release to File Exchange
%
%   ver 1.1
%      Fixed erroneous NaN option of UniformOutput.
%      Usage of NaN is discouraged.
%      Instead, UniformOutput can now be entered as a
%      logical, making it congruent with CELLFUN et.al.
%      Path can now be entered as [], as well as ''.
%      Minor changes in comments.
%
%   ver 1.2
%      Change in comment. As an old hand at DOS, I have up
%      to now ASSUMED that the MATLAB dir command supported ? as well as *.
%
%   ver 1.3
%      Gives better error info when no error handling routine is specified
%
%   ver 1.4
%      Removed the PATH parameter. It was there only for UIGETFILE and as
%      it turns out, it is seldom used and makes the calling sequence
%      cumbersome and less intuitive.
%      Made the List parameter false by default if a function handle is
%      used as first argument.

	[fun,pth,depth,par,msg]=parseparams(varargin);
	error(msg);
	nargso=max(0,nargout-par.List);
	
	% Resolve wildcards and recursion
	if par.CheckExistence
		pth=recurse(pth(:),depth);
		end
		
	% Possibly sort the paths
	if par.Sort
		if par.LexicalSort
			[qq,ix]=sort(lower(pth));
			pth=pth(ix);
		else
			pth=sort(pth);
			end
		end
	
	nf=length(pth);
	if ~isempty(fun)
	
		% Apply the function to all files
		argout=cell(nf,nargso);
		for i=1:nf
			if ~isempty(par.ErrorHandler)
				try
					[argout{i,:}]=fun(pth{i},par.Args{:});
				catch
					err=lasterror;
					forward.message = err.message;
					forward.identifier = err.identifier;
					[argout{i,:}] = par.ErrorHandler(forward, pth{i},par.Args{:});
					end
			else
				[argout{i,:}]=fun(pth{i},par.Args{:});
				end
			end

		% Distribute into callers variables
		varargout=mat2cell(argout,nf,ones(1,nargso));
		if par.UniformOutput
			for i=1:nargso
				dim=1;
				if ~isempty(varargout{i}{1})
					sz=size(varargout{i}{1});
					for j=2:size(varargout{i},1)
						if ~isequal(sz,size(varargout{i}{j}))
							error('Output is not uniform');
							end
						end
					if islogical(par.UniformOutput)
						dim=find([sz 1]==1,1);
					else
						dim=par.UniformOutput;
						end
					end
				varargout{i} = cat(dim,varargout{:,i}{:});
				end
			end
			
		% Possibly append the path list
		if par.List && nargout
			varargout{nargout}=pth;
			end
	else
		if par.List
			varargout{1}=pth;
			end;
		end

function [fun,files,depth,params,msg]=parseparams(argin)
%PARSEPARAMS Parse the input
	fun=[];
	files={'*'};
	depth=0;
	params=struct( ...
				  'UniformOutput', true ...
				, 'ErrorHandler', [] ...
				, 'List', true ...
				, 'Args', {{}} ...
				, 'Sort', true ...
				, 'LexicalSort', true ...
				, 'CheckExistence', true ...
				);
	msg=[];
	nargs=length(argin);
	em=struct( ...
		    'IllChar'	, '''FILES'' must be char or cellstr' ...
		  , 'IllArg'	, 'Named parameters must start with a string' ...
		  );
	base=1;
	if 0<=nargs-base	% Filenames argument
		if isa(argin{1}, 'function_handle')
			fun=argin{1};
			params.List=false;
			base=base+1;
			end
		end

	if 0<=nargs-base	% Filenames argument
		if ~isstring(argin{base})
			msg=em.IllChar;
			return;
			end
		files=cellstr(argin{base});
		base=base+1;
		end

	if 0<=nargs-base	% Depth argument
		if isnumeric(argin{base})
			depth=argin{base};
			base=base+1;
			end
		end

	if 0<=nargs-base	% Named arguments
		try
			params=parse_pv_pairs(params,argin(base:end));
		catch
			err=lasterror;
			msg=err.message;
			end
		end

	params.List = params.List~=0;
	params.Sort = params.Sort~=0;
	params.LexicalSort = params.LexicalSort~=0;
	% Keep supporting NaN, but don't document it.
	if isnan(params.UniformOutput)
		params.UniformOutput = true;
		end

function z=recurse(pth,depth)
%RECURSE List directories recursively.
%   RECURSE PTH lists the files in a directory and its
%   subdirectories. Pathnames and wildcards may be used.  For example,
%   RECURSE *.m lists all the M-files in the current directory and all
%   subdirectories.
%
%   RECURSE(PATH,LVL) only goes LVL levels down. Default is infinite.
%
%   Input can be a cellstr of directory names and/or filenames. Wildcards are
%   only considered for filenames. Returned paths are relative to given
%   path.
%
%   Example
%   -------
%   % To get a list of all m-files in Matlabs 'extern' directory:
% 
%   fn=recurse(fullfile(matlabroot,'extern','*.m'])
%
%   See also DIR.

%   Author: Jerker Wågberg, More Research, 2006-01-18

	if nargin<2;depth=inf;end
	if nargin<1;pth=''; end

	pth=cellstr(pth);
	z={};
	for i=1:length(pth)
		
		% Append a [filesep '*'] if pth{i} is an existing directory
		if exist(pth{i},'file')==7
			pth{i}=fullfile(pth{i},'*');
			end
		% Append a filesep if not in base directory
		[p,f,e]=fileparts(pth{i});
		if ~isempty(p)
			p=[p filesep]; %#ok<AGROW>
			end

		% Resolve wildcards and append files to list
		d=dir(pth{i});
		d([d.isdir])=[];
		if ~isempty(d)
			z=[z;strcat(repmat({p},length(d),1),{d.name}')];
			end
		
		% Resolve recursion. Get directories,
		% remove '.' and '..' and recurse
		d=dir(p);
		d(~[d.isdir])=[];
		% Don't know if I can trust . and .. being the two first entries.
		% Playing it safe...
		d(strmatch('.', {d.name}, 'exact'))=[];
		d(strmatch('..', {d.name}, 'exact'))=[];
		nd=length(d);
		if nd && depth
			pths=strcat(repmat({p},nd,1),{d.name}',repmat({[filesep f e]},nd,1));
			pths=recurse(pths,depth-1);
			z=[z;pths];
			end
		end

% parse_pv_pairs, below, is completely John D'Errico's work, except for my
% "weird" indentation and complying to mlint suggestions

function params=parse_pv_pairs(params,pv_pairs)
% parse_pv_pairs: parses sets of property value pairs, allows defaults
% usage: params=parse_pv_pairs(default_params,pv_pairs)
%
% arguments: (input)
%  default_params - structure, with one field for every potential
%             property/value pair. Each field will contain the default
%             value for that property. If no default is supplied for a
%             given property, then that field must be empty.
%
%  pv_array - cell array of property/value pairs.
%             Case is ignored when comparing properties to the list
%             of field names. Also, any unambiguous shortening of a
%             field/property name is allowed.
%
% arguments: (output)
%  params   - parameter struct that reflects any updated property/value
%             pairs in the pv_array.
%
% Example usage:
% First, set default values for the parameters. Assume we
% have four parameters that we wish to use optionally in
% the function examplefun.
%
%  - 'viscosity', which will have a default value of 1
%  - 'volume', which will default to 1
%  - 'pie' - which will have default value 3.141592653589793
%  - 'description' - a text field, left empty by default
%
% The first argument to examplefun is one which will always be
% supplied.
%
%   function examplefun(dummyarg1,varargin)
%   params.Viscosity = 1;
%   params.Volume = 1;
%   params.Pie = 3.141592653589793
%   params.Description = '';
%   params=parse_pv_pairs(params,varargin);
%   params
%
% Use examplefun, overriding the defaults for 'pie', 'viscosity'
% and 'description'. The 'volume' parameter is left at its default.
%
%   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')
%
% params = 
%     Viscosity: 10
%        Volume: 1
%           Pie: 3
%   Description: 'Hello world'
%
% Note that capitalization was ignored, and the property 'viscosity'
% was truncated as supplied. Also note that the order the pairs were
% supplied was arbitrary.

	npv = length(pv_pairs);
	n = npv/2;

	if n~=floor(n)
		error 'Property/value pairs must come in PAIRS.'
		end
	if n<=0
		% just return the defaults
		return
		end

	if ~isstruct(params)
		error 'No structure for defaults was supplied'
		end

	% there was at least one pv pair. process any supplied
	propnames = fieldnames(params);
	lpropnames = lower(propnames);
	for i=1:n
		pi = lower(pv_pairs{2*i-1});
		vi = pv_pairs{2*i};

		ind = strmatch(pi,lpropnames,'exact');
		if isempty(ind)
			ind = strmatch(pi,lpropnames);
			if isempty(ind)
				error(['No matching property found for: ',pv_pairs{2*i-1}])
			elseif length(ind)>1
				error(['Ambiguous property name: ',pv_pairs{2*i-1}])
				end
			end
		pi = propnames{ind};

		% override the corresponding default in params
		params.(pi)=vi;
		end

function z=isstring(x)
	z=ischar(x) || iscellstr(x);
