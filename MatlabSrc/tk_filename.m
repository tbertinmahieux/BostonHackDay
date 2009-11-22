function F = TK_filename(TK)

if TK(2) ~= '/'
  TK = [TK(3),'/',TK(4),'/',TK];
end


F = ['mat/',TK,'.mat'];
