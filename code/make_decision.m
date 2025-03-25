function Dtime = make_decision(D,win_len,silence_offset)
 
% D                             % klasifikatoriaus isejimo vektorius
win_len_ms = win_len*1000;      % apdorojamo slenkancio lango dydis ms
% silence_offset = 0.05;        % sekundemis; Koki intervala paimti pries/po snekos detektavimo
win_per_sec = 1000/win_len_ms;	% langu sekundeje
num_of_win = length(D);         % langu skaicius

b = ones(1,fix(win_per_sec*silence_offset))/fix(win_per_sec*silence_offset);
Df = filter(b,1,D); Df(Df>0) = 1;
Df_lr = filter(b,1,fliplr(D)); Df_lr = fliplr(Df_lr); Df_lr(Df_lr>0) = 1; % apsuktas vektorius, kad iterptu intervala pries detektuota garsa
Df = double(Df | Df_lr);

%% Fill start/end vectors
Dtime = fill_time_vector(num_of_win,win_len_ms,Df); % Cia rezultato vektorius: [pradzia pabaiga]

end