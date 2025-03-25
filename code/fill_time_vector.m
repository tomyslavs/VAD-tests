function Dtime = fill_time_vector(num_of_win,win_len_ms,Df)
    Dstart = 0; Dend = 0;
    Dff = cat(2,0,Df); % extend Df with zero for filtering
    for i=1:num_of_win
        if Dff(i)==0 && Dff(i+1)==1
            Dstart=cat(1,Dstart,i*win_len_ms/1000); % speech start time
        end
        if Dff(i)==1 && Dff(i+1)==0
            Dend=cat(1,Dend,i*win_len_ms/1000); % speech end time
        end
    end
    if length(Dend)<length(Dstart) % if record ends with speech
        Dend=cat(1,Dend,num_of_win*win_len_ms/1000);
    end
    if length(Dend)>1
        Dend=Dend(2:end);
        Dstart=Dstart(2:end);
    end
    Dtime = cat(2,Dstart,Dend);
end 