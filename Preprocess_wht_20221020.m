clc
clear
load('./Result/20221104/D2Three_Couple_epochs.mat');
fs = 512;
% %% sEEG
% for i =11:15
%     if i~=13
%     Couple_epochs{i} = reshape(cell2mat(Couple_epochs{i}), [numel(Couple_epochs{i})-1,2500]);
%     end
% end

%%

for i = 1:length(Couple_epochs)
    C_c = Couple_epochs{i};
    %M_C = mean(C_c,1);

    for p = 1:size(C_c,1)
        temp = C_c(p,:);
        temp_r = bandpass(temp,[80,100],fs);
        temp_s = bandpass(temp,[12,16],fs);
        temp_o = bandpass(temp,[0.16,1.25],fs);
        [scalar,~,ts] = spectrogram(temp_r,10,1,20,fs);
        subplot(3,1,1)
        spectrogram(temp_r,10,1,20,fs,'yaxis')
        colorbar off
        subplot(3,1,2)
        plot(temp_s)
        hold on
        plot(temp_r)
        plot(temp_o)
        hold off
        subplot(3,1,3)
        plot(temp)
    end
end


%% EEG_couple

for i = 1:length(EEG_couple)
    C_c = EEG_couple{i};
    
    for q = 8:size(C_c,2)
        C_c_c(:,:) = C_c(:,q,:);
        for p = 1:size(C_c_c,1)
            temp = C_c_c(p,:);
            % temp_r = bandpass(temp,[80,100],fs);
            temp_s = bandpass(temp,[12,16],fs);
            temp_o = bandpass(temp,[0.16,1.25],fs);
            [scalar,~,ts] = spectrogram(temp_s,10,1,20,fs);
            subplot(3,1,1)
            spectrogram(temp_s,10,1,20,fs,'yaxis')
            colorbar off
            subplot(3,1,2)
            plot(temp_s)
            hold on
            % plot(temp_r)
            plot(temp_o.*0.2)
            hold off
            subplot(3,1,3)
            plot(temp)
        end
    end
end

%% statistical
Couple_nums = zeros(10,12);
for p = 11:22
    
    EEG_ind = Couple_ind{p};
    for i = 1:size(EEG_ind,1)
        s_1 = EEG_ind(i,1);
        for k = 1:10
            s_all_2 = Couple_ind{k}(:,2) - s_1;
            if abs(min(s_all_2)) <= 750
                Couple_nums(k, p-10) = Couple_nums(k, p-10)+1;
            end
        end
    end
end
