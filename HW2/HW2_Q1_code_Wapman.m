function main
%% Gamma = 0
gamma = 0;
valuesLeft = [1; 0];
valuesRight = [0; 2];

rewardLeft0 = reward(gamma, valuesLeft)
rewardRight0 = reward(gamma, valuesRight)

%% Gamma = 0.5
gamma = 0.5;

rewardLeft0_5 = reward(gamma, valuesLeft)
rewardRight0_5 = reward(gamma, valuesRight)

%% Gamma = 0.9
gamma = 0.9
rewardLeft0_9 = reward(gamma, valuesLeft)
rewardRight0_9 = reward(gamma, valuesRight)

end

function[result] = reward(gamma, values)

    idx = 0;

    result = 0;
    
    for count = 0:1:1000
        result = result + (gamma^count)*values(idx+1);
        idx = ~idx;
    end

end
