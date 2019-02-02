function main
%% Initialization
clc
clear
close all

%% Parameters

% Actions
game.stick = 1;
game.hit = 2;

% Rewards
game.winReward = 1;
game.drawReward = 0;
game.loseReward = -1;

% Player [12-21] * Dealer [1-12] * hasUsableAce [0-1] = 200 states. NOTE:
% Although the player's state for simulation can only be [12-12], the array
% must be allocated with a full 21 states so that they can be indexed from
% [12-21]
numPlayerScoreStates = 21;
numUsableAceStates = 2;
numDealerShowingStates = 10;

numActions = 2;  % Hit, stick

numEpisodes = 10000;
numRuns = 100;

actualStateVal = -0.27726;

%% Ordinary Importance Sampling

mseOrdinaryAll = zeros(numEpisodes,numRuns);
mseWeightedAll = zeros(numEpisodes, numRuns);
allRho = zeros(numEpisodes, 1);
allRewards = zeros(numEpisodes, 1);
for run = 1:1:numRuns
    for epNum = 1:1:numEpisodes
        
        rho = 1;
        
        playerScore = 13;
        usableAce = 1;
        
        dealerCards = [2; getCards(1)];
        dealerShowing = dealerCards(1);
        
        playerAction = piBehavior(playerScore, game);
        
        [reward, states, actions] = episode(playerScore, dealerCards,usableAce, playerAction, game);
        allRewards(epNum) = reward;
        
        % b(hit|Sk) = b(stick|Sk) = 0.5, since either action has an equal probability for any
        % state
        %
        % pi(hit | Sk) = 1 for S < 20, 0 otherwise
        for i = 1:1:numel(actions)
            numerator = 1;  % Placeholders
            denominator = 1;
            
            if actions(i) == piTarget(states(i), game)
                numerator = 1;
            else
                numerator = 0;
            end
            
            denominator = 0.5;
            
            rho = rho * (numerator/denominator);
        end
        
        allRho(epNum) = rho;
        
    end
    
    ordinarySampling = allRho.*allRewards;
    ordinarySampling = cumsum(ordinarySampling)./(1:1:numEpisodes)';
    
    weightedSampling = allRho.*allRewards;
    weightedSampling = cumsum(weightedSampling)./cumsum(allRho);
    k = find(isnan(weightedSampling));
    weightedSampling(k) = 0;

    % Get MSE
    mseOrdinary = (ordinarySampling - actualStateVal).^2;
    mseOrdinaryAll(:,run) = mseOrdinary;
    
    mseWeighted = (weightedSampling - actualStateVal).^2;
    mseWeightedAll(:,run) = mseWeighted;
end

figure
semilogx(mean(mseOrdinaryAll,2))
hold on
semilogx(mean(mseWeightedAll,2))
xlabel('Episodes (log scale)')
ylabel('Mean Square Error (average over 100 runs)')
ylim([0,5])
legend('Ordinary Sampling', 'Weighted Sampling')
title('Ordinary Sampling vs Weighted Sampling MSE')

save('results2.mat')

end

function[reward, states, actions] = episode(playerScore, dealerCards, haveAce, firstAction, game)

states = playerScore;
actions = firstAction;

dealerShowing = dealerCards(1);

% Natural Case: Player has an initial score of 21
if playerScore == 21
    % Check whether the dealer also has a natural
    dealerScores = sum(dealerCards);  % Default case where Ace = 1
    if any(dealerCards == 1)
        dealerScores = [dealerScores; dealerScores+10];  % Count Ace = 11
    end
    
    if any(dealerScores == 21)
        reward = game.drawReward;
        return;
    else
        reward = game.winReward;
        return;
    end
end

% If score is less than 20, always draw a new card. NOTE: the initial
% player score is assumed to count the ace as an 11. This can be changed
% later depending on future card draws

action = firstAction; % Initialize
while playerScore < 20
    
    if action == game.hit
        card = getCards(1);
        playerScore = playerScore + card;
    else
        break;
    end
    
    % If the player is over 21 and the player has a usable ace, count the
    % ace as a 1 (ie subtract 10)
    if (playerScore > 21 & haveAce == 1)
        playerScore = playerScore - 10;
        haveAce = 0;
    end
    
    % Append to state
    states = [states; playerScore];
    
    % Determine next action based on current new state
    action = piBehavior(playerScore, game);
    actions = [actions; action];
end

% Check if the player's score is > 21. If yes, no need for the dealer to
% play
if playerScore > 21
    reward = game.loseReward;
    return;
end

% Dealer's turn
dealerScore = sum(dealerCards);  % Default case where Ace = 1
dealerAces = sum(dealerCards == 1);

if dealerAces
    dealerScore = dealerScore + dealerAces*10;  % Apply Aces by default.
    
    if dealerScore > 21
        dealerScore = dealerScore - 10;
        dealerAces = dealerAces - 1;
    end
end

while dealerScore < 17
    card = getCards(1);
    
    if card == 1
        dealerScore = dealerScore + 11;
        dealerAces = dealerAces + 1;
    else
        dealerScore = dealerScore + card;
    end
    
    if (dealerScore > 21 & dealerAces > 0)
        dealerScore = dealerScore - 10;
        dealerAces = dealerAces - 1;
    end
end

% Calculate reward
if dealerScore > 21
    reward = game.winReward;
else
    if playerScore > dealerScore
        reward = game.winReward;
    elseif playerScore == dealerScore
        reward = game.drawReward;
    else
        reward = game.loseReward;
    end
end

return;

end

function[value] = getCards(score)
value = min(10, randi([1,13], score, 1));
end

function[action] = piTarget(numCards, game)

if numCards < 20
    action = game.hit;
else
    action = game.stick;
end
end

function[action] = piBehavior(score, game)
action = randi([game.stick, game.hit]);
end