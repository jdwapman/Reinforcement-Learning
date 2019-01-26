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

maxIter = 1e7;
%% Monte Carlo Exploring Starts

% 1) Initialize:
pi_s = zeros(numPlayerScoreStates, numDealerShowingStates, numUsableAceStates)+2;  % Initialize policy to always pass
Q = zeros(numPlayerScoreStates, numDealerShowingStates, numUsableAceStates, numActions);  % State-action values
V = zeros(numPlayerScoreStates, numDealerShowingStates, numUsableAceStates);  % State values (max of Q for a state)
numVisits = zeros(numPlayerScoreStates, numDealerShowingStates, numUsableAceStates, numActions);  % Number of times each state-action pair has been visited

for itrCount = 1:1:maxIter

    % Choose statem action randomly such that all pairs have P > 0
    playerScore = randi([12,21]);  % 10 states
    usableAce = randi([0,1]);  % 2 states
    
    dealerCards = getCards(2);
    dealerShowing = dealerCards(1);
    
    if playerScore < 20
        playerAction = randi([game.stick, game.hit]);
    else
        playerAction = game.stick;
    end
  
    reward = episode(playerScore, dealerCards,usableAce, playerAction, pi_s, Q, game);

    % For blackjack, can't loop states, so don't need to check if visited
    % more than once since the player's score can only increase

    % Incorporate newest reward into average
    % Qn+1 = Qn + 1/n*(Rn-Qn)
    n = numVisits(playerScore, dealerShowing, usableAce+1, playerAction) + 1;
    
    Q(playerScore, dealerShowing, usableAce+1, playerAction) = ...
        Q(playerScore, dealerShowing,usableAce+1, playerAction) + ...
        (1/n)*(reward - Q(playerScore, dealerShowing,usableAce+1, playerAction));

    % Increment n
    numVisits(playerScore, dealerShowing,usableAce+1, playerAction) = n;
    
    % Update policy
    [argVal, argMax] = max(Q(playerScore, dealerShowing,usableAce+1, :));
    V(playerScore, dealerShowing,usableAce+1) = argVal;
    pi_s(playerScore, dealerShowing,usableAce+1) = argMax;
end

dealerAxis = 1:1:10;
playerAxis = 12:1:21;

[xx,yy] = ndgrid(playerAxis, dealerAxis);

figure
surf(xx,yy,V(12:21,1:10,1))
xlabel('Player Sum')
ylabel('Dealer Showing')
title('V_* (No Usable Ace)')
xlim([12,21])
ylim([1,10])
zlim([-1,1])

figure
surf(xx,yy,V(12:21,1:10,2))
xlabel('Player Sum')
ylabel('Dealer Showing')
title('V_* (Usable Ace)')
xlim([12,21])
ylim([1,10])
zlim([-1,1])

figure
image(1:10,11:1:21,pi_s(11:21,1:10,1), 'CDataMapping', 'scaled')
title('\pi_* (No Usable Ace)')
xlabel('Dealer Showing')
ylabel('Player Sum')
ax=gca;
ax.YDir='normal'

figure
image(1:10,11:1:21,pi_s(11:21,1:10,2), 'CDataMapping', 'scaled')
title('\pi_* (Usable Ace)')
xlabel('Dealer Showing')
ylabel('Player Sum')
ax=gca;
ax.YDir='normal'

save('results.mat')

end

function[reward] = episode(playerScore, dealerCards, haveAce, firstAction, pi_s, Q, game)

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
    
    % Determine next action based on current new state
    if playerScore < 20
        action = pi_s(playerScore, dealerShowing, haveAce+1);
    end
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

function[value] = getCards(numCards)
    value = min(10, randi([1,13], numCards, 1));
end