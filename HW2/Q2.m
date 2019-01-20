function main()
%% Gridworld Initialization
    world.SIZE = 5;
    
    world.A_POS = [1; 2];
    world.A_PRIME_POS = [5; 2];

    world.B_POS = [1; 4];
    world.B_PRIME_POS = [3; 4];
    
    world.A_REWARD = 10;
    world.B_REWARD = 5;
    
    world.OUT_OF_BOUNDS_REWARD = -1;
    
    world.values = zeros(world.SIZE, world.SIZE);
    
    world.DISCOUNT = 0.9;
    
%% Compute state values
while true
    
    old_value = world.values(world.A_POS(1), world.A_POS(2));
    
    % Iterate over the elements of the matrix to update the state values
    for i = 1:1:world.SIZE
       for j = 1:1:world.SIZE
           world.values(i,j) = newValue(i, j, world);
       end
    end
    
    new_value = world.values(world.A_POS(1), world.A_POS(2));
    
    % Calculate to 3 decimal places
    if new_value - old_value < 1e-4
       break 
    end
end

% Print result
world.values
end

function[reward] = getReward(i, j, i_next, j_next, world)
% getReward: Function to return the reward from transitioning to state
% (i,j). Rewards outside the bounds of the world return a penalty, and
% transitions out of states A or B have special rewards
%
% Inputs:
%   i: Matrix state row coordinate to compute the reward at
%   j: Matrix state column coordinate to compute the reward at
% Outputs:
%   reward: The reward from transitioning to state s' from state s

    % Default
    reward = 0;
    
    % Check if in A
    if ([i; j] == world.A_POS)
       reward = world.A_REWARD;
       return
    end
    
    % Check if in B
    if ([i; j] == world.B_POS)
        reward = world.B_REWARD;
        return
    end

    % Check if next step is outside the grid
    if (i_next < 1)             || ...
        (j_next < 1)            || ...
        (i_next > world.SIZE)   || ...
        (j_next > world.SIZE)
    
       reward = world.OUT_OF_BOUNDS_REWARD;
       return
    end
    
end

function[newValue] = newValue(i, j, world)
% newValue: Updates the new state value of a given state.
%
% Inputs:
%   i: i index of the current state
%   j: j index of the current state
%   world: struct to describe the gridworld
% Outputs:
%   newValue: new computed state value
    
    % First, determine the rewards that would be determined by going in
    % each direction from the current state
    rewardUp = getReward(i, j, i-1, j, world);
    rewardDown = getReward(i, j, i+1, j, world);
    rewardLeft = getReward(i, j, i, j-1, world);
    rewardRight = getReward(i, j, i, j+1, world);
    
    % Second, determine the state values that would result from moving in
    % each possible direction
    valueUp = rewardUp + ...
        world.DISCOUNT*getValueSPrime(i, j, i-1,j,world);
    valueDown = rewardDown + ...
        world.DISCOUNT*getValueSPrime(i, j, i+1,j,world);
    valueLeft = rewardLeft + ...
        world.DISCOUNT*getValueSPrime(i, j, i,j-1,world);
    valueRight = rewardRight + ...
        world.DISCOUNT*getValueSPrime(i, j, i,j+1,world);
    
    % Find the maximal state value
    newValue = max([valueUp, valueDown, valueLeft, valueRight]);
 
end

function[value] = getValueSPrime(i, j, i_next, j_next, world)
% getValueSPrime: Gets the state value of the next state given the current
% state. NOTE: If the current state is A or B, the next state will be A' or
% B' regardless of the action decision
%
% Inputs:
%   i: Current state i index
%   j: Current state j index
%   i_next: Next state i index
%   j_next: Next state j index
% Outputs:
%   value: The state value of the next state

    % If the current state is A, the value of the next state is A'
    % regardless of the action taken
    if ([i;j] == world.A_POS)
       value = world.values(world.A_PRIME_POS(1), world.A_PRIME_POS(2));
       return
    end
    
    % Same logic for B
    if ([i;j] == world.B_POS)
       value = world.values(world.B_PRIME_POS(1), world.B_PRIME_POS(2));
       return
    end

    % Check if the action would take the agent off the map. If so, the
    % value of the next state is 0
    if (i_next < 1)             | ...
        (j_next < 1)            | ...
        (i_next > world.SIZE)   | ...
        (j_next > world.SIZE)
        value = 0;
    else
        value = world.values(i_next,j_next);
    end
end

