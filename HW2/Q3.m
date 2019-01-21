function main
    %% Reset workspace
    clc
    clear
    close all
    
    %% Set up problem parameters
    
    prob.rentProfit = 10;  % Amount of money Jack makes per car rental
    prob.costPerCarMoved = 2;  % Cost to move car from location 1 to 2
    prob.meanRequests = [3; 4];  % Average number of requests in the morning
    prob.meanReturns = [3; 2];  % Average number of returns at the end of the day
    prob.maxCars = 20;  % Maximum number of cars a location can support
    prob.maxCarsMoved = 5;  % Maximum number of cars that can be moved each night
    prob.discountRate = 0.9;

    % Used to store the policy of how many cars should be moved to/from
    % each location at the end of the night
    prob.values = zeros(prob.maxCars+1, prob.maxCars+1);
    prob.policies = zeros(prob.maxCars+1, prob.maxCars+1);
    
%% Implement reinforcement learning
    for itrCount = 1:1:1

        % Policy evaluation
        while true
            delta = 0;  % Used to check whether the values have converged 

            % Iterate over all states
            for i = 0:1:prob.maxCars
                for j = 0:1:prob.maxCars

                    % Note: for values and policies, must add +1 to index
                    v = prob.values(i+1, j+1);  % Get current value for the current state

                    % Get the state value using the Bellman equation
                    V = Bellman(prob, [i;j], prob.policies(i+1,j+1)); 
                    
                    prob.values(i+1,j+1)=V;  % Save to updated state values
                    delta = max([delta, abs(v-V)]);
                end
            end

            % Check whether the values have converged
            if delta < 1e-3
               delta
               prob.values
               break 
            end 
        end
        
        % Policy improvement
        policy_stable = 1;
        
        % For each s in S
        for i=0:1:prob.maxCars
            for j=0:1:prob.maxCars
                old_action = prob.policies(i+1,j+1);
                
                % Get new action
                actionVals = [];
                
                % Valid actions based on present cars
                possibleActions = [-min(i,prob.maxCarsMoved):1:min(j,prob.maxCarsMoved)];  
                
                % Check all actions
                for a = possibleActions
                    pi_s = Bellman(prob, [i;j], a);
                    actionVals = [actionVals; pi_s];
                end
                
                % Find maximum argument
                [argMax, argVal] = max(actionVals);
                
                new_action = possibleActions(argVal);
                
                prob.policies(i+1,j+1) = new_action;
                
                if new_action ~= old_action
                   policy_stable = 0; 
                end
                
            end
        end
        
        if policy_stable
           prob.values
           prob.policies
           return
        end

    end
end

function[value] = Bellman(problem, state, action)
    
    % Next state is the current state plus the expected number of returns,
    % then changed by the number of cars transferred
    
    nextState = state;
    nextState = nextState + problem.meanReturns;
    nextState(1) = nextState(1) + action;  % Move cars
    nextState(2) = nextState(2) - action; 
    
    if any(nextState <= 0)
        nextState
    end
    
    % Saturate to maxCars
    nextState(nextState > problem.maxCars) = problem.maxCars;

    value = reward(problem, nextState, action) + ...
            problem.discountRate * problem.values(nextState(1)+1,nextState(2)+1);
end

function[val] = reward(problem, state, action)

    reward1 = min(problem.meanRequests(1), state(1));
    reward2 = min(problem.meanRequests(2), state(2));
    
    reward = (reward1 + reward2) * problem.rentProfit;
    
    loss = abs(action) * problem.costPerCarMoved;
    
    val = reward - loss;
   
end