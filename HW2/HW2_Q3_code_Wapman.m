function main
    %% Reset workspace
    clc
    clear
    close all
    
    %% Set up problem parameters
    
    % Note: p is a structure to store the problem parameters, to make it
    % easier to pass many values to the various functions
    
    % Maximum number of cars that can be at a location
    p.maxCars = 20;
    
    % Income per car rented
    p.rentIncome = 10;
    
    % Maximum number of cars that can be moved overnight
    p.maxCarsMoved = 5;
    
    % Cost to move car between locations
    p.costPerCarMoved = 2;
    
    % Lamba paramter for a poisson distribution
    p.meanRentals1 = 3;
    p.meanRentals2 = 4;
    
    % Lambda parameter for a poisson distribution
    p.meanReturns1 = 3;
    p.meanReturns2 = 2;
    
    p.discountRate = 0.9;
    
    % Poisson probability of requesting a combination of cars
    p.rentalProbs = zeros(p.maxCars+1, p.maxCars+1);
    
    % Poisson probability of returning a combination of cars
    p.returnProbs = zeros(p.maxCars+1, p.maxCars+1);
    
    values = zeros(p.maxCars+1, p.maxCars+1);  % State values
    policy = zeros(p.maxCars+1, p.maxCars+1);  % Policy values
    
    %% Precompute Poisson Probabilities
    for i = 0:1:p.maxCars
        for j = 0:1:p.maxCars
            rentalProb = poisspdf(i, p.meanRentals1) * poisspdf(j, p.meanRentals2);
            p.rentalProbs(i+1,j+1) = rentalProb;
            
            returnProb = poisspdf(i, p.meanReturns1) * poisspdf(j, p.meanReturns2);
            p.returnProbs(i+1,j+1) = returnProb;
        end
    end
    
    %% Implement reinforcement learning   
    for itrCounter = 0:1:5
        
        %  Plot results 
        % Policy plot
        figure
        x = 0:1:p.maxCars;
        y = 0:1:p.maxCars;
        levels = -p.maxCarsMoved:1:p.maxCarsMoved;
        %[C,h] = contourf(x, y, policy, levels);
        image(x,y,flip(policy), 'CDataMapping', 'scaled');
        colorbar()
        xlabel("# Cars at Second Location")
        ylabel("# Cars at First Location")
        title(strcat('\pi_', string(itrCounter)))
        %clabel(C,h);
        
        % Policy evaluation
        while 1
            delta = 0;  % Used to check whether the states have converged

           % Iterate over all states
           for i = 0:1:p.maxCars
               for j = 0:1:p.maxCars
                   
                   % Note: for values and policies, must add +1 to index
                   v = values(i+1,j+1);  % Get the current value for the current state

                   % Set the state value using the Bellman equatio
                   V = Bellman(p, values, [i,j], policy(i+1,j+1));

                   values(i+1, j+1) = V;  % Save the updated state values
                   delta = max(delta, abs(v-V));
               end
           end

           if delta < 1e-2
              break 
           end

        end

        % Policy improvement
        policy_stable = 1;  % Initialization
        
        % Iterate over all states
        for i = 0:1:p.maxCars
            for j = 0:1:p.maxCars
                action_returns = [];

                old_policy = policy(i+1,j+1);
                
                % Determines which actions are valid based on the number of
                % cars at each location.
                validActions = -min(j,p.maxCarsMoved):...
                                1:min(i, p.maxCarsMoved);
                            
                % Loop over all actions            
                for a = validActions
                    retval = Bellman(p, values, [i,j], a);
                    action_returns = [action_returns; retval];
                end
                
                % Determine the best action (which gives the maximum state
                % value
                [argVal, argMax] = max(action_returns);
                new_policy = validActions(argMax);
                policy(i+1,j+1) = new_policy;

                % Check whether the policy is stable
                if new_policy ~= old_policy
                   policy_stable = 0; 
                end
            end
        end
        
        if itrCounter == 4
            % Plot values at the fourth (final) iteration
            figure
            [xx, yy] = meshgrid(0:1:p.maxCars, 0:1:p.maxCars);
            mesh(xx,yy,values);
            title("v_\pi_4");
            xlabel("# Cars at First Location");
            ylabel("# Cars at Second Location"); 
        end
    end   
    
    % Save results
    save('resultsQ3.mat')   
end

function[value] = Bellman(p, values, state, action)
    value = 0;  % Initialize to 0
    
    % Cost for car transfer
    loss = abs(action) * p.costPerCarMoved;  
    value = value - loss;
    
    % Go through all possible rental requests
    for numRentals1 = 0:1:15 . % Only go to 15 to improve speed
        for numRentals2 = 0:1:15
            
            % Probability for the given combination of rental requests at
            % the two locations
            rentalProb = p.rentalProbs(numRentals1+1, numRentals2+1);
       
            % Moving cars
            numCars1 = min(state(1) - action, p.maxCars);
            numCars2 = min(state(2) + action, p.maxCars);
            
            % Saturate the number of requests to the number of cars
            % actually available
            rentals1 = min(numCars1, numRentals1);
            rentals2 = min(numCars2, numRentals2);
            
            % Calculate the income from renting the cars
            income = (rentals1 + rentals2) * p.rentIncome;
            numCarsRented1 = numCars1 - rentals1;
            numCarsRented2 = numCars2 - rentals2;          

            % Update the state value calculation.
%             returns1 = p.meanReturns1;
%             returns2 = p.meanReturns2;
%             numCars1 = min(numCars1+returns1, p.maxCars);
%             numCars2 = min(numCars2+returns2, p.maxCars);
%             value = value + rentalProb*(income+p.discountRate*values(numCars1+1, numCars2+1));

            % Higher numbers of returns have lower probabilities.
            for returns1 = 0:1:10 . % Only go to 10 to improve speed
                for returns2 = 0:1:10
                    
                    % Probability for the given combination of rental
                    % returns at the two locations
                    returnProb = p.returnProbs(returns1+1, returns2+1);
                    
                    % Update the number of cars at the locations to get the
                    % next state
                    numCarsReturned1 = min(numCarsRented1 + returns1, p.maxCars);
                    numCarsReturned2 = min(numCarsRented2 + returns2, p.maxCars);
                    
                    % Total probability of getting this combination of
                    % requests and returns at the two locations
                    totalProb = rentalProb*returnProb;
                    
                    % Update the state value
                    value = value + totalProb*(income+...
                        p.discountRate*values(numCarsReturned1+1, numCarsReturned2+1));
                end
            end  
        end
    end  
end

% References:
% https://medium.com/@jaems33/this-is-start-of-my-exploration-into-learning-about-reinforcement-learning-d505a68a2d6
