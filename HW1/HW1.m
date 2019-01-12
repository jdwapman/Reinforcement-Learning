function main
%% Perform initialization
    clc
    clear
    close all
    
    k = 10;  % Number of bandits
    N = 2000;  % Number of iterations to run the bandit problem for averaging
    steps = 1000; % Number of time steps to run the bandit problem for
    
%% Epsilon-greedy method
    rewardFig = figure;
    optFig = figure;
    
    for eps = [0, 0.01, 0.1]
        
       c = 0;
       [reward, optimal] = iterateBandit(N, steps, eps, k, c);
       
       figure(rewardFig)
       plot(reward)
       hold on
       
       figure(optFig)
       plot(optimal)
       hold on
    end
    
    % Plot formatting
    figure(rewardFig)
    xlabel("Steps")
    ylabel("Average Reward")
    title("Average Reward vs Number of Steps")
    legend("eps = 0 (greedy)", "eps = 0.01", "eps = 0.1")
    
    figure(optFig)
    xlabel("Steps")
    ylabel("% Optimal Action")
    title("% Optimal Action vs Number of Steps")
    legend("eps = 0 (greedy)", "eps = 0.01", "eps = 0.1")
    ylim([0, 1]);
   
    % Convert y-axis values to percentage values by multiplication
    a=[cellstr(num2str(get(gca,'ytick')'*100))]; 
    % Create a vector of '%' signs
    pct = char(ones(size(a,1),1)*'%'); 
    % Append the '%' signs after the percentage values
    new_yticks = [char(a),pct];
    % 'Reflect the changes on the plot
    set(gca,'yticklabel',new_yticks)
    
%% UCB Method

    rewardFig = figure;
    
    for c = [0, 2]
        
       eps = 0.1;
       [reward, optimal] = iterateBandit(N, steps, eps, k, c);
       
       figure(rewardFig)
       plot(reward)
       hold on
       
    end
    
    % Plot formatting
    figure(rewardFig)
    xlabel("Steps")
    ylabel("Average Reward")
    title("Average Reward vs Number of Steps")
    legend("Epsilon-greedy eps = 0.1", "UCB c = 2")
    
end

function[reward, optimal] = iterateBandit(N, steps, eps, k, c)

    RtAll = zeros(steps, N); % Holds all reward values over all trials
    optimalAll = zeros(steps, N);

    for i = 1:1:N  % Runs N trials (2000)
        [reward, optimal] = banditProblem(eps, k, c, steps);

        Rt(:,i) = reward;

        t = 1:1:steps;
        t=t';
        percentOptimal = cumsum(optimal)./t;

        optimalAll(:,i) = percentOptimal;
    end

    reward = mean(Rt, 2);
    optimal = mean(optimalAll, 2);

end

function[Rt, optimalAction] = banditProblem(eps, k, c, steps)

    q = normrnd(0,1, [k, 1]); % Actual expected reward values
    Q = zeros(k, 1); % Initial guess of the expected reward of each 
    N = zeros(k, 1); % Number of times each arm has been pulled
    
    % Determine the optimal reward
    [optMax, optArg] = max(q);

    Rt = [];  % Stores the rewards at each time step  
    optimalAction = [];

    for t = 1:1:steps

        %[argval, argmax] = max(Q);
        
        argmax = actionSelect(Q, c, t, N);

        % Choose whether to do the greedy action or a random action
        if c == 0
            choice = binornd(1, 1-eps);
        else
            choice = 1;
        end

        if choice  % If the optimal action should be taken
            A = argmax;
        else  % Otherwise, the random action should be taken
            A = randi([1, k]);
        end
        
        if A == optArg
            optimalAction = [optimalAction; 1];
        else
            optimalAction = [optimalAction; 0];
        end

        R = normrnd(q(A), 1);  % Adds gaussian random noise to the true value

        Rt = [Rt; R];  % Save the reward for plotting purposes

        N(A) = N(A) + 1; % Increment number of evaluations of action A

        Q(A) = Q(A) + 1/N(A)*(R-Q(A));  % Update estimate of action A

    end
end

function[arg] = actionSelect(Q, c, t, N)
    max = -Inf;
    arg = 0;
    
    for a = 1:1:numel(Q)
        if N(a) == 0; % Avoids division by zero
           x = Inf; 
        else
            x = Q(a)+c*sqrt(log(t)/N(a));
        end
        
        if x > max
           max = x;  % Update new max
           arg = a;  % Update new arg
        end
    end
end