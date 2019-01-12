function main
%% Perform initialization

    k = 10;  % Number of bandits
    N = 2000;  % Number of iterations to run the bandit problem for averaging
    steps = 1000; % Number of time steps to run the bandit problem for

    % Repeat N times to get an average
    figure
    for eps = [0, 0.01, 0.1];  % Parameter to choose how often to explore
        RtAll = zeros(steps, N); % Holds all reward values over all trials
        
        for i = 1:1:N  % Runs N trials (2000)
            Rt(:,i) = banditProblem(eps, k, steps);
        end

        plot(mean(Rt, 2));  % Plot the average of all 2000 trials
        hold on

    end
    
    % Plot formatting
    xlabel("Steps")
    ylabel("Average Reward")
    title("Average Reward vs Number of Steps")
    legend("eps = 0 (greedy)", "eps = 0.01", "eps = 0.1")
    
end

function[Rt] = banditProblem(eps, k, steps)

    q = normrnd(0,1, [k, 1]); % Actual expected reward values
    Q = zeros(k, 1); % Initial guess of the expected reward of each 
    N = zeros(k, 1); % Number of times each arm has been pulled

    Rt = [];  % Stores the rewards at each time step  

    for t = 1:1:steps

        [argval, argmax] = max(Q);

        % Choose whether to do the greedy action or a random action
        choice = binornd(1, 1-eps);

        if choice  % If the optimal action should be taken
            A = argmax;
        else  % Otherwise, the random action should be taken
            A = randi([1, k]);
        end

        R = normrnd(q(A), 1);  % Adds gaussian random noise to the true value

        Rt = [Rt; R];  % Save the reward for plotting purposes

        N(A) = N(A) + 1; % Increment number of evaluations of action A

        Q(A) = Q(A) + 1/N(A)*(R-Q(A));  % Update estimate of action A

    end
end