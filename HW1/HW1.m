function main
%% Perform initialization

    k = 10;
    N = 2000;  % Number of iterations
    steps = 1000; 

    % Repeat N times to get an average
    figure
    for eps = [0, 0.01, 0.1];
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
    N = zeros(k, 1);

    Rt = [];

    for t = 1:1:steps

        [argval, argmax] = max(Q);

        % Choose whether to do the greedy action or a random action
        choice = binornd(1, 1-eps);

        if choice
            A = argmax;
        else
            A = randi([1, k]);
        end

        R = normrnd(q(A), 1);

        Rt = [Rt; R];

        N(A) = N(A) + 1; % Increment number of evaluation of action A

        Q(A) = Q(A) + 1/N(A)*(R-Q(A));  % Update estimate of action A

    end
end