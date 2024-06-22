function [t, x] = GetLinearPath(duration)
    % Initialize data storage
    t = [];
    x = [];
    
    % Create figure and slider
    f = figure('Name', 'Linear Motion Tracker', 'Position', [100, 100, 600, 400]);
    slider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1.5, 'Value', 0, ...
                       'Position', [150, 50, 300, 20]);
    ax = axes('Parent', f, 'Position', [0.1, 0.3, 0.85, 0.6]);
    hold(ax, 'on');
    ylabel(ax, 'Position');
    xlabel(ax, 'Time (s)');
    title(ax, 'Position vs. Time');
    grid(ax, 'on');

    % Setup timer
    sampleRate = 0.01; % Sample rate in seconds
    numSamples = floor(duration / sampleRate);
    timerObj = timer('ExecutionMode', 'fixedSpacing', 'Period', sampleRate, ...
                     'TasksToExecute', numSamples, 'TimerFcn', @timerCallback, ...
                     'StopFcn', @stopTimer);

    % Timer callback function
    function timerCallback(~, ~)
        % Get the current time and slider value
        currentTime = toc;
        currentPos = get(slider, 'Value');
        t = [t, currentTime];
        x = [x, currentPos];
        plot(ax, t, x, 'b-o');
    end

    % Stop function
    function stopTimer(~, ~)
        delete(timerObj);
        close(f);
    end

    % Start timing and the timer
    tic;
    start(timerObj);
    uiwait(f);
end
