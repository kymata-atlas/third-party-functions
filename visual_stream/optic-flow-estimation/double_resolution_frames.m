
framedirs = dir('60Hz/trial_*');
directoryNames = {framedirs([framedirs.isdir]).name};

for i = 1:length(directoryNames)
    thisframedir = directoryNames{i};
    files = dir(['60Hz/' thisframedir '/*.png']);
    status = mkdir(['120Hz/' thisframedir]);
    count = 1;
    % Loop through each
    for id = 1:length(files)
        % Convert to number
        for b=1:2
            name = ['120Hz/' thisframedir '/frame' num2str(count, '%0.2d')];
            copyfile(['60Hz/' thisframedir '/' files(id).name], sprintf('%s.png', name));
            count = count +1;
        end
    end
end
