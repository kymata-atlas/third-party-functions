function ImageSeq = load_our_stim(first_last_frame)

    total_trials = 400;
    total_frames = 60;

    if nargin == 0
        trimmed = true;
        first_frame = 1;
        last_frame = total_trials * total_frames;
    else
        trimmed = false;
        first_frame = first_last_frame(1);
        last_frame = first_last_frame(2);
    end%if
    
    n_frames = 1 + last_frame - first_frame;
    
    [w, h, ~] = size(imread(filename_from_trial_and_frame(1, 1)));
    
    % Preallocate
    ImageSeq = zeros(w, h, n_frames);
    
    frame_count = 1;
    for overall_frame = first_frame:last_frame
        [trial, frame] = get_trial_and_frame_from_overall_frame(overall_frame);
        
        fprintf('Loading image for trial %03d/%03d, frame %02d/%02d\n', trial, total_trials, frame, total_frames);
        
        image_path = filename_from_trial_and_frame(trial, frame);
        frame_image = imread(image_path);
        lab_image = rgb2lab(frame_image);
        luminocity = lab_image(:, :, 1);
        ImageSeq(:, :, frame_count) = luminocity;
            
        frame_count = frame_count + 1;
    end%for frame
end%function

function [trial_i, frame_i] = get_trial_and_frame_from_overall_frame(overall_frame)
    total_frames = 60;
    trial_i = ceil(overall_frame / total_frames);
    frame_i = mod(overall_frame, total_frames);
end

function filename = filename_from_trial_and_frame(trial, frame)
    base_dir = '/Users/cai/Box Sync/Kymata/KYMATA-visual-stimuli-dataset-3_01/video_stimuli_trials';
    filename = fullfile(base_dir, sprintf('trial_%03d', trial), sprintf('frame%02d.png', frame));
end