function ImageSeq = stream_our_stim(starting_at, n_frames, image_mode)

    % Gets a block of stimulus W x H x `n_frames`, with the first frame
    % being `starting_at`.
    % MUST be called in sequence with `starting_at` incremeting by 1 each 
    % time and `n_frames` always the same (or it will throw an error).
    
    %% Persistent state
    
    persistent f; % Number of frames in the data block
    persistent current_first_frame; % The index of the first frame
    persistent data_block;  % The w, h, t block of image data
    persistent use_lab;
    
    %% Initialise and validate persistent state
    
    if isempty(f)
        f = n_frames;
    else
        assert(n_frames == f);
    end
    
    if isempty(current_first_frame)
        current_first_frame = starting_at;
    else
        assert(starting_at == current_first_frame + 1);
        current_first_frame = starting_at;
    end
    
    if isempty(use_lab)
        % We only validate input the first time
        if strcmpi(image_mode, 'lab')
            use_lab = true;
        elseif strcmpi(image_mode, 'greyscale')
            use_lab = false;
        else
            error('Use ''lab'' or ''greyscale'' only.');
        end
    else
        assert(use_lab == strcmpi(image_mode, 'lab'));
    end
        
    
    
    %% Non persistent state

    total_trials = 400;
    total_frames = 60;
    
    current_last_frame = current_first_frame + f - 1;
    
    assert(current_last_frame <= total_frames * total_trials);
    
    %% Data loading
    
    % The first time it's called, we fill up the block
    if isempty(data_block)
        % Load first frame
        data_block = load_frame(current_first_frame, use_lab);
        % Load remaining frames
        frame_count = 2;
        for overall_frame = current_first_frame+1:current_last_frame
            data_block = cat(3, data_block, load_frame(overall_frame, use_lab));
            frame_count = frame_count + 1;
        end%for frame
        
    % After the first block has been cached, we only need to get an
    % additional 1 frame
    else
        % Drop the first frame
        data_block(:,:,1:end-1) = data_block(:,:,2:end);
        
        % Load a new last frame
        data_block(:,:,end) = load_frame(current_last_frame, use_lab);
        
    end%if datablock empty
    
    %% Return the current data block
    ImageSeq = data_block;
    
end%function

%% LOCAL FUNCTIONS

function frame_image = load_frame(frame_id, use_lab)
    total_frames = 60;
    total_trials = 400;

    [trial, frame] = get_trial_and_frame_from_overall_frame(frame_id);
    fprintf('Loading image for trial %03d/%03d, frame %02d/%02d\n', trial, total_trials, frame, total_frames);

    image_path = filename_from_trial_and_frame(trial, frame);
    frame_image = imread(image_path);
    if use_lab
        lab_image = rgb2lab(frame_image);
        % Extract lumanance from LAB image
        frame_image = lab_image(:, :, 1);
    else
        frame_image = rgb2gray(frame_image);
    end

end

function [trial_i, frame_i] = get_trial_and_frame_from_overall_frame(overall_frame)
    total_frames = 60;
    trial_i = ceil(overall_frame / total_frames);
    frame_i = mod(overall_frame, total_frames);
    % Make sure frame 59 is followed by frame 60, not frame 0.
    if frame_i == 0
        frame_i = total_frames;
    end
end

function filename = filename_from_trial_and_frame(trial, frame)
    base_dir = '/Users/cai/Dox/Work/UCL Kymata lab/Data/KYMATA-visual-stimuli-dataset-3_01/video_stimuli_trials';
    filename = fullfile(base_dir, sprintf('trial_%03d', trial), sprintf('frame%02d.png', frame));
end
