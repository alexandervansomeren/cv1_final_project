function bow_image_classification

run('vlfeat-0.9.20/toolbox/vl_setup');
N = 50;  % sample size per category
k_means_sample_size = 10000;
image_data_path = 'Caltech4/ImageData/';
classes_path = strcat('classes_sample_', num2str(N),'.mat');
centroids_path = strcat('debug_centroids_sample_', num2str(N),'.mat');
vocabulary_size = 400;
color_space = 'd_RGB'; % {'d_RGB';'d_rgb';'d_gray';'d_opponent'}


% Check whether SIFT descriptors have already been found for this
% samplesize
if exist(classes_path, 'file')
    load(classes_path);
else
    %% Initialize classes TODO: add test data
    class_names = [{'airplanes'},{'cars'},{'faces'},{'motorbikes'}];
    
    classes = struct([]);
    for current_class = 1:length(class_names)
        image_files = dir(strcat(image_data_path, class_names{current_class}, '_train/'));
        image_files = {image_files.name};
        image_files(1:3) = []; % remove first junk files
        sample = randsample(image_files,N);
        
        classes(current_class).name = class_names{current_class};
        classes(current_class).image_files = image_files;
        classes(current_class).image_files_sample = sample;
    end
    
    %% Feature Extraction and Description
    
    for current_class = 1:length(classes)
        classes(current_class).image_samples = struct([]);
        for i = 1:length(classes(current_class).image_files_sample)
            im_path = strcat(image_data_path, ...
                class_names{current_class},'_train/', ...
                classes(current_class).image_files_sample{i});
            image_RGB = imread(im_path);
            image_rgb = RGB2rgb(image_RGB);
            image_gray = rgb2gray(image_RGB);
            image_opponent = RGB2opponent(image_RGB);
            
            % add image path to data model
            classes(current_class).image_samples(i).im_path = im_path;
            
            % RGB
            d_RGB = {};
            [~, d_RGB.R] = vl_sift(im2single(image_RGB(:,:,1)));
            [~, d_RGB.G] = vl_sift(im2single(image_RGB(:,:,2)));
            [~, d_RGB.B] = vl_sift(im2single(image_RGB(:,:,3)));
            
            % rgb
            d_rgb = {};
            [~, d_rgb.R] = vl_sift(im2single(image_rgb(:,:,1)));
            [~, d_rgb.G] = vl_sift(im2single(image_rgb(:,:,2)));
            [~, d_rgb.B] = vl_sift(im2single(image_rgb(:,:,3)));
            
            % gray
            d_gray = {};
            [~, d_gray.gray] = vl_sift(im2single(image_gray));
            
            % opponent
            d_opponent = {};
            [~, d_opponent.R] = vl_sift(im2single(image_opponent(:,:,1)));
            [~, d_opponent.G] = vl_sift(im2single(image_opponent(:,:,2)));
            [~, d_opponent.B] = vl_sift(im2single(image_opponent(:,:,3)));
            
            % add descriptors to the data model
            classes(current_class).image_samples(i).d_RGB = d_RGB;
            classes(current_class).image_samples(i).d_rgb = d_rgb;
            classes(current_class).image_samples(i).d_gray = d_gray;
            classes(current_class).image_samples(i).d_opponent = d_opponent;
        end
    end
    classes = rmfield(classes, 'image_files');
    classes = rmfield(classes, 'image_files_sample');
    save(classes_path, 'classes');
end

%% Building Visual Vocabulary

% Color spaces (all fields of image_samples, except im_path)
spaces = fieldnames(rmfield(classes(1).image_samples(1), 'im_path'));

% Specify number of descriptor matrices from SIFT. Note that this is
% slightly too big due to the grayscale channel. But we will decrease its
% size later.
n_descriptors_mat = length(classes) ...       % amount of classes
    * N ...               % number of images per class
    * length(spaces) ...  % number of color spaces
    * 3;                  % number of channels

% Create combined matrix of all descriptors
all_descriptors = cell([1, n_descriptors_mat]);
d = 1;  % index for descriptors

for current_class = 1:length(classes)
    for i = 1:2:length(classes(current_class).image_samples) % 1/2 of images
        channels = fieldnames(classes(current_class).image_samples(i).(color_space));
        for k = 1:length(channels)
            all_descriptors{d} = ...
                classes(current_class).image_samples(i).(color_space).(channels{k});
            d = d+1;
        end
    end
end

% Remove empty cells
all_descriptors = all_descriptors(~cellfun('isempty',all_descriptors));

% Convert to matrix
all_descriptors = double(cell2mat(all_descriptors)');

% TAKE SAMPLE (DEBUG!)
some_descriptors = datasample(all_descriptors,k_means_sample_size,...
                                1, 'replace',false);

if exist(centroids_path, 'file')
    load(centroids_path);
else
    % Run parallel k means
    pool = parpool;                      % Invokes workers
    stream = RandStream('mlfg6331_64');  % Random number stream
    options = statset('UseParallel',1,'UseSubstreams',1,...
        'Streams',stream);
    
    [~, centroids] = kmeans(some_descriptors, vocabulary_size, 'Options', options); % , 'MaxIter', 250
    save(centroids_path, 'centroids');
end
%% Quantize Features Using Visual Vocabulary
% for current_class = 1:length(classes)
%    for i = 1:length(classes(current_class).image_samples) % images
%        for j = 1:length(spaces);
%            channels = fieldnames(classes(current_class).image_samples(i).(spaces{j}));
%            for k = 1:length(channels)
%                descriptor = ...
%                    classes(current_class).image_samples(i).(spaces{j}).(channels{k});
%                d = d+1;
%            end
%        end
%    end
% end

end