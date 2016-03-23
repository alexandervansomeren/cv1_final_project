function bow_image_classification

run('vlfeat-0.9.20/toolbox/vl_setup');
N = 50;  % sample size per category
kmeans_image_sample_index = 1:N;
svm_image_sample_index = N+1:2*N;
k_means_sample_size = 10000;
image_data_path = 'Caltech4/ImageData/';
classes_path = strcat('classes_sample_', num2str(N),'.mat');
centroids_path = strcat('debug_centroids_sample_', num2str(N),'.mat');
vocabulary_size = 400;
color_space = 'd_RGB'; % {'d_RGB';'d_rgb';'d_gray';'d_opponent';'d_RGB_dense';..}
fast_kmeans = false;
dense_sift_bin_size = 15;

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
        image_files(1:2) = []; % remove first junk files
        
        image_files_test = dir(strcat(image_data_path, class_names{current_class}, '_test/'));
        image_files_test = {image_files_test.name};
        image_files_test(1:2) = []; % remove first junk files
        
        % Sample for kmeans and svm
        sample = randsample(image_files, 2*N, false);
        
        classes(current_class).name = class_names{current_class};
        classes(current_class).image_files = image_files;
        classes(current_class).image_files_sample = sample;
        classes(current_class).image_files_test = image_files_test;
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
%             d_RGB_dense = {};
%             [~, d_RGB_dense.R] = vl_dsift(im2single(image_RGB(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_RGB_dense.G] = vl_dsift(im2single(image_RGB(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_RGB_dense.B] = vl_dsift(im2single(image_RGB(:,:,3)), 'size', dense_sift_bin_size);
            
            % rgb
            d_rgb = {};
            [~, d_rgb.R] = vl_sift(im2single(image_rgb(:,:,1)));
            [~, d_rgb.G] = vl_sift(im2single(image_rgb(:,:,2)));
            [~, d_rgb.B] = vl_sift(im2single(image_rgb(:,:,3)));
%             d_rgb_dense = {};
%             [~, d_rgb_dense.R] = vl_dsift(im2single(image_rgb(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_rgb_dense.G] = vl_dsift(im2single(image_rgb(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_rgb_dense.B] = vl_dsift(im2single(image_rgb(:,:,3)), 'size', dense_sift_bin_size);
            
            % gray
            d_gray = {};
            [~, d_gray.gray] = vl_sift(im2single(image_gray));
            d_gray_dense = {};
            [~, d_gray_dense.gray] = vl_dsift(im2single(image_gray), 'size', dense_sift_bin_size);
            
            % opponent
            d_opponent = {};
            [~, d_opponent.R] = vl_sift(im2single(image_opponent(:,:,1)));
            [~, d_opponent.G] = vl_sift(im2single(image_opponent(:,:,2)));
            [~, d_opponent.B] = vl_sift(im2single(image_opponent(:,:,3)));
%             d_opponent_dense = {};
%             [~, d_opponent_dense.R] = vl_dsift(im2single(image_opponent(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_opponent_dense.G] = vl_dsift(im2single(image_opponent(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_opponent_dense.B] = vl_dsift(im2single(image_opponent(:,:,3)), 'size', dense_sift_bin_size);
            
            
            % add descriptors to the data model
            classes(current_class).image_samples(i).d_RGB = d_RGB;
%             classes(current_class).image_samples(i).d_RGB_dense = d_RGB_dense;
            classes(current_class).image_samples(i).d_rgb = d_rgb;
%             classes(current_class).image_samples(i).d_rgb_dense = d_rgb_dense;
            classes(current_class).image_samples(i).d_gray = d_gray;
            classes(current_class).image_samples(i).d_gray_dense = d_gray_dense;
            classes(current_class).image_samples(i).d_opponent = d_opponent;
%             classes(current_class).image_samples(i).d_opponent_dense = d_opponent_dense;
        end
        classes(current_class).images_test = struct([]);
        for i = 1:length(classes(current_class).image_files_test)
            im_path = strcat(image_data_path, ...
                class_names{current_class},'_test/', ...
                classes(current_class).image_files_test{i});
            image_RGB = imread(im_path);
            image_rgb = RGB2rgb(image_RGB);
            image_gray = rgb2gray(image_RGB);
            image_opponent = RGB2opponent(image_RGB);
            
            % add image path to data model
            classes(current_class).images_test(i).im_path = im_path;
            
            % RGB
            d_RGB = {};
            [~, d_RGB.R] = vl_sift(im2single(image_RGB(:,:,1)));
            [~, d_RGB.G] = vl_sift(im2single(image_RGB(:,:,2)));
            [~, d_RGB.B] = vl_sift(im2single(image_RGB(:,:,3)));
%             d_RGB_dense = {};
%             [~, d_RGB_dense.R] = vl_dsift(im2single(image_RGB(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_RGB_dense.G] = vl_dsift(im2single(image_RGB(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_RGB_dense.B] = vl_dsift(im2single(image_RGB(:,:,3)), 'size', dense_sift_bin_size);
            
            % rgb
            d_rgb = {};
            [~, d_rgb.R] = vl_sift(im2single(image_rgb(:,:,1)));
            [~, d_rgb.G] = vl_sift(im2single(image_rgb(:,:,2)));
            [~, d_rgb.B] = vl_sift(im2single(image_rgb(:,:,3)));
%             d_rgb_dense = {};
%             [~, d_rgb_dense.R] = vl_dsift(im2single(image_rgb(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_rgb_dense.G] = vl_dsift(im2single(image_rgb(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_rgb_dense.B] = vl_dsift(im2single(image_rgb(:,:,3)), 'size', dense_sift_bin_size);
            
            % gray
            d_gray = {};
            [~, d_gray.gray] = vl_sift(im2single(image_gray));
            d_gray_dense = {};
            tic;
            [~, d_gray_dense.gray] = vl_dsift(im2single(image_gray), 'size', dense_sift_bin_size);
            toc;
            
            % opponent
            d_opponent = {};
            [~, d_opponent.R] = vl_sift(im2single(image_opponent(:,:,1)));
            [~, d_opponent.G] = vl_sift(im2single(image_opponent(:,:,2)));
            [~, d_opponent.B] = vl_sift(im2single(image_opponent(:,:,3)));
%             d_opponent_dense = {};
%             [~, d_opponent_dense.R] = vl_dsift(im2single(image_opponent(:,:,1)), 'size', dense_sift_bin_size);
%             [~, d_opponent_dense.G] = vl_dsift(im2single(image_opponent(:,:,2)), 'size', dense_sift_bin_size);
%             [~, d_opponent_dense.B] = vl_dsift(im2single(image_opponent(:,:,3)), 'size', dense_sift_bin_size);
            
            
            % add descriptors to the data model
            classes(current_class).image_test(i).d_RGB = d_RGB;
%             classes(current_class).image_test(i).d_RGB_dense = d_RGB_dense;
            classes(current_class).image_test(i).d_rgb = d_rgb;
%             classes(current_class).image_test(i).d_rgb_dense = d_rgb_dense;
            classes(current_class).image_test(i).d_gray = d_gray;
            classes(current_class).image_test(i).d_gray_dense = d_gray_dense;
            classes(current_class).image_test(i).d_opponent = d_opponent;
%             classes(current_class).image_test(i).d_opponent_dense = d_opponent_dense;
        end
    end
    classes = rmfield(classes, 'image_files');
    classes = rmfield(classes, 'image_files_sample');
    classes = rmfield(classes, 'image_files_test');
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
    for i = 1:2:length(classes(current_class).image_samples(kmeans_image_sample_index)) % 1/2 of images
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
elseif (fast_kmeans)
    [~, centroids, ~] = fkmeans(all_descriptors, vocabulary_size);
else
    % Run parallel k means
    pool = parpool;                      % Invokes workers
    stream = RandStream('mlfg6331_64');  % Random number stream
    options = statset('UseParallel',1,'UseSubstreams',1,...
        'Streams',stream);
    
    [~, centroids] = kmeans(all_descriptors, vocabulary_size, 'Options', options); % , 'MaxIter', 250
    delete(gcp('nocreate')); % quit parallel pool
    save(centroids_path, 'centroids');
end

%% Quantize Features Using Visual Vocabulary
for current_class = 1:length(classes)
    for i = 1:length(classes(current_class).image_samples) % images
        d = 1;
        image_descriptors = cell([1 length(spaces)*3]);
        for j = 1:length(spaces);
            channels = fieldnames(classes(current_class).image_samples(i).(spaces{j}));
            for k = 1:length(channels)
                descriptor = ...
                    classes(current_class).image_samples(i).(spaces{j}).(channels{k});
                image_descriptors{d} = descriptor;
                d = d+1;
            end
        end
        image_descriptors = image_descriptors(~cellfun('isempty', image_descriptors));
        h = getHistFromDescriptor(double(cell2mat(image_descriptors)'), centroids);
        classes(current_class).image_samples(i).histogram = h;
    end
end

save('debug_classes_hist.mat', 'classes');



%% Classification

% Train models
for current_class_svm = 1:length(classes)
    % Create trainset table
    svm_train_data = array2table(zeros(length(classes)*N,vocabulary_size+1));
    svm_train_data.Properties.VariableNames(vocabulary_size+1) = {'class'};
    d_index = 1;
    % Retrieve histograms
    for current_class = 1:length(classes)
        if strcmp(classes(current_class).name, classes(current_class_svm).name)
            c = 1;
        else
            c = 2;
        end
        for i = 1:length(classes(current_class).image_samples(svm_image_sample_index)) % images
            svm_train_data(d_index,1:vocabulary_size) = ...
                array2table(classes(current_class).image_samples(i).histogram);
            svm_train_data(d_index,vocabulary_size+1) = {c};
            d_index = d_index + 1;
        end
    end
    classes(current_class_svm).svm_model = fitcsvm(svm_train_data, 'class');
end

save('debug_svm_models.mat', 'classes');

%% Evaluation
for eval_class = 1:length(classes)
    hists_to_predict = zeros(length(classes(current_class).images_test)* ...
                                length(classes), vocabulary_size);
    for current_class = 1:length(classes)
        for i = 1:length(classes(current_class).images_test)  % images
            image_descriptors = cell([1 length(spaces)*3]);
            for j = 1:length(spaces);
                channels = fieldnames(classes(current_class).images_test(i).(spaces{j}));
                for k = 1:length(channels)
                    descriptor = ...
                        classes(current_class).image_samples(i).(spaces{j}).(channels{k});
                    image_descriptors{d} = descriptor;
                end
            end
            image_descriptors = image_descriptors(~cellfun('isempty', image_descriptors));
            hists_to_predict(i,:) = getHistFromDescriptor(double(cell2mat(image_descriptors)'), centroids);
        end
    end
    [label,Score] = classes(eval_class).svm_model.Methods.predict(hists_to_predict);
end

end



%% Helper functions

% Helper function that creates the histogram given a matrix of descriptors
% and a matrix of centroids
function h = getHistFromDescriptor(descriptors, centroids)
D = pdist2(descriptors,centroids);
[~,I] = min(D, [], 2);
h = hist(I, size(centroids,1));
end