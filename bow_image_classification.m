function bow_image_classification

run('vlfeat-0.9.20/toolbox/vl_setup');
N = 5;  % sample size per category
image_data_path = 'Caltech4/ImageData/';


%% Initialize classes
class_names = [{'airplanes'},{'cars'},{'faces'},{'motorbikes'}];

classes = struct([]);
for current_class = 1:length(classes)
    image_files = dir(strcat(image_data_path, class_names{current_class}, '_train/'));
    image_files = {image_files.name};
    sample = randsample(image_files,N);
    
    classes(current_class).name = class_names{current_class};
    classes(current_class).image_files = image_files;
    classes(current_class).image_files_sample = sample;
end

%% Feature Extraction and Description

for current_class = 1:length(classes)
    classes(current_class)
    for i = 1:length(classes(current_class).image_files_sample)
        im_path = strcat(image_data_path, ...
            class_names{current_class},'_train/', ...
            classes(current_class).image_files_sample{i});
        image_RGB = imread(im_path);
        image_rgb = RGB2rgb(image_RGB);
        image_gray = rgb2gray(image_RGB);
        image_opponent = RGB2opponent(image_RGB);
        [~, d] = vl_sift(im2single(image_RGB));
    end
end

    
end