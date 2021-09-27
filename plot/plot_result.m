%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(s-1)

sample_num = 2;

%input_files = strings([1,3]);
input_files_A = {
    '../data/Darcy_1_7_0.5/output1_7_test_100.mat' ,  '../results/train_12_3_test_1_7/12_3_with_u_norm_test_model.mat', '12-3-u-norm-test';
    
   '../data/Darcy_1_7_0.5/output1_7_test_100.mat' , '../results/train_12_3_test_1_7/last_layer/12_3_with_u_norm_test_model_train_20.mat', '12-3-u-norm-test_last_layer';
   
   '../data/Darcy_1_7_0.5/output1_7_test_100.mat' , '../results/train_12_3_test_1_7/last_layer/12_3_with_u_norm_test_model_train_1000.mat', '12-3-u-norm-test_last_layer_1000';
};

input_files = string(input_files_A);
disp(input_files(1,3));
coeff_idx = 92;
learn_idx = 12;
s = 421;
for sample_idx = 1:sample_num
        coeff_file = input_files(sample_idx,1);
        learn_file = input_files(sample_idx,2);
        case_name = input_files(sample_idx,3);
        plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
  
end


% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/load_model_12_3/12_3_with_norm_train.model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =1;
% case_name = '12-3-with-norm-train';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/12_3_with_norm_test.model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =2;
% case_name = '12-3-with-norm-test';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_1_7_test_1_7/1_7_with_norm_train_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =3;
% case_name = '1-7-with-train';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_1_7_test_1_7/1_7_no_norm_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =4;
% case_name = '1-7-no-norm';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/load_model_12_3/12_3_with_norm_a_train.model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =5;
% case_name = '12-3-a-norm-train';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/load_model_12_3/12_3_with_norm_u_train.model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =6;
% case_name = '12-3-u-norm-train';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/12_3_with_a_norm_test_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =7;
% case_name = '12-3-a-norm-test';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/12_3_with_u_norm_test_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =8;
% case_name = '12-3-u-norm-test';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/last_layer/12_3_with_a_norm_test_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =9;
% case_name = '12-3-a-norm-test-last-layer-train-with 1000';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/last_layer/12_3_with_a_norm_test_model_train_20.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =10;
% case_name = '12-3-a-norm-test-last-layer-train-with 20';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/12_3_no_norm_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =11;
% case_name = '12-3-no-norm';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
% 
% 
% 
% coeff_file = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
% learn_file = '../results/train_12_3_test_1_7/12_3_with_norm_train_test_model.mat';
% coeff_idx = 92;
% learn_idx = 12;
% s = 421;
% sample_idx =12;
% case_name = '12-3-with-norm-train-test';
% plot_onefile(sample_num,sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)


function coeff = plot_onefile(sample_num, sample_idx,coeff_file,coeff_idx,learn_file,learn_idx,s,case_name)
    [X,Y] = meshgrid(0:(1/(s-1)):1);
    %filename = '../data/Darcy_1_7_0.5/output1_7_test_100.mat';
    coeff= squeeze(load(coeff_file).coeff(coeff_idx,:,:));
    generate_ground = squeeze(load(coeff_file).sol(coeff_idx,:,:));

    %Plot coefficients and solutions
     subplot(sample_num,5,5*(sample_idx-1)+1);
     surf(X,Y,coeff); 
     
     view(2); 
     shading interp;
     colorbar;
     subtitle(strcat(case_name,' coeff'))
     
     subplot(sample_num,5,5*(sample_idx-1)+2);
     surf(X,Y,generate_ground); 
     view(2); 
     shading interp;
     colorbar;
     subtitle(strcat(case_name,' sol generate'))

     [X,Y] = meshgrid(0:(5/(s-1)):1);
    %filename = '../results/train_12_3_test_1_7/12_3_no_norm_model.mat';
     ground_truth = squeeze(load(learn_file).sol_ground(learn_idx,:,:));
     learn_result = squeeze(load(learn_file).sol_learn(learn_idx,:,:));
     
     subplot(sample_num,5,5*(sample_idx-1)+3);
     surf(X,Y,ground_truth); 
     view(2); 
     shading interp;
     colorbar;
     subtitle(strcat(case_name,' sol ground truth'))
     
     subplot(sample_num,5,5*(sample_idx-1)+4);
     surf(X,Y,learn_result); 
     view(2); 
     shading interp;
     colorbar;
     subtitle(strcat(case_name,' sol learned'))
     
     subplot(sample_num,5,5*(sample_idx-1)+5);
     surf(X,Y,ground_truth-learn_result); 
     view(2); 
     shading interp;
     colorbar;
     subtitle(strcat(case_name,' sol error'))
     
end


%Create mesh (only needed for plotting)



% subplot(2,2,3)
% surf(X,Y,thresh_a); 
% view(2); 
% shading interp;
% colorbar;
% subplot(2,2,4)
% surf(X,Y,thresh_p); 
% view(2); 
% shading interp;
% colorbar;

 
 
 


