% 针对FC脑连接矩阵对称性，进行输出修正  2020-11-17
%   X = [temp； X]; ---》 X = [X; temp];             % 2022-5-7
%   PU_1个数已经进行修正，为85个                      % 2022-5-15

function [train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label]=svm_two_suppose_FC(index, data_name)
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% data_name = 'KKI_data';
% index = 1;

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};

% [ methodID ] = readInput( listFS );
selection_method = listFS{10}; % Selected rfe

% Load the data and select features for classification
% load fisheriris
load([data_name '.mat'])
% X_temp = inform.brain_conn_show;    % FC对称矩阵
X_temp = inform.brain_conn;                 % FC上三角矩阵
Y_temp = inform.tag_C;
X = [];

for i=1:max(size(X_temp))    
   temp = X_temp{1,i}';
   X = [X; temp];             % 2022-5-7
end

% X = meas; clear meas
% Extract the Setosa class
Y = nominal(ismember(Y_temp,1)); 

%[train_h0_data, train_h0_label, test_h0_label] = train_h0(index, X, Y, selection_method, Y_temp);
[train_h0_data, train_h0_label, test_h0_label] = train_h0_1(index, X, Y, selection_method, Y_temp,data_name);

[train_h0_data, train_h0_label] = energy_normalization(train_h0_data, train_h0_label);

[train_h1_data, train_h1_label, test_h1_label] = train_h1(index, X, Y, selection_method, Y_temp);

[train_h1_data, train_h1_label] = energy_normalization(train_h1_data, train_h1_label);
end

function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    % sample_energy_map2 = [agv_energy_1*ones(1,length(sample_energy_tmp(train_label))) avg_energy_0*ones(1,length(sample_energy_tmp(~train_label)))];
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h0_data, train_h0_label,  test_h0_label] = train_h0(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));   

%numF = size(X_train,2);
% numF = 110;   %tyb 2020-9-4
numF = 50;   %tyb 2020-11-17

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

   % k = 110; % select the first 110 features
    k = numF; % select the first 55 features

    %svmStruct = fitcsvm(X_train(:,ranking<=k),Y_train,'Standardize',true,'KernelFunction','RBF',...
    %'KernelScale','auto','OutlierFraction',0.0);

    %C = predict(svmStruct,X_train(:,ranking<=k));
    %err_rate = sum(Y_train~= C)/max(size(Y_train)); % mis-classification rate
    %% conMat = confusionmat(Y_test,C); % the confusion matrix
    %fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',...
    %    selection_method,100*(1-err_rate),err_rate);
   
 
    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y_temp;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);

    return
end

function [train_h0_data, train_h0_label,  test_h0_label] = train_h0_1(index, X, Y, selection_method, Y_temp, data_name)

X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));   


numF = 50;   %tyb 2020-11-17
k = numF; % select the first 55 features

% 给前一百的值
if strcmp(data_name,'Peking_1_data')
    ranking=[36,138,188,206,282,286,350,364,462,559,679,898,1111,1431,1447,1541,1564,1687,1773,1793,1839,1840,1843,1941,2001,2087,2383,2407,2431,2439,2565,2637,2670,2742,2774,2784,3033,3038,3039,3068,3345,3376,3486,3589,3674,3679,3686,3765,3772,3799,1471,500,3232,2561,1341,3238,2405,2964,1110,3678,32,122,3146,533,532,2289,1551,501,1809,2456,2538,2782,3988,766,999,463,3788,78,3966,2819,2640,2914,912,780,116,3766,338,2347,3424,2137,42,1563,1387,1567,2192,2023,44,3714,3415,1923];
end
 
if strcmp(data_name,'Peking_data')
    ranking=[14,282,285,289,489,561,589,759,780,817,818,829,833,973,987,999,1142,1317,1343,1431,1742,1781,1793,1846,1972,2011,2087,2133,2142,2145,2209,2269,2716,2808,2972,2975,3117,3145,3167,3198,3208,3547,3607,3608,3664,3714,3750,3801,3816,3951,1687,22,3394,3718,3970,149,3020,3415,3288,3150,3337,1840,1413,3967,1971,2743,1346,268,1816,3655,1780,3202,1014,1974,2377,2029,2813,377,157,736,1176,2027,2633,1783,1975,3064,728,396,3479,2737,580,1933,1842,1862,2903,2370,857,1830,1324,2033];
end

if strcmp(data_name,'NI_data')
    ranking=[77,98,184,259,269,272,404,419,500,503,884,1025,1079,1097,1146,1163,1174,1198,1242,1243,1268,1322,1394,1724,1845,1916,1941,2070,2090,2104,2558,2627,2881,3052,3209,3273,3450,3525,3578,3580,3584,3649,3678,3709,3760,3874,3908,3962,3980,3991,1373,3120,1582,2400,1726,3934,1269,276,3274,267,3211,1781,841,1946,266,237,1942,3766,82,2718,253,448,2903,2133,247,3675,1917,2096,3659,3680,2023,1478,180,3717,3926,2628,1884,977,1706,166,1830,3248,252,174,535,885,1157,3444,1714,3527];
end


if strcmp(data_name,'NYU_data')
    ranking=[18,30,88,297,358,462,491,502,511,543,562,767,795,821,897,936,1111,1226,1229,1306,1308,1361,1389,1492,1558,1653,1900,1909,2029,2159,2489,2517,2602,2723,2794,2887,2910,2918,2965,3139,3337,3385,3469,3500,3593,3690,3708,3740,3958,3969,1071,1429,2627,2943,3713,1080,354,637,1629,1630,3586,1557,2161,3388,3897,2967,3353,3794,2752,3234,2335,2511,545,506,283,1058,304,3089,2547,2103,3473,3324,2963,2104,3833,580,3321,41,2397,1079,20,3465,65,2257,1717,2415,2626,3727,3923,2493];
end


if strcmp(data_name,'KKI_data')
    ranking=[62,191,211,260,348,362,374,387,417,504,669,745,982,1035,1160,1297,1404,1540,1724,1735,1834,1895,1944,1945,1960,2083,2224,2519,2520,2589,2684,2702,2936,2945,2946,3140,3171,3431,3502,3516,3557,3572,3576,3586,3626,3731,3913,3932,3967,4000,2503,3815,2280,817,3481,2789,3585,819,3796,2775,2707,2235,603,943,445,1474,2501,1332,3733,103,3550,2518,219,3006,3802,986,3568,1837,1353,1645,3339,1796,2851,629,1908,1026,2632,51,238,710,940,978,3498,1859,1352,595,52,2506,3255,1452];
end

    train_h0_data = X_train(:,ranking(1:k));
    train_h0_label = Y_temp;

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];

    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);

    return
end

function [train_h1_data, train_h1_label, test_h1_label] = train_h1(index, X, Y, selection_method, Y_temp)

X_train = double(X);
Y_temp(index) = ~Y_temp(index);
if Y(index) == 'true'
    Y(index,1) = 'false';
else
    Y(index,1) = 'true';
end

Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h1_label = double(Y(index));

%numF = size(X_train,2);
% numF = 110;
numF = 50;   %tyb 2020-11-17

% feature Selection on training data
    switch lower(selection_method)
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights, subset] = ILFS_auto(X_train, Y_train , 4, 0 );
        case 'mrmr'
            ranking = mRMR(X_train, Y_train, numF);
        
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
        
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
        
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
        
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
        
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 2;
            ranking = UDFS(X_train , nClass ); 
        
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);     
        
        case 'llcfs'   
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
        
        otherwise
            disp('Unknown method.')
    end

    %k = 110; % select the first 110 features
    k =numF; % select the first 55 features
    
    
    %svmStruct = fitcsvm(X_train(:,ranking<=k),Y_train,'Standardize',true,'KernelFunction','RBF',...
    %'KernelScale','auto','OutlierFraction',0.0);

    %C = predict(svmStruct,X_train(:,ranking<=k));
    %err_rate = sum(Y_train~= C)/max(size(Y_train)); % mis-classification rate
    %% conMat = confusionmat(Y_test,C); % the confusion matrix
    %fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',...
    %    selection_method,100*(1-err_rate),err_rate);

   
    
    train_h1_data = X_train(:,ranking(1:k));
    train_h1_label = Y_temp;

    train_h1_data(index,:)=[];
    train_h1_label(index)=[];

    [~,indx_1] = sort(train_h1_label,'descend');
    train_h1_label= train_h1_label(indx_1);
    train_h1_data = train_h1_data(indx_1,:);

    return
end
