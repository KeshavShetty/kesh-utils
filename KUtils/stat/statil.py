import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import normaltest, ttest_ind, chi2_contingency
    
def run_stat_test(df, target_feature_name, p_value_cutoff=0.05):
    
    df_categorical = df.select_dtypes(include=['object', 'category'])
    
    categorical_column_names = list(df_categorical.columns)
    categorical_column_names.remove(target_feature_name)
    numerical_column_names =  [i for i in df.columns if not i in df_categorical.columns]
    
    for col_name in categorical_column_names:
        binary_classification_chi_test_for_categorical(df, col_name, target_feature_name, p_value_cutoff)
        
    for col_name in numerical_column_names:
        binary_classification_normal_stat_clt_test(df, col_name, target_feature_name, p_value_cutoff)    

def binary_classification_chi_test_for_categorical(df, feature_to_examine, target_feature_name, p_value_cutoff=0.05):
    cr_tab = pd.crosstab(df[feature_to_examine],df[target_feature_name])
    chi_stat = chi2_contingency(cr_tab)
    print('Running stats on ' + feature_to_examine)
    print(f'  Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')
    
    if chi_stat[1]<p_value_cutoff:
        print('\tRejecting null hypothesis that ' + feature_to_examine + ' is not significant.')
        print('\tSo Feature has significance and prdective power')
    else:
        print('\tCannot Reject null hypothesis that ' + feature_to_examine + ' is not significant.')
        print('\tSo may not be significant or has any predictive power)')
    print('----------------------------------------------------------------------------------------------------------')
    
def binary_classification_normal_stat_clt_test(df, feature_to_examine, target_feature_name, p_value_cutoff=0.05) :

    print('Running stats on ' + feature_to_examine)    
    df[target_feature_name] = df[target_feature_name].astype('category')
    
    #cat_unique_list = list(df[target_feature_name].unique())
    
    cat_values_counts = df[target_feature_name].value_counts()
    
    cat_values_counts = cat_values_counts.sort_index()
    
    negative_class_label = cat_values_counts.index[0]
    positive_class_label = cat_values_counts.index[1]
        
    negative_df = df[df[target_feature_name]==negative_class_label][feature_to_examine]
    positive_df = df[df[target_feature_name]==positive_class_label][feature_to_examine]
        
    
    # feature_to_examine Mean and Median of full dataset
    fullset_mean=round(df[feature_to_examine].mean(),2)
    fullset_median=df[feature_to_examine].median()
    
    # feature_to_examine Mean and Median of negative dataset
    negative_median=negative_df.median()
    negative_mean=round(negative_df.mean(),2)
    
    # feature_to_examine Mean and Median of positive dataset
    positive_median=positive_df.median()
    positive_mean=round(positive_df.mean(),2)
    
    negative_means = []
    positive_means = []
    sample_means=[]
    
    negative_median=negative_df.median()
    negative_mean=round(negative_df.mean(),2)
    
    positive_median=positive_df.median()
    positive_mean=round(positive_df.mean(),2)
    
    # Test Central Limit Theorem (CLT)  by sampling
    for _ in range(1000):
        samples = df[feature_to_examine].sample(n=200)
        sampleMean = np.mean(samples)
        sample_means.append(sampleMean)
        
        samples = negative_df.sample(n=100)
        sampleMean = np.mean(samples)
        negative_means.append(sampleMean)
        
        samples = positive_df.sample(n=100)
        sampleMean = np.mean(samples)
        positive_means.append(sampleMean)
        
    fig, ax = plt.subplots(1,3,figsize=(16,6))   
    
    ax[0].axvline(fullset_median, color='blue', linestyle='-')
    ax[0].axvline(fullset_mean, color='blue', linestyle='--')    
    ax[0]=sns.distplot(df[feature_to_examine],bins=15,ax=ax[0])
    ax[0].set_xlabel(feature_to_examine + ' [Full dataset Mean:'+ str(fullset_mean) + ' ,Median:' + str(fullset_median) +']')
    
    ax[1].axvline(positive_median, color='green', linestyle='-')
    ax[1].axvline(positive_mean, color='green', linestyle='--')
    ax[1].axvline(negative_median, color='red', linestyle='-')
    ax[1].axvline(negative_mean, color='red', linestyle='--')
    ax[1]=sns.kdeplot(positive_df, label='Positive',shade=True, ax=ax[1], color="green")
    ax[1]=sns.kdeplot(negative_df, label='Negative', shade=True, ax=ax[1], color="red")
    ax[1].set_xlabel(feature_to_examine + '\n [-ve class Mean:'+ str(negative_mean) + ' ,-ve class Median:' + str(negative_median) +']' + '\n[+ve class Mean:'+ str(positive_mean) + ' ,+ve class Median:' + str(positive_median) +']')
    
    ax[2].axvline(positive_median, color='green', linestyle='-')
    ax[2].axvline(positive_mean, color='green', linestyle='--')
    ax[2].axvline(negative_median, color='red', linestyle='-')
    ax[2].axvline(negative_mean, color='red', linestyle='--')        
    ax[2] =sns.kdeplot(positive_means, label='+ve Class',shade=True,ax=ax[2], color="green")
    ax[2] =sns.kdeplot(negative_means, label='-ve Class', shade=True,ax=ax[2], color="red")
    
    ax[2].set_xlabel(feature_to_examine)
    ax[2].set_xlabel(feature_to_examine)
    ax[2].set_ylabel('Kernel Density Estimate')    
    ax[2].set_title('Appled Central Limit Theorem')    
    
    plt.show()
    
    chi_stat, p_value = normaltest(df[feature_to_examine], axis=0)
    print(f"Normal Test for the feature {feature_to_examine} distribution chi_stat={chi_stat} p-value={p_value}") 
    if p_value<p_value_cutoff:
        print(f'\tLow p-value(<{p_value_cutoff}) indicates it is unlikely that data came from a normal distribution.(NOT Normal)')
    else:
        print(f'\t**p-value>{p_value_cutoff} indicates it is most likely data came from a normal distribution.')
        
    #Null hypothesis : data came from a normal distribution.    
    # If the p-val is very small, it means it is unlikely that the data came from a normal distribution
    
    print('Skewness Interpretation: Fairly symmetrical if the skewness is between -0.5 and 0.5)')
    fullset_skew = pd.DataFrame.skew(df[feature_to_examine], axis=0)    
    print_skewness_report(fullset_skew, 'Full dataset')
    
    negative_df_skew = pd.DataFrame.skew(negative_df, axis=0)
    print_skewness_report(negative_df_skew, 'Negative dataset')
    
    positive_df_skew = pd.DataFrame.skew(positive_df, axis=0)
    print_skewness_report(positive_df_skew, 'Positive dataset')
    
    #t-test on independent samples
    t2, p2 = ttest_ind(positive_df,negative_df)
    print("ttest_ind: t = %g  p = %g" % (t2, p2))
    if p2<p_value_cutoff:
        print('Rejecting null hypothesis that there is no difference in Mean of +ve and -ve Class.(Go for alternative hypothesis)')
        print('\tThe feature ' + feature_to_examine + ' would be a predictive feature. Count it as important feature')
    else:
        print('Cannot Reject null hypothesis that there is no difference in Mean of +ve and -ve Class')
        print('\tThe feature ' + feature_to_examine + ' may NOT be important or has any predictive power')
    
    print('----------------------------------------------------------------------------------------------------------')
    
def print_skewness_report(df_skew, df_title):
    print(f"\tSkewness for the {df_title} {df_skew}.") 
    if df_skew<0: # Nagative
        print('\tLeft skewed')
        if df_skew<-0.5:
            print('\tdata NOT symmetrical')
        else:
            print('\tData seems to be fairly symmetrical')
    elif df_skew>0: # Positive
        print('\tRight skewed')
        if df_skew>0.5:
            print('\tdata NOT symmetrical')
        else:
            print('\tData seems to be fairly symmetrical')
    else:
        print('\tHighly symmetrical')
        

     