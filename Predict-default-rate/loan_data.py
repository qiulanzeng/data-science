import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

#split data set
from sklearn.cross_validation import train_test_split
#Logistic Regression
import statsmodels.api as sm
#Decision tree
from sklearn import linear_model, datasets

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Decision Tree
from sklearn.naive_bayes import GaussianNB

#In the csv file, the data are split into two parts:
# loans meet the credit policy or not meet the credit policy. We may 
#combine them together



#clean the data
url = 'C:/Users/Qiulan/.spyder/1/LoanStats_2017Q2.csv'
loan = pd.read_csv(url, skiprows = 1, low_memory = False)
loan.shape

loan.unique()

features = ['loan_amnt',	'funded_amnt',	'funded_amnt_inv',	'term', 'int_rate',
'installment',	'grade',	'sub_grade'	,'emp_title',	'emp_length',	'home_ownership',	
'annual_inc',	'verification_status',	'loan_status',
'addr_state','dti',	'delinq_2yrs',	'inq_last_6mths',	'mths_since_last_delinq',
'mths_since_last_record',	'open_acc',	'pub_rec',	'revol_bal',	'revol_util',	'total_acc',
'initial_list_status'	,'out_prncp',	'out_prncp_inv'	,'total_pymnt',	'total_pymnt_inv',	'total_rec_prncp',	
'total_rec_int', 'total_rec_late_fee'	,'recoveries'	,'collection_recovery_fee',
'last_pymnt_amnt', 'collections_12_mths_ex_med'	,'mths_since_last_major_derog' ,'application_type',
'acc_now_delinq'	,'tot_coll_amt'	,'tot_cur_bal',	'open_acc_6m',	'open_il_6m'	,'open_il_12m'	,
'open_il_24m',	'mths_since_rcnt_il',	'total_bal_il'	,'il_util'	,'open_rv_12m',	'open_rv_24m',	'max_bal_bc',	
'all_util'	,'total_rev_hi_lim'	,'inq_fi' ,'total_cu_tl'	,'inq_last_12m',	'acc_open_past_24mths'	,
'avg_cur_bal',	'bc_open_to_buy',	'bc_util',	'chargeoff_within_12_mths'	,'delinq_amnt',	'mo_sin_old_il_acct',	
'mo_sin_old_rev_tl_op',	'mo_sin_rcnt_rev_tl_op'	,'mo_sin_rcnt_tl',	'mort_acc'	,'mths_since_recent_bc'	,
'mths_since_recent_bc_dlq',	'mths_since_recent_inq'	,'mths_since_recent_revol_delinq'	,
'num_accts_ever_120_pd'	,'num_actv_bc_tl'	,'num_actv_rev_tl'	,'num_bc_sats'	,'num_bc_tl',	
'num_il_tl'	,'num_op_rev_tl',	'num_rev_accts',	'num_rev_tl_bal_gt_0',	'num_sats'	,'num_tl_120dpd_2m'	,
'num_tl_30dpd',	'num_tl_90g_dpd_24m'	,'num_tl_op_past_12m'	,'pct_tl_nvr_dlq'	,'percent_bc_gt_75',	
'pub_rec_bankruptcies'	,'tax_liens'	,'tot_hi_cred_lim',	'total_bal_ex_mort',	'total_bc_limit'	,
'total_il_high_credit_limit']

len(features)
loan['policy_code'].unique()

loan0 = loan[features]

loan0.isnull().sum().sort_values(ascending = False)

features0 = ['mths_since_last_record','mths_since_recent_bc_dlq', 'mths_since_last_major_derog', 
                            'mths_since_recent_revol_delinq', 'mths_since_last_delinq', 
                            'il_util','mths_since_recent_inq','emp_title', 'sub_grade',
                            'num_tl_120dpd_2m','mo_sin_old_il_acct','mths_since_rcnt_il','bc_util',
                            'percent_bc_gt_75','bc_open_to_buy','mths_since_recent_bc']
features1 = list(set(features).difference(features0))
len(features1)

loan00 = loan[features1]


# First, drop N/A values (from 42,531 reduced to 42,478 rows, not significant)
# Therefore, we will fill in mean values for NaN values later.

loan1 = loan00.dropna() 
loan1.head()
len(loan1)
len(loan00)

loan1.shape
loan1.dtypes
loan1.columns.to_series().groupby(loan1.dtypes).groups


f = plt.figure(figsize=(10,5))
plt.scatter(loan1['annual_inc'], loan1['funded_amnt'])
plt.title("Plotting Annual Income against Funded Amount")
plt.ylabel('Funded Amount')
plt.xlabel('Annual Income')
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Annual_income0.png")

f = plt.figure(figsize=(10,5))
loan1.annual_inc.hist(figsize=(10,5))
plt.ylabel('Number of Loans')
plt.xlabel('Annual Income')
f.savefig("C:/Users/Qiulan/.spyder/1/Annual_income0_hist.png")

#There are several outliers to be accounted for. Lets limit the data to annual income of $250000.
f = plt.figure(figsize=(10,5))
loan2 = loan1[loan1['annual_inc']<600000]
loan2.annual_inc.hist(figsize=(10,5))
plt.ylabel('Number of Loans')
plt.xlabel('Annual Income')
f.savefig("C:/Users/Qiulan/.spyder/1/Annual_income1_hist.png")

#Let's take a quick look at the funded amount. We will plot funded amount both
# from the unfiltered data frame and the filtered data frame 
#(annual income < $,6000,000).

loan1['annual_inc'].describe()

f = plt.figure(figsize=(10,5))
loan1.funded_amnt.hist()
plt.title("Loan with income maximum of $8,900,000.00")
plt.xlabel("Funded Amount")
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/loan_hist0.png")

f = plt.figure(figsize=(10,5))
loan2.funded_amnt.hist()
plt.title("Loan with income maximum of $6,000,000.00")
plt.xlabel("Funded Amount")
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/loan_hist1.png")
#---------------------------------------------------------------
#clean non-ordinal categorical variables with 2 values
#****************************************************************
#loan_status
f = plt.figure(figsize=(10,8))
axis_font = {'fontname':'Arial', 'size':'18'}
plt.xlabel('Loan Status', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
loan2.loan_status.value_counts().plot(kind='bar',alpha=.30, rot =15)
f.savefig("C:/Users/Qiulan/.spyder/1/loan_status_hist0.png")

#cleaning "loan_status"
loan2['loan_status_clean'] = loan2['loan_status'].map({'Current': 2, 
     'Fully Paid': 1, 'Charged Off':0, 'Late(31-120 days)':0, 
     'In Grace Period': 0, 'Late(16-30 days)': 0, 'Default': 0})
loan2 = loan2[loan2.loan_status_clean != 2] 
loan2["loan_status_clean"] = loan2["loan_status_clean"].apply(lambda loan_status_clean: 0 if loan_status_clean == 0 else 1)

f = plt.figure(figsize=(10,8))
axis_font = {'fontname':'Arial', 'size':'18'}
plt.xlabel('Loan Status', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
loan2["loan_status_clean"].value_counts().plot(kind='bar',alpha=.30, rot =0)
f.savefig("C:/Users/Qiulan/.spyder/1/loan_status_hist1.png")

#***************************************************************
loan2.initial_list_status.unique()
loan2['initial_list_status_clean'] = loan2['initial_list_status'].map({'w': 1, 'f': 0})
loan2.initial_list_status_clean.unique()

#---------------------------------------------------------------
#***************************************************************
#clean non-ordinal categorical variables with more than 2 values

#***************************************************************
loan2['application_type'].unique()
loan2['home_ownership'].unique()
loan2['verification_status'].unique()


application_type = pd.get_dummies(loan2.application_type)
loan2 = loan2.join(application_type)

home_ownership = pd.get_dummies(loan2.home_ownership)
loan2 = loan2.join(home_ownership)

verification_status = pd.get_dummies(loan2.verification_status)
loan2 = loan2.join(verification_status)

addr_state = pd.get_dummies(loan2.addr_state)
loan2 = loan2.join(addr_state)

#---------------------------------------------------------------
#***************************************************************
#clean ordinal categorical variables 
#Grade
loan2['grade'].unique()
loan2['grade'] = loan2['grade'].map({'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1})
loan2['grade'].unique()

#---------------------------------------------------------------
#***************************************************************
#Convert data types to float
#***************************************************************
#Clean Employment Length and convert the type to float
loan2['emp_length'] = loan2.emp_length.str.replace('+','')
loan2['emp_length'] = loan2.emp_length.str.replace('<','')
loan2['emp_length'] = loan2.emp_length.str.replace('years','')
loan2['emp_length'] = loan2.emp_length.str.replace('year','')
loan2['emp_length'] = loan2.emp_length.str.replace('n/a','0')
loan2['emp_length']= loan2.emp_length.astype('float64')

loan2.emp_length.unique()


#Clean term and convert the type to float
loan2['term'] = loan2.term.str.replace(' months','')
loan2['term']= loan2.term.astype('float64')


#convert the type of int_rate and revol_util to float.

loan2['int_rate'] = loan2.int_rate.replace('%','',regex=True).astype('float')/100
loan2['revol_util'] = loan2.revol_util.replace('%','',regex=True).astype('float')/100

#---------------------------------------------------------------
#***************************************************************
#check data types
loan2.columns.to_series().groupby(loan2.dtypes).groups
loan2.isnull().sum()
#***************************************************************


# Feature Importance with Extra Trees Classifier

from sklearn.ensemble import ExtraTreesClassifier

X_Variables = list(loan2.columns.values)

X = loan2[X_Variables]
X= X.drop(['application_type','initial_list_status','home_ownership','verification_status','addr_state', 'loan_status_clean','loan_status'], axis = 1)
X.dtypes

Xvar = X.values

yvar = loan2['loan_status_clean'].values

# feature extraction
model = ExtraTreesClassifier()
model.fit(Xvar, yvar)
print model.feature_importances_

feature = pd.Series(model.feature_importances_, index = X.columns.values)
feature = feature.sort_values(ascending = False)

#plot a bar chart

f = plt.figure(figsize=(25, 6))

width = 1/1.5
length = len(feature.index)
#length = 20;
x = list(np.arange(length)+1)
y = feature[:length].values

LABLES = list(feature[:length].index)
plt.bar(x,y , width, color = "blue", align = 'center')
plt.xticks(x, LABLES, rotation='vertical')
plt.show()

f.savefig("C:/Users/Qiulan/.spyder/1/features.png", bbox_inches='tight')


mean_accuracy_lrT = []
mean_accuracy_lrP = []
mean_accuracy_RFT = []
mean_accuracy_RFP = []

for number in range(10, 20, 10):
    impt_feature = feature[:number].index
    
    #shuffle here
    loan2 = loan2.sample(frac=1).reset_index(drop=True)
    
    loan3 = loan2[impt_feature]
    X1 = loan3.values
    y1 = loan2['loan_status_clean'].values
    
    
    accuracy_lrT = []
    accuracy_lrP = []
    accuracy_RFT = []
    accuracy_RFP = []
    
    #***************************************************************
    #Split data
    
    for i in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(X1,y1,test_size=0.6)
    
    
    #***************************************************************
    #Predict using Logistic Regression
    
    
        clf = linear_model.LogisticRegression()
        model_lr = clf.fit(X_train,Y_train)
        accuracy_lrT.append(model_lr.score(X_train, Y_train))
        accuracy_lrP.append(model_lr.score(X_test, Y_test))
    
    
    
    #***************************************************************
    #Predict using Decision Tree
    
        #clf1 = GaussianNB()
        clf1 = RandomForestClassifier(max_depth=5, random_state=0)
        
        model_RF  = clf1.fit(X_train,Y_train)
        accuracy_RFT.append(model_RF.score(X_train, Y_train))
        accuracy_RFP.append(model_RF.score(X_test, Y_test))
        
    #end for loop
    mean_accuracy_lrT.append(np.mean(accuracy_lrT))
    mean_accuracy_lrP.append(np.mean(accuracy_lrP))
    mean_accuracy_RFT.append(np.mean(accuracy_RFT))
    mean_accuracy_RFP.append(np.mean(accuracy_RFP))


from sklearn import metrics
def measure_performance(X,y,model, boolien, show_classification_report=True, show_confusion_matrix=True):
    y_pred=model.predict(X)   

    if show_classification_report:
        if boolien ==1: # For logistic regression model
            print "Classification report for Logistic Regression model "
        if boolien ==0: #For Random forest model
            print "Classification report for Random Forest model "
        print metrics.classification_report(y,y_pred),"\n"
        
    if show_confusion_matrix:
        if boolien ==1: #For Logistic Regression model
            print "Confusion matrix for Logistic Regression model"            
        if boolien == 0: #For Random Forest model
            print "Confusion matrix for Random Forest model"
        print metrics.confusion_matrix(y,y_pred),"\n"    
        
measure_performance(X1,y1,model_lr, 1, show_classification_report=True,
                    show_confusion_matrix=True)

measure_performance(X1,y1,model_RF, 0, show_classification_report=True,
                    show_confusion_matrix=True)


impt_coef_lr = pd.DataFrame(clf.coef_.T, index = impt_feature)
impt_coef_RF = pd.DataFrame(clf1.feature_importances_, index = impt_feature)

impt_coef_lr=impt_coef_lr.sort_values(ascending = False)


