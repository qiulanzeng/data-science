import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import scipy.stats

url = r'C:/Users/Qiulan/.spyder/1/RejectStats_2017Q2.csv'
loan = pd.read_csv(url, skiprows = 1, low_memory = False)
loan.dtypes


np.random.seed(sum(map(ord, "distributions")))


#**************************************************
#Amount_Requested
loan['Amount Requested'].describe()
f = plt.figure()
sns.distplot(loan['Amount Requested'])
f.savefig("C:/Users/Qiulan/.spyder/1/Amount_Requested.png")

f = plt.figure()
plt.boxplot(loan['Amount Requested'], 0, 'rs', 0)
f.savefig("C:/Users/Qiulan/.spyder/1/Amount_Requested_box.png")

x=loan['Amount Requested']

fig, ax = plt.subplots()
kde =scipy.stats.gaussian_kde(x,bw_method=0.3)
dist_space = np.linspace( min(x), max(x), 100 )
ax.plot( dist_space, kde(dist_space) )
ax.hist(x, 50, normed = True, facecolor='lightgray')

axis_font = {'fontname':'Arial', 'size':'14'}
plt.xlabel('Amount Requested', **axis_font)
plt.ylabel('Frequency', **axis_font)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
fig.savefig("C:/Users/Qiulan/.spyder/1/Amount_Requested.png")



#**************************************************
#Risk_Score
loan.Risk_Score.describe()
loan.Risk_Score.isnull().value_counts(normalize=True)
Risk_Score_dropna = loan.Risk_Score.dropna() 


f = plt.figure()
sns.distplot(Risk_Score_dropna)
f.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_dropna.png")

f = plt.figure()
plt.boxplot(Risk_Score_dropna, 0, 'rs', 0)
f.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_dropna_box.png")

x=Risk_Score_dropna
fig, ax = plt.subplots()
kde =scipy.stats.gaussian_kde(x,bw_method=0.3)
dist_space = np.linspace( min(x), max(x), 100 )
ax.plot( dist_space, kde(dist_space) )
ax.hist(x, 50, normed = True, facecolor='lightgray')

axis_font = {'fontname':'Arial', 'size':'14'}
plt.xlabel('Risk Score', **axis_font)
plt.ylabel('Frequency', **axis_font)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
fig.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_dropna.png")

#**************************************************
#Debt-To-Income Ratio

loan['Debt-To-Income Ratio'] = loan['Debt-To-Income Ratio'].replace('%','',regex=True).astype('float')/100
loan['Debt-To-Income Ratio'].isnull().value_counts(normalize=True)
loan['Debt-To-Income Ratio'].describe()
f = plt.figure()
sns.distplot(loan['Debt-To-Income Ratio'])
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_To_Income_Ratio.png")

f = plt.figure()
plt.boxplot(loan['Debt-To-Income Ratio'], 0, 'rs', 0)
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_To_Income_Ratio_box.png")



#filter

x = loan[loan['Debt-To-Income Ratio']<1]['Debt-To-Income Ratio']

x=x.dropna()
x.isnull().value_counts(normalize=True)


f = plt.figure()
sns.distplot(x)
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_To_Income_Ratio_filter.png")


fig, ax = plt.subplots()
kde =scipy.stats.gaussian_kde(x,bw_method=0.2)
dist_space = np.linspace( min(x), max(x), 100 )
ax.plot( dist_space, kde(dist_space) )
ax.hist(x, 50, normed = True, facecolor='lightgray')

axis_font = {'fontname':'Arial', 'size':'14'}
plt.xlabel('Debt to Income Ratio', **axis_font)
plt.ylabel('Frequency', **axis_font)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
fig.savefig("C:/Users/Qiulan/.spyder/1/Debt_To_Income_Ratio_filter.png")


#**************************************************
#emp_length
loan['Employment Length'].describe()
loan['Employment Length'] = loan['Employment Length'] .str.replace('+','')
loan['Employment Length']  = loan['Employment Length'] .str.replace('<','')
loan['Employment Length']  = loan['Employment Length'] .str.replace('years','')
loan['Employment Length']  = loan['Employment Length'] .str.replace('year','')
loan['Employment Length']  = loan['Employment Length'] .str.replace('n/a','0')
loan['Employment Length'] =loan['Employment Length'].astype('float64')
loan['Employment Length'].unique()

x = loan['Employment Length']
x.isnull().value_counts(normalize=True)



f = plt.figure()
pd.value_counts(x).plot.bar()

f.savefig("C:/Users/Qiulan/.spyder/1/Employment_length.png")

x.describe()

#**************************************************
#Count_of_Declined_Loan_Application Date


loan["Application Date"] = loan["Application Date"].astype("datetime64")

f = plt.figure(figsize=(24, 12))
loan["Application Date"].groupby([loan["Application Date"].dt.year, 
              loan["Application Date"].dt.month,
              loan["Application Date"].dt.date]).count().plot(marker = "o", rot = 20)

axis_font = {'fontname':'Arial', 'size':'18'}
plt.xlabel('Date', **axis_font)
plt.ylabel('Count of Declined Loan', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Declined_loan_time.png")
#f.savefig("C:/Users/Qiulan/.spyder/1/Declined_loan_time.png", bbox_inches='tight')



#**************************************************
#Debt_to_income_ratio_time
f = plt.figure(figsize=(24, 12))
loan["Debt-To-Income Ratio"].groupby([loan["Application Date"].dt.year, 
              loan["Application Date"].dt.month,
              loan["Application Date"].dt.date]).mean().plot(marker='o', rot =20)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Date', **axis_font)
plt.ylabel('Debt to Income Ratio', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_to_income_ratio_time.png")


#**************************************************
#Amount_Requested_time
f = plt.figure(figsize=(24, 12))
loan["Amount Requested"].groupby([loan["Application Date"].dt.year, 
              loan["Application Date"].dt.month,
              loan["Application Date"].dt.date]).mean().plot(marker='o', rot = 20)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Date', **axis_font)
plt.ylabel('Amount Requested', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Amount_Requested_time.png")

#**************************************************
#Risk_score_time
f = plt.figure(figsize=(24, 12))
loan["Risk_Score"].groupby([loan["Application Date"].dt.year, 
              loan["Application Date"].dt.month,
              loan["Application Date"].dt.date]).mean().plot(marker='o', rot = 20)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Date', **axis_font)
plt.ylabel('Risk Score', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_time.png")

#**************************************************
#Employment_length_time
f = plt.figure(figsize=(24, 12))
data0=loan["Employment Length"].groupby([loan["Application Date"].dt.year, 
              loan["Application Date"].dt.month,
              loan["Application Date"].dt.date]).mean()
data0.plot(marker='o', rot = 20)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Date', **axis_font)
plt.ylabel('Employment Length', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Employment_length_time.png")
loan["Employment Length"].describe()



#------------------------------------------------------------
#The following is for year-month-date

loan["Application Date"] = loan["Application Date"].astype("datetime64")

data0=loan.groupby([loan["Application Date"].dt.month])["Application Date"].count()
data=loan.groupby([loan["Application Date"].dt.month]).mean()
data.index.names = ['Month']

data.loc[:, ~data.columns.str.contains('Policy Code')]

#**************************************************
#Debt_to_income_ratio_yeartime
f = plt.figure(figsize=(10, 6))
data1=loan["Debt-To-Income Ratio"].groupby([loan["Application Date"].dt.year,
                                     loan["Application Date"].dt.month]).mean()
data1.plot(marker='o', rot=20)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Year and Month', **axis_font)
plt.ylabel('Debt to Income Ratio', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_to_income_ratio_yeartime.png")


#**************************************************
#Amount_Requested_yeartime
f = plt.figure(figsize=(10, 6))
data2=loan["Amount Requested"].groupby([loan["Application Date"].dt.year,
                                     loan["Application Date"].dt.month]).mean()
data2.plot(marker='o')
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Year', **axis_font)
plt.ylabel('Amount Requested', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Amount_Requested_yeartime.png")

#**************************************************
#Risk_score_yeartime
f = plt.figure(figsize=(10, 6))
data3=loan["Risk_Score"].groupby([loan["Application Date"].dt.year,
                                     loan["Application Date"].dt.month]).mean()
data3.plot(marker='o')
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Year', **axis_font)
plt.ylabel('Risk Score', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_yeartime.png")

#**************************************************
#Employment_length_yeartime
f = plt.figure(figsize=(10, 6))
data4=loan["Employment Length"].groupby([loan["Application Date"].dt.year,
                                     loan["Application Date"].dt.month]).mean()
data4.plot(marker='o')
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('Year', **axis_font)
plt.ylabel('Employment Length', **axis_font)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
f.savefig("C:/Users/Qiulan/.spyder/1/Employment_length_yeartime.png")
loan["Employment Length"].describe()


#---------------------------------------------------------
#**************************************************
#Declined loan of states

loan["Application Date"] = pd.to_datetime(loan["Application Date"])

f = plt.figure(figsize=(20, 10))
data = loan.groupby([loan["State"],                    
                     loan["Application Date"].dt.month])["State"].count().unstack()
data.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Count of Declined Loan', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Declined_loan_states.png", bbox_inches='tight')


data_CA_percentage = data.ix['CA'].divide(data.mean())
data_FL_percentage = data.ix['FL'].divide(data.mean())
data_TX_percentage = data.ix['TX'].divide(data.mean())
data_NY_percentage = data.ix['NY'].divide(data.mean())
data_WV_percentage = data.ix['WV'].divide(data.mean())





f = plt.figure(figsize=(20, 10))
data1 = loan.groupby([loan["State"],                    
                     loan["Application Date"].dt.month])['Amount Requested'].mean().unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Amount Requested', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Amount_requested_states.png", bbox_inches='tight')



f = plt.figure(figsize=(20, 12))
data1 = loan.groupby([loan["State"],                    
                     loan["Application Date"].dt.month])['Employment Length'].mean().unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Employment Length', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Employment_length_states.png", bbox_inches='tight')



f = plt.figure(figsize=(20, 12))
data1 = loan.groupby([loan["State"],                    
                     loan["Application Date"].dt.month])['Debt-To-Income Ratio'].mean().unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Debt-To-Income Ratio', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Debt_To_Income_Ratio_states.png", bbox_inches='tight')


f = plt.figure(figsize=(20, 12))
data1 = loan.groupby([loan["State"],                    
                     loan["Application Date"].dt.month])['Risk_Score'].mean().unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Risk Score', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Risk_Score_states.png", bbox_inches='tight')




url = r'C:/Users/Qiulan/.spyder/1/LoanStats_2017Q2.csv'
loan1 = pd.read_csv(url, skiprows = 1, low_memory = False)
loan1["issue_d"] = pd.to_datetime(loan1["issue_d"])

f = plt.figure(figsize=(20, 12))
data1 = loan1.groupby([loan1["addr_state"],                    
                     loan1["issue_d"].dt.month])['annual_inc'].mean().unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'12'}
plt.xlabel('States', **axis_font)
plt.ylabel('Reported Annual Income During Registration', **axis_font)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Annual_income_states.png", bbox_inches='tight')

#----------------------------------------------------------------
#******************************************************************
#loan title

loan['Loan Title'].unique()


loan['Loan Title'][loan['Loan Title']=='debt_consolidation']='Debt consolidation'
loan['Loan Title'][loan['Loan Title']=='home_improvement']='Home improvement'
loan['Loan Title'][loan['Loan Title']=='major_purchase']='Major purchase'

loan['Loan Title'].unique()

data = loan.groupby([loan["Loan Title"],
                      loan["Application Date"].dt.month])["Loan Title"].count()

f = plt.figure(figsize=(20, 12))
data1 = data.unstack()
data1.plot(kind = "bar", stacked= True)
axis_font = {'fontname':'Arial', 'size':'14'}
plt.xlabel('Loan Title', **axis_font)
plt.ylabel('Count of Declined Loan', **axis_font)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(['April','May','June'])
plt.show()
f.savefig("C:/Users/Qiulan/.spyder/1/Declined_lotan_title.png", bbox_inches='tight')






#f = plt.figure(figsize=(20, 10))
#
#ind = np.arange(len(data))
#ind_name = data.index
#
#
#width = 0.35
#d1 = data[4]
#d2 = data[5]
#d3 = data[6]
#p1=plt.bar(ind, d1,  color='green')
#p2=plt.bar(ind, d2,   color='red', bottom=d1)
#p3=plt.bar(ind, d3,   color='yellow', bottom=d2)
#
#p1.set_xticklabels(list(ind_name))
#p2.set_xticklabels(list(ind_name))
#p3.set_xticklabels(list(ind_name))
#
#axis_font = {'fontname':'Arial', 'size':'12'}
#plt.xlabel('States', **axis_font)
#plt.ylabel('Count of Declined Loan', **axis_font)
#plt.xticks(fontsize = 8)
#plt.yticks(fontsize = 10)
#plt.legend(['April','May','June'])
#plt.show()
#f.savefig("C:/Users/Qiulan/.spyder/1/Declined_loan_states.png", bbox_inches='tight')
