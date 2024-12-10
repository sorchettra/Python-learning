# -*- coding: utf-8 -*-

# -- Generate PD result (20240930)  --

# # ODR and Lifetime PD by Products


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# set minimun default date
default_date = '201909'

# set product_id
sme_loan_product_id = [3200,3101,3201,3102,3103,3105,3203,3104]
home_improvement_product_id = [3400,3401]
staff_loan_product_id = [3900,3901]
business_loan_product_id = [3100]
solidarity_credit_product_id = [3001]
agriculture_loan_product_id = [3300,3301]
consumption_loan_product_id = [3500]

# ### Import dataset


# Workspace file directory
workspace_directory = "/data/workspace_files/01_ECL_Dataset/02_Dataset by Products/00_PD_Dataset_by_Segment/"

sme_loan_R10 = pd.read_csv(workspace_directory+"1.SME_2023.csv", index_col=0, low_memory=False)
home_improvement_R10 = pd.read_csv(workspace_directory+"3.HomeImprovement_2023.csv", index_col=0, low_memory=False)
agriculture_loan_R10 = pd.read_csv(workspace_directory+"5.Agri_2023.csv", index_col=0, low_memory=False)
staff_loan_R10 = pd.read_csv(workspace_directory+"7.Staff_2023.csv", index_col=0, low_memory=False)
business_loan_R10 = pd.read_csv(workspace_directory+"2.Business_2023.csv", index_col=0, low_memory=False)
solidarity_credit_R10 = pd.read_csv(workspace_directory+"4.Solidarity_2023.csv", index_col=0, low_memory=False)
consumption_loan_R10 = pd.read_csv(workspace_directory+"6.Consumption_2023.csv", index_col=0, low_memory=False)


sme_loan_R10.shape, business_loan_R10.shape, solidarity_credit_R10.shape, home_improvement_R10.shape, agriculture_loan_R10.shape, consumption_loan_R10.shape, staff_loan_R10.shape

# Workspace file directory
workspace_directory_2 = "/data/workspace_files/02_ECL_Dataset_2024/01_PD_Dataset/01_Dataset_by_products/"

sme_loan_R18 = pd.read_csv(workspace_directory_2+"1.SME_2024.csv", index_col=0, low_memory=False)
home_improvement_R18 = pd.read_csv(workspace_directory_2+"3.HomeImprovement_2024.csv", index_col=0, low_memory=False)
agriculture_loan_R18 = pd.read_csv(workspace_directory_2+"5.Agri_2024.csv", index_col=0, low_memory=False)
staff_loan_R18 = pd.read_csv(workspace_directory_2+"7.Staff_2024.csv", index_col=0, low_memory=False)
business_loan_R18 = pd.read_csv(workspace_directory_2+"2.Business_2024.csv", index_col=0, low_memory=False)
solidarity_credit_R18 = pd.read_csv(workspace_directory_2+"4.Solidarity_2024.csv", index_col=0, low_memory=False)
consumption_loan_R18 = pd.read_csv(workspace_directory_2+"6.Consumption_2024.csv", index_col=0, low_memory=False)

sme_loan_R18.shape, business_loan_R18.shape, solidarity_credit_R18.shape, home_improvement_R18.shape, agriculture_loan_R18.shape, consumption_loan_R18.shape, staff_loan_R18.shape

def filter_and_combine_R10_R18(df_R10, df_R18):
    # filter only necessary column
    col = ['ReportDate', 'Arrangement_ID', 'DaysLate', 'default', 'Max_Term']
    R10 = df_R10.loc[:, ['new_reportdate', 'New_ARRANGEMENT_ID', 'Daylate', 'default', 'max_tenure']]
    R10.columns = col
    R18 = df_R18.loc[:, ['ReportDate', 'Arrangement_ID', 'DaysLate', 'default', 'Max_Term']]
    # filter date
    R10 = R10[R10['ReportDate'] <= 202102]
    print('R10:', R10.shape) 
    R18 = R18[R18['ReportDate'] >= 20210330]
    print('R18:', R18.shape)
    # combine both dataset
    combine_df = pd.concat([R10, R18], axis=0).reset_index(drop=True)
    
    combine_df = combine_df.sort_values(by=['ReportDate'])
    print('Total nbr of rows:', combine_df.shape)
    return combine_df

# combine the two datasets for each ECL Segment
sme_loan = filter_and_combine_R10_R18(sme_loan_R10, sme_loan_R18)
home_improvement = filter_and_combine_R10_R18(home_improvement_R10, home_improvement_R18)
agriculture_loan = filter_and_combine_R10_R18(agriculture_loan_R10, agriculture_loan_R18)    
staff_loan = filter_and_combine_R10_R18(staff_loan_R10, staff_loan_R18)
business_loan = filter_and_combine_R10_R18(business_loan_R10, business_loan_R18)
solidarity_credit = filter_and_combine_R10_R18(solidarity_credit_R10, solidarity_credit_R18)
consumption_loan = filter_and_combine_R10_R18(consumption_loan_R10, consumption_loan_R18)

sme_loan['ReportDate'].min(), sme_loan['ReportDate'].max()

# written off dataset 
wo = pd.read_csv(workspace_directory_2+"writtenoff_data_2024.csv", index_col=0)
wo.shape

# remove duplicate 
wo1 = wo.drop_duplicates(['WrittenoffDate', 'Arrangement_ID'])
wo1.shape

wo.head()

wo1['ReportDate'].unique()

# ## Functions
# ## 1. ODR by Products


def process_wo(wo_df, product_id):
    """ Function to manipulate written off dataset for concatenation """
    wo_df['ReportDate'] = wo['WrittenoffDate'].astype('str').str[:6]
    # remove duplicate
    wo_dup = wo_df[~wo_df.duplicated(["Arrangement_ID", "ProductID", "ReportDate"])]
    wo_dup['default'] = 4

    # filter only those default after 201806
    #wo_dup = wo_dup[wo_dup['reporting_date'] >= default_date]
    # filter by product
    wo_biz_loan = wo_dup[wo_dup['ProductID'].isin(product_id)]

    print("Nbr of written off loans", wo_dup.shape)
    print("Nbr of product wo loans", wo_biz_loan.shape)

    # filter only 3 columns and rename
    col = ['ReportDate', 'Arrangement_ID', 'default']
    wo_biz_loan1 = wo_biz_loan[col]
    wo_biz_loan1 = wo_biz_loan1.rename(columns= {'ReportDate':'month', 'Arrangement_ID': 'unikey'})    
    wo_biz_loan1['month'] = pd.to_numeric(wo_biz_loan1['month'], errors='coerce')
    print("After filter columns:", wo_biz_loan1.shape)

    return wo_biz_loan1

def concat_wo_to_df(wo_product_df, product_loan_df):
    product_loan_df['ReportDate'] = product_loan_df['ReportDate'].astype('str').str[:6]
    product_loan_df['ReportDate'] = pd.to_numeric(product_loan_df['ReportDate'], errors='coerce')
    # Loan Product: filter only 3 columns and rename for concatenation
    col = ['ReportDate', 'Arrangement_ID', 'default']
    business_loan_1 = product_loan_df[col]
    business_loan_1 = business_loan_1.rename(columns= {'ReportDate':'month', 'Arrangement_ID': 'unikey'}) 
    print("Nbr of loans:", business_loan_1.shape)
    
    # filter only wo that are in df_product_loan
    wo_biz_loan1 = wo_product_df[wo_product_df['unikey'].isin(business_loan_1['unikey'])]
    print("Nbr of written off loans:", wo_biz_loan1.shape)

    # concat wo to the df
    biz_loan_update = pd.concat([business_loan_1, wo_biz_loan1], axis=0, ignore_index=True)
    biz_loan_update['default'] = pd.to_numeric(biz_loan_update['default'])
    # replace 4 to 1 for counting as default
    biz_loan_update['default'] =  biz_loan_update['default'].replace(4, 1)

    # remove duplicate
    biz_loan_update = biz_loan_update[~biz_loan_update.duplicated(["unikey", "month"])]
    # sort columns
    business_loan_1 = biz_loan_update.sort_values(by=['unikey', 'month'])
    print("Nbr of consolidated loans:", business_loan_1.shape)

    return business_loan_1

# Function to convert the data to pivot for computation purpose
def convert_to_pivot(business_loan, wo_df, product_id):
    
    # Step 1: written off dataset
    wo_biz_loan1 = process_wo(wo, product_id)

    # Step 2: Create business_loan_1 by selecting specific columns, removing duplicates, and sorting
    business_loan_1 = concat_wo_to_df(wo_biz_loan1, business_loan)

    # Step 3: Pivot the data
    business_loan1 = pd.pivot_table(business_loan_1, columns='month', index='unikey', values='default', aggfunc='mean').reset_index()
    
    return business_loan1, wo_biz_loan1

# Function to compute ODR 
def odr_computation(df):
    df1 = df.copy()

    # ODR Computation
    ODR = pd.DataFrame(columns=["Month", "Performing", "Non-Performing", "ODR"])
    month_columns = df1.columns[1:]

    for i in range(len(month_columns)):
        if i == len(month_columns) - 11:
            break

        current_month = month_columns[i]
        performing_data_1 = df1[~df1[current_month].isna() & (df1[current_month] == 0)]
        npl_1 = performing_data_1.iloc[:, i + 2:i + 14].sum(axis=1, skipna=True)
        npl_2 = (npl_1 >= 1).sum(skipna=True)

        ODR.loc[i] = [current_month, len(performing_data_1), npl_2, npl_2 / len(performing_data_1)]
        
        # convert three columns to int
        cols = ['Month','Performing','Non-Performing'] 
        ODR[cols] = ODR[cols].astype('int')        
        # convert three columns to int
        cols = ['Month','Performing','Non-Performing'] 
        ODR[cols] = ODR[cols].astype('int')

    return ODR

def compute_quarterly_odr(odr_result):
    quarterly_ODR = odr_result.copy()
    quarterly_ODR['Month1'] = quarterly_ODR['Month'].astype('str').str[4:]
    quarterly_ODR = quarterly_ODR[quarterly_ODR['Month1'].isin(['03', '06', '09', '12'])]
    return quarterly_ODR

# ## 2. Lifetime PD Computation by Product


def process_wo_for_lifetime_pd(wo_product_loan):
    # filter from wo_df and manipulate for MIA (replace 4 as MIA3)
    col = ['month', 'unikey', 'default']
    wo_product_loan_lifetime = wo_product_loan[col]
    wo_loan_lifetime = wo_product_loan_lifetime.rename(columns= {'default': 'final_mia'}) 
    wo_loan_lifetime['month'] = pd.to_numeric(wo_loan_lifetime['month'], errors='coerce')
    wo_loan_lifetime['final_mia'] =  wo_loan_lifetime['final_mia'].replace(4, 'MIA3')
    print("Nbr of written off loans:", wo_loan_lifetime.shape)
    
    return wo_loan_lifetime

def create_wo_row(df):
    #unique values of months
    month_list = df[['month','ReportMonth','Next12Months','report_movement']].drop_duplicates()

    #list of wo loan and its max report date
    wo_aa = df[df['Is_WrittenOff']==1].groupby(['unikey'])['month'].max().reset_index()

    #creating new rows for wo loan from the date of write off till max reporting date of dataset
    wo_newrow_list=[]
    for index, loan in wo_aa.iterrows():
        for i, m in month_list.iterrows():
            if m['month'] > loan['month']:
                wo_newrow_list.append([ m['month'], loan['unikey'], 'MIA3'
                                       ,m['ReportMonth'], m['Next12Months'], 'MIA3'
                                       ,1, m['report_movement'], 'MIA3MIA3'])
    wo_newrow_df = pd.DataFrame(wo_newrow_list,columns=df.columns)
    df = df.append(wo_newrow_df, ignore_index=True)
    return df

def processing_pd_computation(df, wo_product_loan):

    # Filter only the neccessary columns
    # rename the column names
    # drop duplicate
    # max tenure = maximun term per loan product/loan account (to confirm with Risk)
    product_loan_lifetime = df.loc[:, ['ReportDate', 'Arrangement_ID', 'DaysLate', 'default', 'Max_Term']]
    product_loan_lifetime = product_loan_lifetime.rename(columns={'ReportDate': 'ReportMonth', 'Arrangement_ID': 'LoanID', 'DaysLate': 'days_late', 'default': 'default_flag', 'Max_Term': 'tenure'})
    product_loan_lifetime = product_loan_lifetime.drop_duplicates(subset=['ReportMonth', 'LoanID'])

    # Create MIA Migration based on Short-term and Long-term Loan --> Final MIA
    # define conditions based on Loan Stages
    conditions = [
        (product_loan_lifetime['default_flag'] == 1) & (product_loan_lifetime['tenure'] > 12),
        (product_loan_lifetime['days_late'] == 0) & (product_loan_lifetime['tenure'] > 12),
        (product_loan_lifetime['days_late'] >= 1) & (product_loan_lifetime['days_late'] < 30) & (product_loan_lifetime['tenure'] > 12),
        (product_loan_lifetime['days_late'] >= 30) & (product_loan_lifetime['days_late'] < 90) & (product_loan_lifetime['tenure'] > 12),
        (product_loan_lifetime['default_flag'] == 1) & (product_loan_lifetime['tenure'] <= 12),
        (product_loan_lifetime['days_late'] == 0) & (product_loan_lifetime['tenure'] <= 12),
        (product_loan_lifetime['days_late'] >= 1) & (product_loan_lifetime['days_late'] < 15) & (product_loan_lifetime['tenure'] <= 12),
        (product_loan_lifetime['days_late'] >= 15) & (product_loan_lifetime['days_late'] < 31) & (product_loan_lifetime['tenure'] <= 12),
    ]

    #define results
    results = ['MIA3', 'MIA0', 'MIA1', 'MIA2', 'MIA3', 'MIA0', 'MIA1', 'MIA2']

    #create new column based on conditions
    product_loan_lifetime['final_mia'] = np.select(conditions, results)
    print("Nbr of loans:", product_loan_lifetime.shape)

    """ Add on 03/01/2024 for written off """
    # filter from loan_df after adding MIA and rename columns
    col = ['ReportMonth', 'LoanID', 'final_mia']
    prod_loan_lifetime_dedup = product_loan_lifetime[col]
    prod_loan_lifetime_dedup = prod_loan_lifetime_dedup.rename(columns= {'ReportMonth':'month', 'LoanID': 'unikey'}) 
    print("Nbr of loans after removing duplicate:", prod_loan_lifetime_dedup.shape)
    
    # written off df
    wo_product_loan_lifetime = process_wo_for_lifetime_pd(wo_product_loan)
    # filter only written off in the product
    wo_product_loan_lifetime = wo_product_loan_lifetime[wo_product_loan_lifetime['unikey'].isin(prod_loan_lifetime_dedup['unikey'])]
    print("Nbr of loans wo:", wo_product_loan_lifetime.shape) # adding 
    # concate wo to product loan lifetime 
    #prod_loan_lifetime_final = pd.concat([prod_loan_lifetime_dedup, wo_product_loan_lifetime], axis=0, ignore_index=True)    
    #print("Nbr of loans:", product_loan_lifetime.shape)


    #product_loan_lifetime = prod_loan_lifetime_final.copy()
    product_loan_lifetime = prod_loan_lifetime_dedup.copy()
    # Convert 'ReportMonth' to a datetime object & Add 12 months to the "ReportMonth" column
    product_loan_lifetime['ReportMonth'] = pd.to_datetime(product_loan_lifetime['month'], format='%Y%m', errors='coerce')
    product_loan_lifetime['Next12Months'] = product_loan_lifetime['ReportMonth'] + pd.DateOffset(months=12)

    # Format the "Next12Months" & "ReportMonth"  column in the "yyyy-m" format
    product_loan_lifetime['Next12Months'] = product_loan_lifetime['Next12Months'].dt.strftime('%Y-%m')
    product_loan_lifetime['ReportMonth']= product_loan_lifetime['ReportMonth'].dt.strftime('%Y-%m')


    # Find the next 12 months MIA movement
    # Step1: sort the "Report_Month" column and find the loan status in the next 12 months
    product_loan_lifetime['mia_lead_12m'] = product_loan_lifetime.sort_values(by=['ReportMonth'], ascending=True).groupby(['unikey'])['final_mia'].shift(-12)
    product_loan_lifetime['Is_WrittenOff'] = np.where(product_loan_lifetime['unikey'].isin(wo_product_loan_lifetime['unikey']), 1, 0)
    # Step2: 
    # 1. fill missing value of the next 12 months as 'FS' - Fully settle
    # 2. fill missing value of current ReportMonth as 'NA' to count
    #product_loan_lifetime['mia_lead_12m'] = product_loan_lifetime['mia_lead_12m'].fillna('FS')
    product_loan_lifetime['mia_lead_12m'] = np.where(
            (product_loan_lifetime['mia_lead_12m'].isnull()) & (product_loan_lifetime['Is_WrittenOff'] == 1), 'MIA3',
                np.where((product_loan_lifetime['mia_lead_12m'].isnull()) & (product_loan_lifetime['Is_WrittenOff'] == 0), 'FS',
                    product_loan_lifetime['mia_lead_12m'])
            )
  
    product_loan_lifetime['final_mia'] = product_loan_lifetime['final_mia'].fillna('NA')

    # Step3: create "report_movement" period and "mia_movement" consolidation
    product_loan_lifetime['report_movement'] = product_loan_lifetime['ReportMonth'] + '_' + product_loan_lifetime['Next12Months']
    product_loan_lifetime['mia_movement'] = product_loan_lifetime['final_mia'] + product_loan_lifetime['mia_lead_12m']
    
    product_loan_lifetime = create_wo_row(product_loan_lifetime)
    print("Nbr of loans wo1:", product_loan_lifetime.shape)

    # view the dataframe
    return product_loan_lifetime

# ### MIA Movement


def generate_mia_pivot_table(data):
    """ Number of Accounts """
    mia_pivot_table = pd.pivot_table(data, columns='mia_lead_12m', index=['report_movement', 'final_mia'],
                                     values='unikey', aggfunc=pd.Series.nunique, fill_value=0, margins=True)
    new_index = pd.MultiIndex.from_product(
            [data['report_movement'].unique(), ['MIA0', 'MIA1', 'MIA2', 'MIA3']] 
            ,names=['report_movement', 'final_mia'])
    mia_pivot_table= mia_pivot_table.reindex(new_index,fill_value=0)
    #mia_pivot_table = mia_pivot_table.sort_values(by=['report_movement']) # add on 28/10/2024
    return mia_pivot_table

def calculate_mia_percentage(mia_pivot_table):
    """ Percentage of Number of Accounts """
    mia_movement_percentage = mia_pivot_table.div(mia_pivot_table.iloc[:,-1], axis=0) * 100
    return mia_movement_percentage

# ### Transition Matrices


# Function to calculate Transition Matrices
def calculate_average_for_nth_row(dataframe, n):
    nth_row = dataframe.groupby(['report_movement']).nth(n)
    nth_row = nth_row.drop(['final_mia'], axis=1)
    average = nth_row.mean()
    return average

def consolidate_averages(dataframe, n_values):
    consolidated_dataframe = pd.DataFrame()
    
    for n in n_values:
        result = calculate_average_for_nth_row(dataframe, n)
        consolidated_dataframe[f'MIA{n}'] = result
    
    return consolidated_dataframe

# call functions
def compute_first_matric(df_mia_movement_per):
    # DataFrame: 'mia_movement_percentage1'
    n_values = [0, 1, 2, 3]  # List of values for n

    # reset index of mia movement table
    mia_movement_percentage1 = df_mia_movement_per.reset_index()
    mia_movement_percentage1.drop(index=mia_movement_percentage1.index[-1], axis=0, inplace=True)

    # calculate average
    consolidated_result = consolidate_averages(mia_movement_percentage1, n_values)

    # Transpose the consolidated DataFrame
    transposed_result = consolidated_result.T

    # Rename the first column name to "12 Months"
    transition_matrics_12m = transposed_result.reset_index()
    transition_matrics_12m.columns = ['12 Months'] + list(transition_matrics_12m.columns[1:])
    # drop "FS" column (optional if needed)
    transition_matrics_12m = transition_matrics_12m.drop(columns='FS', axis=0)

    # Print result
    return transition_matrics_12m

def replace_mia3(df_matrics_to_replace):
    """
    For transition matrix, MIA3 (loan is already in stage 3, we'll treat it as a loan that already default. 
    That's why in the matrix, we put in 100% without calculation. The same for FS)
    """
    Average_MIA3 = pd.DataFrame({
        'MIA3': [0, 0, 0, 100, 100]
    })
    Average_MIA3 = Average_MIA3.T
    Average_MIA3.columns = ['MIA0', 'MIA1', 'MIA2', 'MIA3', 'All']

    # drop the average row and include the 0 row for MIA3
    i = df_matrics_to_replace.iloc[3:4,:].index
    df_matrics_to_replace = df_matrics_to_replace.drop(i)

    # add MIA3 as 0 like Deliotte
    concated_trn_matrics_12m = pd.concat([df_matrics_to_replace, Average_MIA3], axis=0)
    concated_trn_matrics_12m = concated_trn_matrics_12m.fillna('MIA3')
    concated_trn_matrics_12m = concated_trn_matrics_12m.drop('All', axis=1)
    
    return concated_trn_matrics_12m

# Create function to compute Transition Matrics for each transition months
def transition_matrics(df1, df2):
    transition_matrics_a = df1.iloc[:,0:4].to_numpy()
    transition_matrics_b = df2.iloc[:,0:4].to_numpy()
    transition_matrics_nth_m = np.dot(transition_matrics_a, transition_matrics_b)/ 100
    transition_matrics_nth_m = transition_matrics_nth_m
    
    transition_matrics_nth_m = pd.DataFrame(transition_matrics_nth_m)
    transition_matrics_nth_m.columns = list(transition_matrics_12m.columns[1:5])
    transition_matrics_nth_m.index = list(transition_matrics_12m.columns[1:5])
    
    return transition_matrics_nth_m

def consolidate_transition_matrics(df_12m, df_24m, df_36m, df_48m, df_60m, df_72m, df_84m, df_96m, df_108m, df_120m):

    # Define a list of values to add as new columns
    list_of_months = ['12 Months', '24 Months', '36 Months', '48 Months', '60 Months', '72 Months', '84 Months', '96 Months', '108 Months', '120 Months']
    list_of_years = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10']
    list_of_order = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]

    # Create a list of DataFrames 
    dataframes =  [transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m
                ]

    # Create a new list to store the modified DataFrames
    modified_dataframes = []

    # Loop through the list of DataFrames and add a new columns with one value from the list
    for i, df in enumerate(dataframes):
        # Create a new DataFrame
        df_copy = df.copy()
        
        # Create a new columns with a single value from the list
        df_copy["transition_months"] = list_of_months[i]
        df_copy["transition_years"] = list_of_years[i]
        df_copy["order_nbr"] = list_of_order[i]
        
        # Append the modified DataFrame to the new list
        modified_dataframes.append(df_copy)

    # Concatenate the separate DataFrames into a single result DataFrame
    result_df = pd.concat(modified_dataframes)
    result_df.index.name = 'MIA_Stage'

    result_df = result_df.reset_index()

    return result_df

# ### Cumulative PD and Marginal PD


def compute_cumulative_PD(result_df):
    # Pivot to compute Cumulative PD 
    cumulative_pd = pd.pivot_table(result_df, index='MIA_Stage', columns=['order_nbr','transition_years'], values='MIA3').sort_values(by='order_nbr', axis=1, ascending=True)
    # remove 'order_nbr' as it's just for sorting
    cumulative_pd.columns = cumulative_pd.columns.droplevel(level=0)
    return cumulative_pd

def compute_marginal_PD(cumulative_pd):
    # Compute Marginal PD
    marginal_pd = round(cumulative_pd.diff(axis=1), 2)
    marginal_pd['Y1'] = marginal_pd['Y1'].fillna(cumulative_pd['Y1'])
    return marginal_pd

# ## 3. Result
# ### 1. SME Loan


sme_loan.head()

sme_loan1, wo_loan1 = convert_to_pivot(sme_loan, wo1, sme_loan_product_id)
sme_loan_odr = odr_computation(sme_loan1)
sme_loan_odr

# Quarterly ODR result
sme_loan_quarterly_ODR = compute_quarterly_odr(sme_loan_odr)
sme_loan_quarterly_ODR


sme_loan_lifetime = processing_pd_computation(sme_loan, wo_loan1)

sme_loan_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
sme_loan_lifetime1 = sme_loan_lifetime[~sme_loan_lifetime['ReportMonth'].isin(report_month)]
sme_loan_lifetime1['ReportMonth'].max()

sme_loan_lifetime1 = sme_loan_lifetime1[sme_loan_lifetime1['month'] >= 201909]
sme_loan_lifetime1 

# Number of Accounts
sme_loan_mia_movement = generate_mia_pivot_table(sme_loan_lifetime1)
print(sme_loan_mia_movement.sort_values(['report_movement', 'final_mia']))

# Percentage of Number of Accounts
sme_loan_mia_movement_percentage = calculate_mia_percentage(sme_loan_mia_movement)
print(sme_loan_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(sme_loan_mia_movement_percentage)
transition_matrics_12m

transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
transition_matrics_12m1


# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

transition_matrics_12m1

sme_transition_matrics_result = transition_matrics_result.copy()
sme_transition_matrics_result = sme_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
sme_loan_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
sme_loan_cumulative_pd

# Marginal PD
sme_loan_marginal_pd = compute_marginal_PD(sme_loan_cumulative_pd)
sme_loan_marginal_pd

# ### 2. Business Loan


business_loan.head()

business_loan.shape

business_loan1, biz_wo_loan = convert_to_pivot(business_loan, wo1, business_loan_product_id)
business_loan_odr = odr_computation(business_loan1)
business_loan_odr

# Quarterly ODR result
business_loan_quarterly_ODR = compute_quarterly_odr(business_loan_odr)
business_loan_quarterly_ODR

business_loan_lifetime = processing_pd_computation(business_loan, biz_wo_loan)
business_loan_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
business_loan_lifetime1 = business_loan_lifetime[~business_loan_lifetime['ReportMonth'].isin(report_month)]
business_loan_lifetime1['ReportMonth'].max()

business_loan_lifetime1 = business_loan_lifetime1[business_loan_lifetime1['month'] >= 201909]
business_loan_lifetime1 

# Number of Accounts
business_mia_movement = generate_mia_pivot_table(business_loan_lifetime1)
#print(business_mia_movement)
print(business_mia_movement.sort_values(['report_movement', 'final_mia']))

# Percentage of Number of Accounts
business_mia_movement_percentage = calculate_mia_percentage(business_mia_movement)
print(business_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(business_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

biz_transition_matrics_result = transition_matrics_result.copy()
biz_transition_matrics_result = biz_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
business_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
business_cumulative_pd

# Marginal PD
business_marginal_pd = compute_marginal_PD(business_cumulative_pd)
business_marginal_pd

# ### 3. Home Improvement Loan


home_improvement.head()

home_improvement1, home_wo_loan = convert_to_pivot(home_improvement, wo1, home_improvement_product_id)
home_improvement_odr = odr_computation(home_improvement1)
home_improvement_odr

# Quarterly ODR result
home_improvement_quarterly_ODR = compute_quarterly_odr(home_improvement_odr)
home_improvement_quarterly_ODR

home_improvement_lifetime = processing_pd_computation(home_improvement, home_wo_loan)
home_improvement_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
home_improvement_lifetime1 = home_improvement_lifetime[~home_improvement_lifetime['ReportMonth'].isin(report_month)]
home_improvement_lifetime1['ReportMonth'].max()

home_improvement_lifetime1 = home_improvement_lifetime1[home_improvement_lifetime1['month'] >= 201909]
home_improvement_lifetime1 

# Number of Accounts
home_improvement_mia_movement = generate_mia_pivot_table(home_improvement_lifetime1)
print(home_improvement_mia_movement)

# Percentage of Number of Accounts
home_improvement_mia_movement_percentage = calculate_mia_percentage(home_improvement_mia_movement)
print(home_improvement_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(home_improvement_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

home_transition_matrics_result = transition_matrics_result.copy()
home_transition_matrics_result = home_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
home_improvement_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
home_improvement_cumulative_pd

# Marginal PD
home_improvement_marginal_pd = compute_marginal_PD(home_improvement_cumulative_pd)
home_improvement_marginal_pd

# to save memory 
del sme_loan_R10 
del business_loan_R10
del solidarity_credit_R10 
del home_improvement_R10 
del agriculture_loan_R10 
del consumption_loan_R10
del staff_loan_R10

# to save memory 
del sme_loan_R18 
del business_loan_R18
del solidarity_credit_R18 
del home_improvement_R18 
del agriculture_loan_R18 
del consumption_loan_R18
del staff_loan_R18

# ### 4. Solidarity Credit


solidarity_credit.head()

solidarity_credit1, sc_wo_loan = convert_to_pivot(solidarity_credit, wo, solidarity_credit_product_id)
solidarity_credit_odr = odr_computation(solidarity_credit1)
solidarity_credit_odr

solidarity_credit1, sc_wo_loan = convert_to_pivot(solidarity_credit, wo1, solidarity_credit_product_id)
solidarity_credit_odr = odr_computation(solidarity_credit1)
solidarity_credit_odr

# Quarterly ODR result
solidarity_credit_quarterly_ODR = compute_quarterly_odr(solidarity_credit_odr)
solidarity_credit_quarterly_ODR

solidarity_credit_lifetime = processing_pd_computation(solidarity_credit, sc_wo_loan)
solidarity_credit_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
solidarity_credit_lifetime1 = solidarity_credit_lifetime[~solidarity_credit_lifetime['ReportMonth'].isin(report_month)]
solidarity_credit_lifetime1['ReportMonth'].max()

solidarity_credit_lifetime1 = solidarity_credit_lifetime1[solidarity_credit_lifetime1['month'] >= 201909]
solidarity_credit_lifetime1 

# Number of Accounts
solidarity_credit_mia_movement = generate_mia_pivot_table(solidarity_credit_lifetime1)
print(solidarity_credit_mia_movement)

# Percentage of Number of Accounts
solidarity_credit_mia_movement_percentage = calculate_mia_percentage(solidarity_credit_mia_movement)
print(solidarity_credit_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(solidarity_credit_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result = transition_matrics_result[transition_matrics_result['transition_years'].isin(['Y1', 'Y2', 'Y3'])]
transition_matrics_result

sc_transition_matrics_result = transition_matrics_result.copy()
sc_transition_matrics_result = sc_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
solidarity_credit_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
solidarity_credit_cumulative_pd

# Marginal PD
solidarity_credit_marginal_pd = compute_marginal_PD(solidarity_credit_cumulative_pd)
solidarity_credit_marginal_pd

# ### 5. Agriculture Loan


agriculture_loan.head()

#agriculture_loan = agriculture_loan[agriculture_loan['ReportDate'] >= 20210331]
agriculture_loan.shape

agriculture_loan1, agri_wo_loan = convert_to_pivot(agriculture_loan, wo1, agriculture_loan_product_id)
agriculture_loan_odr = odr_computation(agriculture_loan1)
agriculture_loan_odr

# Quarterly ODR result
agriculture_loan_quarterly_ODR = compute_quarterly_odr(agriculture_loan_odr)
agriculture_loan_quarterly_ODR

agriculture_loan_lifetime = processing_pd_computation(agriculture_loan, agri_wo_loan)
agriculture_loan_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
agriculture_loan_lifetime1 = agriculture_loan_lifetime[~agriculture_loan_lifetime['ReportMonth'].isin(report_month)]
agriculture_loan_lifetime1['ReportMonth'].max()

agriculture_loan_lifetime1 = agriculture_loan_lifetime1[agriculture_loan_lifetime1['month'] >= 201909]
agriculture_loan_lifetime1 


# Number of Accounts
agriculture_loan_mia_movement = generate_mia_pivot_table(agriculture_loan_lifetime1)
print(agriculture_loan_mia_movement)

# Percentage of Number of Accounts
agriculture_loan_mia_movement_percentage = calculate_mia_percentage(agriculture_loan_mia_movement)
print(agriculture_loan_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(agriculture_loan_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

agri_transition_matrics_result = transition_matrics_result.copy()
agri_transition_matrics_result = agri_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
agriculture_loan_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
agriculture_loan_cumulative_pd

# Marginal PD
agriculture_loan_marginal_pd = compute_marginal_PD(agriculture_loan_cumulative_pd)
agriculture_loan_marginal_pd

# ### 6. Consumption Loan


consumption_loan.head()

#consumption_loan = consumption_loan[consumption_loan['ReportDate'] >= 20210331]
consumption_loan.shape

# Compute ODR
consumption_loan1, consump_wo_loan = convert_to_pivot(consumption_loan, wo1, consumption_loan_product_id)
consumption_loan_odr = odr_computation(consumption_loan1)
consumption_loan_odr

# Quarterly ODR result
consumption_loan_quarterly_ODR = compute_quarterly_odr(consumption_loan_odr)
consumption_loan_quarterly_ODR

consumption_loan_lifetime = processing_pd_computation(consumption_loan, consump_wo_loan)
consumption_loan_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
consumption_loan_lifetime1 = consumption_loan_lifetime[~consumption_loan_lifetime['ReportMonth'].isin(report_month)]
consumption_loan_lifetime1['ReportMonth'].max()

consumption_loan_lifetime1 = consumption_loan_lifetime1[consumption_loan_lifetime1['month'] >= 201909]
consumption_loan_lifetime1 

# Number of Accounts
consumption_loan_mia_movement = generate_mia_pivot_table(consumption_loan_lifetime1)
print(consumption_loan_mia_movement)

# Percentage of Number of Accounts
consumption_loan_mia_movement_percentage = calculate_mia_percentage(consumption_loan_mia_movement)
print(consumption_loan_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(consumption_loan_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

comp_transition_matrics_result = transition_matrics_result.copy()
comp_transition_matrics_result = comp_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
consumption_loan_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
consumption_loan_cumulative_pd

# Marginal PD
consumption_loan_marginal_pd = compute_marginal_PD(consumption_loan_cumulative_pd)
consumption_loan_marginal_pd

# ### 7. Staff Loan


staff_loan.head()

# Compute ODR
staff_loan1, staff_wo_loan = convert_to_pivot(staff_loan, wo1, staff_loan_product_id)
staff_loan_odr = odr_computation(staff_loan1)
staff_loan_odr

# Quarterly ODR result
staff_loan_quarterly_ODR = compute_quarterly_odr(staff_loan_odr)
staff_loan_quarterly_ODR

staff_loan_lifetime = processing_pd_computation(staff_loan, staff_wo_loan)
staff_loan_lifetime.head()

# remove the ReportMonth outside of the study timeframe 
#report_month = ['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
report_month = ['2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09']
staff_loan_lifetime1 = staff_loan_lifetime[~staff_loan_lifetime['ReportMonth'].isin(report_month)]
staff_loan_lifetime1['ReportMonth'].max()

staff_loan_lifetime1 = staff_loan_lifetime1[staff_loan_lifetime1['month'] >= 201909]
staff_loan_lifetime1 

# Number of Accounts
staff_loan_mia_movement = generate_mia_pivot_table(staff_loan_lifetime1)
print(staff_loan_mia_movement)

# Percentage of Number of Accounts
staff_loan_mia_movement_percentage = calculate_mia_percentage(staff_loan_mia_movement)
print(staff_loan_mia_movement_percentage)

# Transition Matrices
transition_matrics_12m = compute_first_matric(staff_loan_mia_movement_percentage)
transition_matrics_12m1 = replace_mia3(transition_matrics_12m)
# apply Transition Matrics function 
transition_matrics_12m1 = transition_matrics_12m1.set_index("12 Months")
transition_matrics_24m = transition_matrics(transition_matrics_12m1, transition_matrics_12m1)
transition_matrics_36m = transition_matrics(transition_matrics_12m1, transition_matrics_24m)
transition_matrics_48m = transition_matrics(transition_matrics_12m1, transition_matrics_36m)
transition_matrics_60m = transition_matrics(transition_matrics_12m1, transition_matrics_48m)
transition_matrics_72m = transition_matrics(transition_matrics_12m1, transition_matrics_60m)
transition_matrics_84m = transition_matrics(transition_matrics_12m1, transition_matrics_72m)
transition_matrics_96m = transition_matrics(transition_matrics_12m1, transition_matrics_84m)
transition_matrics_108m = transition_matrics(transition_matrics_12m1, transition_matrics_96m)
transition_matrics_120m = transition_matrics(transition_matrics_12m1, transition_matrics_108m)

# Transition Matrices consolidated result
transition_matrics_result = consolidate_transition_matrics(transition_matrics_12m1, transition_matrics_24m, transition_matrics_36m
                            , transition_matrics_48m, transition_matrics_60m, transition_matrics_72m
                            , transition_matrics_84m, transition_matrics_96m, transition_matrics_108m, transition_matrics_120m)
transition_matrics_result

staff_transition_matrics_result = transition_matrics_result.copy()
staff_transition_matrics_result = staff_transition_matrics_result.drop(['transition_years', 'order_nbr'], axis=1)

# Cumulative PD
staff_loan_cumulative_pd = compute_cumulative_PD(transition_matrics_result)
staff_loan_cumulative_pd

# Marginal PD
staff_loan_marginal_pd = compute_marginal_PD(staff_loan_cumulative_pd)
staff_loan_marginal_pd



# ## 4. Export Result


import pandas as pd
# ODR
report_date = str(20241030)
export_directory = "/data/workspace_files/02_ECL_Dataset_2024/01_PD_Dataset/01_Dataset_by_products/01_ODR Result/"

with pd.ExcelWriter(export_directory + 'Monthly and Quarterly ODR Result_v0.2 ' + report_date + '.xlsx',
                    engine='xlsxwriter') as writer:
    sme_loan_odr.to_excel(writer, sheet_name='SME_loan_monthly_ODR')
    sme_loan_quarterly_ODR.to_excel(writer, sheet_name='SME_loan_quarterly_ODR')
    business_loan_odr.to_excel(writer, sheet_name='business_loan_monthly_ODR')
    business_loan_quarterly_ODR.to_excel(writer, sheet_name='business_loan_quarterly_ODR')
    home_improvement_odr.to_excel(writer, sheet_name='home_improvement_monthly_ODR')
    home_improvement_quarterly_ODR.to_excel(writer, sheet_name='home_improvement_quarterly_ODR')
    solidarity_credit_odr.to_excel(writer, sheet_name='solidarity_credit_monthly_ODR')
    solidarity_credit_quarterly_ODR.to_excel(writer, sheet_name='solidarity_credit_quarterly_ODR')
    agriculture_loan_odr.to_excel(writer, sheet_name='agriculture_loan_monthly_ODR')
    agriculture_loan_quarterly_ODR.to_excel(writer, sheet_name='agriculture_loan_quarterly_ODR')
    consumption_loan_odr.to_excel(writer, sheet_name='consumption_loan_monthly_ODR')
    consumption_loan_quarterly_ODR.to_excel(writer, sheet_name='consumption_loan_quarterly_ODR')
    staff_loan_odr.to_excel(writer, sheet_name='staff_loan_monthly_ODR')
    staff_loan_quarterly_ODR.to_excel(writer, sheet_name='staff_loan_quarterly_ODR')

# Lifeitme PD
report_date = str(20241030)
export_directory = "/data/workspace_files/02_ECL_Dataset_2024/01_PD_Dataset/01_Dataset_by_products/02_Lifetime PD/"

with pd.ExcelWriter(export_directory+'Amret IFRS 9 SME Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    sme_loan_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    sme_loan_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    sme_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')    
    sme_loan_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    sme_loan_marginal_pd.to_excel(writer, sheet_name='Marginal PD')


with pd.ExcelWriter(export_directory+'Amret IFRS 9 Business Loan Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    business_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    business_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    biz_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    business_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    business_marginal_pd.to_excel(writer, sheet_name='Marginal PD')

with pd.ExcelWriter(export_directory+'Amret IFRS 9 Home Improvement Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    home_improvement_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    home_improvement_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    home_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    home_improvement_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    home_improvement_marginal_pd.to_excel(writer, sheet_name='Marginal PD')


with pd.ExcelWriter(export_directory+'Amret IFRS 9 Solidarity Credit Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    solidarity_credit_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    solidarity_credit_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    sc_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    solidarity_credit_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    solidarity_credit_marginal_pd.to_excel(writer, sheet_name='Marginal PD')


with pd.ExcelWriter(export_directory+'Amret IFRS 9 Agriculture Loan Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    agriculture_loan_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    agriculture_loan_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    agri_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    agriculture_loan_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    agriculture_loan_marginal_pd.to_excel(writer, sheet_name='Marginal PD')

with pd.ExcelWriter(export_directory+'Amret IFRS 9 Consumption Loan Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    consumption_loan_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    consumption_loan_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    comp_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    consumption_loan_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    consumption_loan_marginal_pd.to_excel(writer, sheet_name='Marginal PD')

with pd.ExcelWriter(export_directory+'Amret IFRS 9 Staff Loan Lifetime PD '+ report_date +'.xlsx', engine='xlsxwriter') as writer:
    staff_loan_mia_movement.to_excel(writer, sheet_name='MIA Movement')
    staff_loan_mia_movement_percentage.to_excel(writer, sheet_name='MIA Movement Percentage')
    staff_transition_matrics_result.to_excel(writer, sheet_name='Transition Matrices')
    staff_loan_cumulative_pd.to_excel(writer, sheet_name='Cumulative PD')
    staff_loan_marginal_pd.to_excel(writer, sheet_name='Marginal PD')

# Transpose
report_date = str(20241030)
import os

os.chdir("/data/workspace_files/02_ECL_Dataset_2024/01_PD_Dataset/01_Dataset_by_products/02_Lifetime PD/Transpose/")


sme_loan_lifetime1.to_csv('SME_loan Lifetime PD transpose.csv')
business_loan_lifetime1.to_csv('business_loan Lifetime PD transpose.csv')
home_improvement_lifetime1.to_csv('home_improvement Lifetime PD transpose.csv')
solidarity_credit_lifetime1.to_csv('Solidarity credit Lifetime PD transpose.csv')
agriculture_loan_lifetime1.to_csv('agriculture_loan Lifetime PD transpose.csv')
consumption_loan_lifetime1.to_csv('consumption_loan Lifetime PD transpose.csv')
staff_loan_lifetime1.to_csv('staff_loan Lifetime PD transpose.csv')



