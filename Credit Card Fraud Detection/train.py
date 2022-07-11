# %% [markdown]
# # Credit Card Fraud Prediction - Model Training
# 
# This demo is based on the Machine Learning for Credit Card Fraud detection - Practical handbook, https://fraud-detection-handbook.github.io/fraud-detection-handbook/
# 
# Before running this notebook you need to go through ***01 - Snowpark - Feature Engineering***
# 
# Import the libraries we are using

# %%
# Snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.types as T
import snowflake.snowpark.functions as F
from snowflake.snowpark.types import PandasSeries, PandasDataFrame

# Print the version of Snowpark we are using
from importlib.metadata import version
version('snowflake_snowpark_python')

# %%
import json
import datetime
import sklearn
from sklearn import *
import pandas as pd
import numpy as np
import cachetools

# %%
pd.options.display.max_columns = 500
pd.options.display.max_rows = 100

# %% [markdown]
# Start by connecting to Snowflake and get our data, make sure that ***creds.json*** is updated with your username and password

# %%
with open('creds.json') as f:
    connection_parameters = json.load(f)    

# %%
session = Session.builder.configs(connection_parameters).create()
print(f"Current schema: {session.get_fully_qualified_current_schema()}, current role: {session.get_current_role()}, current warehouse:  {session.get_current_warehouse()}")

# %% [markdown]
# Start by making sure we are using a MEDIUM WH since we will pull back data to the client.

# %%
def scale_wh(sess, wh, size):
    if (len(wh) == 0): 
        return false
    if (len(size) == 0):
        return false
   
    alter_SQL = "ALTER WAREHOUSE " + wh + " SET WAREHOUSE_SIZE = " + size
    sess.sql(alter_SQL).collect()
    return True

# %%
scale_wh(session, session.get_current_warehouse(), "MEDIUM")

# %% [markdown]
# We will not use the full data set for traing so we need to add a filter. Since we want to be sure that the data we used for training is not changed we will use the zero-copy-cloning feature of Snowflake to create a datasnapshot.

# %%
# We will not use the full data set for training/test
start_date_training = datetime.datetime.strptime("2019-05-25", "%Y-%m-%d")
delta_train = delta_delay = delta_test = 7 # Number of days in train, delay and test data sets
end_date_data = start_date_training+datetime.timedelta(days=delta_train+delta_delay+delta_test+1)

# Using Zero-Copy Cloning feature to keep a copy of the data being used for each training

now = datetime.datetime.now()
s = now.strftime("%Y_%m_%d__%H_%M_%S")
print (s);

origin_table_name = "CUSTOMER_TRX_FRAUD_FEATURES"
cloned_table_name = "CUSTOMER_TRX_FRAUD_FEATURES_" + s

sql = "create table " + cloned_table_name + " clone " + origin_table_name

session.sql(sql).collect()
print ("Using Table: ", cloned_table_name)
df_trx = session.table(cloned_table_name).filter((F.col("TX_DATETIME") <= F.to_date(F.lit(end_date_data))) & (F.col("TX_DATETIME") >= F.to_date(F.lit(start_date_training)))).sort(F.col("TX_DATETIME"))
df_trx.show()

# %%
df_trx.count()

# %% [markdown]
# We can get some statistics on our numeric columns by using the **describe**. We are using **to_pandas** to get back the result (5 rows) as a pndas dataframe to use it's printing capabilities

# %%
df_trx.describe().to_pandas()

# %% [markdown]
# We could also check if any of our columns are highly correlated with each other.
# Snowpark does have a correlation function, but not a correlation matrix function, but since we are using Python we can write a function that generates a correlation matrix without puliing back data from Snowflake

# %%
def corr_matrix(df):
    # Pick out only those columns that has numeric data type
    numeric_types = [T.DecimalType, T.LongType, T.DoubleType, T.FloatType, T.IntegerType]
    cols = [c.name for c in df.schema.fields if type(c.datatype) in numeric_types]
    cols2 = cols.copy()
    res_df = None
    for col in cols:
        corr_df = (df.agg([F.corr(col, y) for y in cols2])
                   .toDF(cols2)
                   .select(F.lit(col).as_("COLUMN"), *cols2))
        res_df = res_df.union(corr_df) if res_df else corr_df

    return res_df

# %%
corr_matrix(df_trx.select(['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUST_CNT_TX_1',
       'CUST_AVG_AMOUNT_1', 'CUST_CNT_TX_7', 'CUST_AVG_AMOUNT_7', 'CUST_CNT_TX_30',
       'CUST_AVG_AMOUNT_30', 'NB_TX_WINDOW_1', 'TERM_RISK_1', 'NB_TX_WINDOW_7','TERM_RISK_7',
        'NB_TX_WINDOW_30','TERM_RISK_30', 'TX_FRAUD'])).to_pandas()

# %% [markdown]
# ## Split data in training and testing data sets
# 
# The training set aims at training a prediction model, while the test set aims at evaluating the performance of the prediction model on new data. In a fraud detection context, the transactions of the test set occur chronologically after the transactions used for training the model.
# 
# It is worth noting that we choose our test set to take place one week after the last transaction of the training set. In a fraud detection context, this period separating the training and test set is referred to as the delay period or feedback delay. It accounts for the fact that, in a real-world fraud detection system, the label of a transaction (fraudulent or genuine) is only known after a customer complaint, or thanks to the result of a fraud investigation. Therefore, in a realistic scenario, the annotated data available to train a model and start making prediction for a given day are anterior to that day minus the delay period. Setting a delay period of one week is simplistic. It assumes that the labels (fraudulent or genuine) for all transactions are known exactly one week after they occurred. 
# 
# The function below is creating two Snowpark data frames that will return the training and test data sets.

# %%
def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7,delta_delay=7,delta_test=7):
    
    # Get the training set data
    # Training data is not based on sampling but on a time period
    
    train_df = transactions_df.filter((F.col("TX_DATETIME") >= F.lit(start_date_training)) & (F.col("TX_DATETIME") <= F.lit(start_date_training+datetime.timedelta(days=delta_train))))
    # Get the test set data
    test_dfs = []
    
    # Note: Cards known to be frauded after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed
    
    # First, get known frauded customers from the training set
    known_frauded_customers = train_df.filter(F.col("TX_FRAUD")==F.lit(1)).select(F.col("CUSTOMER_ID"))
    
    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = int(train_df.select(F.min(F.col("TX_TIME_DAYS"))).to_pandas()['MIN("TX_TIME_DAYS")'].values[0])
    # Then, for each day of the test set
    # Get the customers/cards that was not known in the training data and in the delayperiod...
    for day in range(delta_test):
    
        # Get test data for one day, increased by 1 for each loop (starting with 0)
        test_df_day = transactions_df.filter(F.col("TX_TIME_DAYS") == start_tx_time_days_training+
                                                                    delta_train+delta_delay+
                                                                    day)
        
        # Frauded cards from that test day, minus the delay period, are added to the pool of known frauded customers
        test_df_day_delay_period = transactions_df.filter(F.col("TX_TIME_DAYS") == start_tx_time_days_training+
                                                                                delta_train+
                                                                                day-1)
        # fradulent customers identified during the delay period
        new_frauded_customers = test_df_day_delay_period.filter(F.col("TX_FRAUD")==F.lit(1)).select(F.col("CUSTOMER_ID"))
        
        # known_frauded_customers has fradulent customers identified in the training data
        # combine those eith fradulent customers in the delay period, remove duplicates
        known_frauded_customers = known_frauded_customers.union(new_frauded_customers)
        
        # Get the transactions for customers that is not in known_frauded_customers
        test_df_day = test_df_day.join(known_frauded_customers, test_df_day.col("CUSTOMER_ID") == known_frauded_customers.col("CUSTOMER_ID"), 'left')\
                          .filter(F.is_null(known_frauded_customers.col("CUSTOMER_ID")))\
                          .select(F.col("TRANSACTION_ID"), F.col("TX_DATETIME"), test_df_day.col("CUSTOMER_ID").alias("CUSTOMER_ID")\
                                  , F.col("TERMINAL_ID"), F.col("TX_TIME_SECONDS"), F.col("TX_TIME_DAYS"), F.col("TX_AMOUNT"), F.col("TX_FRAUD")\
                                  , F.col("TX_FRAUD_SCENARIO"), F.col("TX_DURING_WEEKEND"), F.col("TX_DURING_NIGHT"), F.col("CUST_AVG_AMOUNT_1")\
                                  , F.col("CUST_CNT_TX_1"), F.col("CUST_AVG_AMOUNT_7"), F.col("CUST_CNT_TX_7"), F.col("CUST_AVG_AMOUNT_30")\
                                  , F.col("CUST_CNT_TX_30"), F.col("NB_TX_WINDOW_1"), F.col("TERM_RISK_1"), F.col("NB_TX_WINDOW_7"), F.col("TERM_RISK_7")\
                                  , F.col("NB_TX_WINDOW_30"), F.col("TERM_RISK_30"))
        # Store as a temporary table?
        # Add it to our test data
        test_dfs.append(test_df_day)
        
    test_df = test_dfs[0].filter(F.is_null(F.col("CUSTOMER_ID")))
    for df in test_dfs:
        test_df = test_df.union(df)
    
    # Sort data sets by ascending order of transaction ID
    train_df=train_df.sort(F.col("TRANSACTION_ID"))
    test_df=test_df.sort(F.col("TRANSACTION_ID"))
    
    return (train_df, test_df)

# %%
(df_train, df_test)=get_train_test_set(df_trx,start_date_training,
                                       delta_train=7,delta_delay=7,delta_test=7)

# %% [markdown]
# How many rows training data set?

# %%
df_train.count()

# %% [markdown]
# How many fraud rows?

# %%
df_train.select(F.sum(F.col("TX_FRAUD"))).show()

# %% [markdown]
# Training data fradulent rows (since Fraud is a 0/1 value it is easier to summarize than to count with filter)

# %% [markdown]
# The test data set number of rows

# %%
df_test.count()

# %% [markdown]
# Test data fraudulent rows

# %%
df_test.select(F.sum(F.col("TX_FRAUD"))).show()

# %% [markdown]
# If we get the mean of the Fraud field we can get the propotion of frauds in our test data set (if we multiply with 100 we get the precentage)

# %%
df_test.select(F.avg(F.col("TX_FRAUD"))).show()

# %% [markdown]
# Have a look at the schema of our training dataframe, we can itirate through the columns (field) to generate a nicer output. 

# %%
for col in df_train.schema.fields:
    print(f"{col.name}, Nullable: {col.nullable}, {col.datatype}")

# %% [markdown]
# Since Snowflake today does not have in-database training of models we need to pull back our data locally into pandas dataframes

# %%
train_df = df_train.to_pandas()
test_df = df_test.to_pandas()

# %% [markdown]
# ### Train the model
# 
# We will define the input and output features as follows:
# 
# * The output feature will be the transaction label TX_FRAUD
# * The input features will be the transaction amount TX_AMOUNT, as well as all the features that were computed in the previous section, which characterize the context of a transaction.

# %%
output_feature="TX_FRAUD"

input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUST_CNT_TX_1',
       'CUST_AVG_AMOUNT_1', 'CUST_CNT_TX_7', 'CUST_AVG_AMOUNT_7', 'CUST_CNT_TX_30',
       'CUST_AVG_AMOUNT_30', 'NB_TX_WINDOW_1', 'TERM_RISK_1', 'NB_TX_WINDOW_7','TERM_RISK_7',
        'NB_TX_WINDOW_30','TERM_RISK_30']


# %%
# Train a model using the RandomForest classifier

classifier = sklearn.ensemble.RandomForestClassifier(random_state=0,n_jobs=-1)
classifier.fit(train_df[input_features], train_df[output_feature])


# %%
# Get the probablities for fraud for our train and test data sets
predictions_train=classifier.predict_proba(train_df[input_features])[:,1]
predictions_test=classifier.predict_proba(test_df[input_features])[:,1]

# %%
# Add the probability to the test data set
test_df['TX_FRAUD_PREDICTED']=predictions_test
test_df.head()

# %% [markdown]
# ### Assess the performance of the model.
# 
# We will compute three performance metrics: The AUC ROC, Average Precision (AP), and Card Precision top-𝑘 (CP@k)
# 
# * The Card Precision top-𝑘 is the most pragmatic and interpretable measure. It takes into account the fact that investigators can only check a maximum of 𝑘 potentially fraudulent cards per day. It is computed by ranking, for every day in the test set, the most fraudulent transactions, and selecting the 𝑘 cards whose transactions have the highest fraud probabilities. The precision (proportion of actual frauded cards out of predicted frauded cards) is then computed for each day. The Card Precision top-𝑘 is the average of these daily precisions. The number 𝑘 will be set to 100 (that is, it is assumed that only 100 cards can be checked every day). 
# 
# * The Average Precision is a proxy for the Card Precision top-𝑘, that integrates precisions for all possible 𝑘 values.
# 
# * The AUC ROC is an alternative measure to the Average Precision, which gives more importance to scores obtained with higher 𝑘 values. It is less relevant in practice since the performances that matter most are those for low 𝑘 values. We however also report it since it is the most widely used performance metric for fraud detection in the literature. 
# 
# Note that all three metrics provide values in the interval [0,1], and that higher values mean better performances.

# %%
def card_precision_top_k_day(df_day,top_k):
    
    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID, 
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)
            
    # Get the top k most suspicious cards
    df_day_top_k=df_day.head(top_k)
    list_detected_frauded_cards=list(df_day_top_k[df_day_top_k.TX_FRAUD==1].CUSTOMER_ID)
    
    # Compute precision top k
    card_precision_top_k = len(list_detected_frauded_cards) / top_k
    
    return list_detected_frauded_cards, card_precision_top_k

def card_precision_top_k(predictions_df, top_k, remove_detected_frauded_cards=True):

    # Sort days by increasing order
    list_days=list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()
    
    # At first, the list of detected frauded cards is empty
    list_detected_frauded_cards = []
    
    card_precision_top_k_per_day_list = []
    nb_frauded_cards_per_day = []
    
    # For each day, compute precision top k
    for day in list_days:
        
        df_day = predictions_df[predictions_df['TX_TIME_DAYS']==day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]
        
        # Let us remove detected frauded cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_frauded_cards)==False]
        
        nb_frauded_cards_per_day.append(len(df_day[df_day.TX_FRAUD==1].CUSTOMER_ID.unique()))
        
        detected_frauded_cards, card_precision_top_k = card_precision_top_k_day(df_day,top_k)
        
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        
        # Let us update the list of detected frauded cards
        if remove_detected_frauded_cards:
            list_detected_frauded_cards.extend(detected_frauded_cards)
        
    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    
    # Returns precision top k per day as a list, and resulting mean
    return nb_frauded_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k

def performance_assessment(predictions_df, output_feature='TX_FRAUD', 
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]], 
                           columns=['AUC ROC','Average precision'])
    
    for top_k in top_k_list:
    
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances['Card Precision@'+str(top_k)]=mean_card_precision_top_k
        
    if rounded:
        performances = performances.round(3)
    
    return performances

# %%
predictions_df=test_df
predictions_df['predictions']= predictions_test 
    
performance_assessment(predictions_df, top_k_list=[100])

# %% [markdown]
# The most interpretable metric is the Card Precision@100, which tells us that every day, 30% of the cards with the highest fraudulent scores were indeed compromised. Since the percentage of frauds in the test set is 0.6%, this proportion of detected frauds is high, and means that the classifier indeed manages to do much better than chance.
# 
# The interpretation of the AUC ROC and Average Precision is less straightforward. However, by definition, it is known that a random classifier would give an AUC ROC of 0.5, and an Average Precision of 0.006 (the proportion of frauds in the test set). The obtained values are much higher (0.866) and (0.65), confirming the ability of the classifier to provide much better predictions than a random model.

# %%
predictions_df['predictions']=0.5
    
performance_assessment(predictions_df, top_k_list=[100])

# %% [markdown]
# ### Deploy model to Snowflake
# 
# In order to use the model in Snowflake for scoring data we need to create a Python UDF.
# 
# There is two ways on how we can deploy the model object, classifier, to Snowflake. 
# 1. We can use the variable directly and have the model object included in UDF code
# 2. We can save it to a file and upload it to a stage and load it form the stage when the UDF is called
# 
# 
# In this example we will use the second option.
# 
# First we save the model object to a file.

# %%
from joblib import dump, load
dump(classifier, 'predict_fraud.joblib')

# %% [markdown]
# Since we are creating a permanent function we need to use a stage so Snowpark can upload our code and model

# %%
session.sql("CREATE STAGE IF NOT EXISTS UDFSTAGE").collect()

# %% [markdown]
# Upload the model file to the new stage

# %%
session.file.put("predict_fraud.joblib", "@UDFSTAGE", auto_compress=False, overwrite=True)

# %% [markdown]
# Below code creates a Python UDF in Snowflake that uses our model (that we trained earlier) for scroing of data.
# 
# We need to provide the packages we will use in the function by **add_packages** and since we are seperatinmg our model from the function we also need to make the UDF aware of the filw by using **add_import**

# %%
session.clear_imports()
session.clear_packages()
session.add_import("@UDFSTAGE/predict_fraud.joblib")
session.add_packages("joblib", "scikit-learn", "cachetools")

# %% [markdown]
# We will create two functions to be used for scoring.
# 
# **read_file** is a helper function that is used with **cachetools** to make sure we only load the model file once and **detect_fraud_batch** is the function that does the scoring. We are using the Python UDF Batch API to create a vectorized UDF that takes a Pandas Dataframe as input, meaning we get fewer calls since each call is on a number of rows at the time (as opposite to a Scalar UDF where we get one row for each call).

# %%
@cachetools.cached(cache={})
def read_file(filename):
    import joblib
    import sys
    import os
    
    import_dir = sys._xoptions.get("snowflake_import_directory")
    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m

@F.udf(name = 'detect_fraud_batch_udf', is_permanent = True, replace = True, stage_location = '@UDFSTAGE')
def detect_fraud_batch(df: PandasDataFrame[float, int, int, int, float, int, float, int, float, int, float, int, float, int, float]) -> PandasSeries[float]:
    import sklearn
    
    pipeline = read_file('predict_fraud.joblib')
    return pipeline.predict_proba(df)[:,1]

# %% [markdown]
# We have now deployed the function and the model to Snowflake and can use it.

# %%
df_cust_trx = session.table("CUSTOMER_TRX_FRAUD_FEATURES").filter(F.col("TX_DATETIME") > "2019-07-15 00:00:00")
df_cust_trx.show()

# %% [markdown]
# We can use the function name, **detect_fraud_batch**, when using the Snowpark API to call the UDF.

# %%
df_cust_trx.select(F.col("TRANSACTION_ID"), F.col("TX_DATETIME"), F.col("CUSTOMER_ID"), F.col("TERMINAL_ID"), 
                    detect_fraud_batch(input_features).as_("FRAUD_PROB")).show()

# %% [markdown]
# If we want to do the same with SQL we could run the following where we need to use the name of the UDF, **detect_fraud_batch_udf**
# 
# `SELECT TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT ,detect_fraud_batch_udf(TX_AMOUNT,TX_DURING_WEEKEND, TX_DURING_NIGHT, CUST_CNT_TX_1, CUST_AVG_AMOUNT_1, CUST_CNT_TX_7, CUST_AVG_AMOUNT_7, CUST_CNT_TX_30,CUST_AVG_AMOUNT_30, NB_TX_WINDOW_1, TERM_RISK_1, NB_TX_WINDOW_7,TERM_RISK_7, NB_TX_WINDOW_30,TERM_RISK_30) AS FRAUD_PROB 
# FROM CUSTOMER_TRX_FRAUD_FEATURES 
# WHERE TX_DATETIME > '2019-07-15 00:00:00' LIMIT 10;`

# %%



