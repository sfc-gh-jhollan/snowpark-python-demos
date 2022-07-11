# %% [markdown]
# # Credit Card Fraud Prediction - Loading Dataset using Snowpark Python
# 
# This example is based on the Machine Learning for Credit Card Fraud detection - Practical handbook, https://fraud-detection-handbook.github.io/fraud-detection-handbook/

# %% [markdown]
# ## Loading Credit Card Transactions into Snowflake
# 
# ### Import the dependencies and connect to Snowflake

# %%
# Snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.types as T
import snowflake.snowpark.functions as F

# Print the version of Snowpark we are using
from importlib.metadata import version
version('snowflake_snowpark_python')

# %%
# Other
import json

# %% [markdown]
# **Before connecting make sure you have updated creds.json with information for your Snowflake account**

# %%
with open('creds.json') as f:
    connection_parameters = json.load(f)

# %%
session = Session.builder.configs(connection_parameters).create()

# %% [markdown]
# The **get_** functions can be use to get information about the current database, schema, role etc

# %%
print(f"Current schema: {session.get_fully_qualified_current_schema()}, current role: {session.get_current_role()}, current warehouse:  {session.get_current_warehouse()}")

# %% [markdown]
# ### Define Staging Area and the Schema for the transaction table
# 
# Using SQL we can create a internal stage and then use the **put** function to uplad the **fraud_transactions.csv.gz** file to it.

# %%
stage_name = "FRAUD_DATA"
# Create a internal staging area for uploading the source file
session.sql(f"CREATE or replace STAGE {stage_name}").collect()

# Upload the source file to the stage
putResult = session.file.put("data/fraud_transactions.csv.gz", f"@{stage_name}", auto_compress=False, overwrite=True)

putResult

# %% [markdown]
# Define the schma for our **CUSTOMER_TRANSACTIONS_FRAUD** table

# %%
# Define the schema for the Frauds table
dfCustTrxFraudSchema = T.StructType(
    [
        T.StructField("TRANSACTION_ID", T.IntegerType()),
        T.StructField("TX_DATETIME", T.TimestampType()),
        T.StructField("CUSTOMER_ID", T.IntegerType()),
        T.StructField("TERMINAL_ID", T.IntegerType()),
        T.StructField("TX_AMOUNT", T.FloatType()),
        T.StructField("TX_TIME_SECONDS", T.IntegerType()),
        T.StructField("TX_TIME_DAYS", T.IntegerType()),
        T.StructField("TX_FRAUD", T.IntegerType()),
        T.StructField("TX_FRAUD_SCENARIO", T.IntegerType())
    ]
)

# %% [markdown]
# Load the **fraud_transactions.csv.gz** to a DataFrame reader and save into a table

# %%
# Crete a reader
dfReader = session.read.schema(dfCustTrxFraudSchema)

# Get the data into the data frame
dfCustTrxFraudRd = dfReader.csv(f"@{stage_name}/fraud_transactions.csv.gz")

# %%
# Write the dataframe in a table
ret = dfCustTrxFraudRd.write.mode("overwrite").saveAsTable("CUSTOMER_TRANSACTIONS_FRAUD")

# %% [markdown]
# ### Read the data from the staging area and create CUSTOMER_TRANSACTIONS_FRAUD, CUSTOMERS and TERMINALS tables

# %%
# Now create Customers and Terminal tables

dfCustTrxFraudTb =session.table("CUSTOMER_TRANSACTIONS_FRAUD")

dfCustomers = dfCustTrxFraudTb.select(F.col("CUSTOMER_ID")).distinct().sort(F.col("CUSTOMER_ID"))

dfTerminals = dfCustTrxFraudTb.select(F.col("TERMINAL_ID")).distinct().sort(F.col("TERMINAL_ID"))
                                
ret2 = dfCustomers.write.mode("overwrite").saveAsTable("CUSTOMERS")

ret3 = dfTerminals.write.mode("overwrite").saveAsTable("TERMINALS")

# %%
dfCustTrxFraudTb.show()

# %%



