# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_05c_cohort_inclusion_exclusion_part_2
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the base cohort from MINAP and applying incluison and exclusion criteria
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **``** : 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Setup and parameters

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

#from functools import reduce

import pandas as pd
import pyspark.pandas as ps
import numpy as np

import re
import io
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %run "./CCU105_01_parameters"

# COMMAND ----------

# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load the dataset saved in part 1

# COMMAND ----------

df_1 = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_1')

# COMMAND ----------

display(df_1.limit(10))

# COMMAND ----------

count_var(df_1, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Inclusion/Exclusion 
# MAGIC

# COMMAND ----------

df_1.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3.1. Null IDs

# COMMAND ----------

# Null IDs
count_var(df_1.filter(f.col("person_id").isNull()), "person_id")


# COMMAND ----------

df_3_1 = df_1.filter(f.col("person_id").isNotNull())


# COMMAND ----------

count_var(df_3_1, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Missing date of birth

# COMMAND ----------

count_var(df_3_1.filter(f.col("date_of_birth").isNull()), "person_id")

# COMMAND ----------

# 
display(df_3_1.filter(f.col("date_of_birth").isNull()).groupBy("isin_demo").agg(f.count("*")))

# COMMAND ----------

display(df_3_1.filter(f.col("date_of_birth").isNull()).groupBy("in_gdppr").agg(f.count("*")))

# COMMAND ----------

# check if this is due to not including MINAP in patient characteristics table
gdppr_ids = gdppr.select(["NHS_NUMBER_DEID"]).withColumnRenamed("NHS_NUMBER_DEID", "person_id").withColumn("in_gdppr_2", f.lit(1))
df_temp = df_3_1.join(gdppr_ids, on="person_id", how="left"    )
display(df_temp.filter(f.col("date_of_birth").isNull()).groupBy("in_gdppr_2").agg(f.count("*")))

# COMMAND ----------

# Drop null DOBs
df_3_2 = df_3_1.filter(f.col("date_of_birth").isNotNull())

# COMMAND ----------

count_var(df_3_2, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3. Death flag =1 and DOD is missing

# COMMAND ----------

display(df_3_2.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# Death flag is 1, DOD  missing
df_temp = (df_3_2
           .filter(f.col("death_flag") == 1)
           .filter(f.col("date_of_death").isNull())
      )
count_var(df_temp, "person_id")


# COMMAND ----------

# Change if needed
df_3_3 = df_3_2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4. DOB > DOD

# COMMAND ----------

df_temp = (df_3_3
           .filter(f.col("death_flag") == 1)
           .filter(f.col("date_of_birth") > f.col("date_of_death"))
      )
count_var(df_temp, "person_id")

# COMMAND ----------

df_3_4 = df_3_3

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5. Any single use IDs?

# COMMAND ----------

display(df_3_4.groupBy("single_use_id", "valid_nhs_number", "mps_id").agg(f.count("*")))

# COMMAND ----------

df_3_4.select(["single_use_id", "valid_nhs_number", "mps_id"]).dtypes

# COMMAND ----------

df_3_5 = df_3_4.filter((f.col("valid_nhs_number")==True)|(f.col("mps_id")==True))
display(df_3_5.groupBy("single_use_id", "valid_nhs_number", "mps_id").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_5, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3.6. Not in GDPPR
# MAGIC
# MAGIC Keep the flag and confirm with the CCU105 team

# COMMAND ----------


display(df_3_5.groupBy("in_gdppr").agg(f.count("*")))


# COMMAND ----------

df_temp = (df_3_5
           .filter(f.col("in_gdppr") == 0)
      )
count_var(df_temp, "person_id")

# COMMAND ----------

df_3_6 = df_3_5


# COMMAND ----------

count_var(df_3_6, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7. Sex

# COMMAND ----------

display(df_3_6.groupBy(["sex", "female_specific", "male_specific"]).agg(f.count("*")))

# COMMAND ----------

# Female with male specific codes
count_var(df_3_6.filter(f.col("sex")=="F").filter(f.col("male_specific")==1), "person_id")

# COMMAND ----------

# Male with female specific codes
count_var(df_3_6.filter(f.col("sex")=="M").filter(f.col("female_specific")==1), "person_id")

# COMMAND ----------

# invalid or null
count_var(df_3_6.filter((f.col("sex")=="I")|(f.col("sex").isNull())), "person_id")

# COMMAND ----------

# Drop invalid sex 

df_3_7 = (df_3_6
          .filter(((f.col("sex")=="M")&(f.col("female_specific")==0))
                   | 
                   ((f.col("sex")=="F")&(f.col("male_specific")==0))))

# COMMAND ----------

count_var(df_3_7, "person_id")

# COMMAND ----------

display(df_3_7.groupBy(["sex", "female_specific", "male_specific"]).agg(f.count("*")))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8. LSOA missing or non-England

# COMMAND ----------

# LSOA missing
df_temp = (df_3_7
           .filter(f.col("lsoa").isNull())
      )
count_var(df_temp, "person_id")


# COMMAND ----------

# Non-England LSOA
display(df_1.groupBy("lsoa_1").agg(f.count("*")))

# COMMAND ----------

df_temp = (df_3_7
           .filter(f.col("lsoa_1") != "E")
      )
count_var(df_temp, "person_id")

# COMMAND ----------

# Keep only England LSOAs
df_3_8 = df_3_7.filter(f.col("lsoa_1")=="E")

# COMMAND ----------


count_var(df_3_8, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Save

# COMMAND ----------

# spark.sql(f'drop table if exists {proj_params.dbc}.ccu105_minap_cohort_part_2')

# COMMAND ----------

# df_3_8.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_minap_cohort_part_2')
save_table(df_3_8, out_name=f'ccu105_minap_cohort_part_2', save_previous=True)

# COMMAND ----------

df_3_8_reload = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_2')    

# COMMAND ----------

df_3_8_reload.columns

# COMMAND ----------

