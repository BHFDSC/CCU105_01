# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_7a_join_all_datasets_with_IMD
# MAGIC
# MAGIC
# MAGIC **Description** Join IMD, comorbiditities, and mortality with the cohort saved in CCU105_05c. 
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

# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %run "./CCU105_01_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load the dataset saved in part 4 

# COMMAND ----------

# df_cohort_05c = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_3')
df_cohort_05c = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_4') # part 4 dataset has the added KRT outcome variable

# COMMAND ----------

display(df_cohort_05c.limit(20))

# COMMAND ----------

count_var(df_cohort_05c, "person_id")

# COMMAND ----------

df_cohort_05c.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Link with IMD
# MAGIC
# MAGIC The closest LSOA to MI index date was extracted in CCU105_05a

# COMMAND ----------

proj_params.path_lsoa_2011_imd_lookup

# COMMAND ----------

imd_lookup = spark.table(f'''{proj_params.path_lsoa_2011_imd_lookup}''')
display(imd_lookup.limit(10))

# COMMAND ----------



# COMMAND ----------

# Link with curated IMD table for decile and quintiles
df_3_a = df_cohort_05c.join(imd_lookup.withColumnRenamed("LSOA_2011", "lsoa"), on="lsoa", how="left")

# COMMAND ----------

display(df_3_a.groupBy("IMD_2019_QUINTILES").agg(f.count("*")))

# COMMAND ----------

display(df_3_a.limit(10))

# COMMAND ----------

count_var(df_3_a, "person_id")

# COMMAND ----------

#numerical IMD
imd_numerical = spark.table(f'''reference_data.dss_corporate.english_indices_of_dep_v02''''')
display(imd_numerical.limit(50))

# COMMAND ----------

imd_num_2015 = imd_numerical.filter(f.col("IMD_YEAR")==2015)
imd_num_2019 = imd_numerical.filter(f.col("IMD_YEAR")==2019)
print(imd_numerical.count())
print(imd_num_2015.count())
print(imd_num_2019.count())

# COMMAND ----------

# Any null IMDs?
print(imd_num_2015.filter(f.col("IMD").isNull()).count())
print(imd_num_2019.filter(f.col("IMD").isNull()).count())

# COMMAND ----------

# Are LSOA to IMD relations unique?
print(imd_num_2015.dropDuplicates(subset=["LSOA_CODE_2011", "IMD"]).count())
print(imd_num_2019.dropDuplicates(subset=["LSOA_CODE_2011", "IMD"]).count())

# COMMAND ----------

imd_num_2015.columns

# COMMAND ----------

imd_num_2015_minimal = (imd_num_2015
                        .select(["LSOA_CODE_2011", "IMD"])
                        .withColumnRenamed("LSOA_CODE_2011", "lsoa")
                        .withColumnRenamed("IMD", "IMD_2015_numerical")
                        )
imd_num_2019_minimal = (imd_num_2019
                        .select(["LSOA_CODE_2011", "IMD"])
                        .withColumnRenamed("LSOA_CODE_2011", "lsoa")
                        .withColumnRenamed("IMD", "IMD_2019_numerical")
                        )

# COMMAND ----------

# Link with curated IMD table for decile and quintiles
df_3 = df_3_a.join(imd_num_2015_minimal, on="lsoa", how="left").join(imd_num_2019_minimal, on="lsoa", how="left")

# COMMAND ----------

count_var(df_3, "person_id")

# COMMAND ----------

display(df_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Join comorbidities
# MAGIC
# MAGIC

# COMMAND ----------

df_comorbidity = spark.table(f'{proj_params.dbc}.ccu105_comorbidities')

# COMMAND ----------

display(df_comorbidity.limit(10))

# COMMAND ----------

count_var(df_comorbidity, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Rename comorbidity names. There are similar names in MINAP 

# COMMAND ----------

cmrbd_list = list(df_comorbidity.columns)
print(cmrbd_list[1:])

# COMMAND ----------

for item in cmrbd_list[1:]:
    df_comorbidity = df_comorbidity.withColumnRenamed(item, f'comorbidity_{item}')

# COMMAND ----------

df_4 = df_3.join(df_comorbidity, on="person_id", how="left")

# COMMAND ----------



# COMMAND ----------

count_var(df_4, "person_id")

# COMMAND ----------

display(df_4.limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Join mortality

# COMMAND ----------

df_death = spark.table(f'{proj_params.dbc}.ccu105_mortality_outocmes')

# COMMAND ----------

display(df_death.limit(10))

# COMMAND ----------

display(df_death.groupBy(["flag_mortality_all_cause","flag_mortality_ckd", "flag_mortality_esrd", "flag_mortality_cvd" ]).agg(f.count("*")))

# COMMAND ----------

count_var(df_death, "person_id")

# COMMAND ----------

df_5 = df_4.join(df_death, on="person_id", how="left")

# COMMAND ----------

count_var(df_5, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.  Save

# COMMAND ----------

# spark.sql(f'''drop table if exists {proj_params.dbc}.ccu105_final_dataset_alpha''')

# COMMAND ----------

# df_5.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_final_dataset_alpha')
save_table(df_5, out_name=f'ccu105_final_dataset_alpha', save_previous=True)

# COMMAND ----------

df_5.columns

# COMMAND ----------

df_5  = spark.table(f'{proj_params.dbc}.ccu105_final_dataset_alpha')

# COMMAND ----------

count_var(df_5, "person_id")

# COMMAND ----------

# any outcomes after mi index and before study end date?
display(df_5.filter(f.col("ckd_dialysis_transplant_outcome")>f.col("mi_index_date")).filter(f.col("ckd_dialysis_transplant_outcome")<=f.col("study_end_date")))

# COMMAND ----------

# any outocmes after mi index?
display(df_5.filter(f.col("ckd_dialysis_transplant_outcome")>f.col("study_end_date")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Check

# COMMAND ----------

# check difference between old dataset and current
old = spark.table(f'dsa_391419_j3w9t..ccu105_final_dataset_alpha_pre20260115_180155')
new = spark.table(f'dsa_391419_j3w9t..ccu105_final_dataset_alpha')

old_not_new = (
    old
    .join(new, 'person_id', how='left_anti')
)

new_not_old = (
    new
    .join(old, 'person_id', how='left_anti')
)

overlap = (
    new
    .join(old, 'person_id', how='inner')
)

display(overlap)
display(new_not_old)
display(old_not_new)

# COMMAND ----------

count_var(old, 'person_id')
count_var(new, 'person_id') 

# COMMAND ----------



# COMMAND ----------

# See if those records in old CCU105 dataset but not in new CCU105 dataset, are largely null for the MINAP variables
 

# COMMAND ----------

display(new.filter(f.col('person_id') == ''))

# COMMAND ----------

# Check which patients arenot in each dataset, respectively
count_var(overlap, 'person_id')
count_var(old_not_new, 'person_id')
count_var(new_not_old, 'person_id')

# COMMAND ----------

_check1 = (
    old_not_new
    .groupBy('year_month')
    .count()
)

display(_check1)

# COMMAND ----------

_check2 = (
    new_not_old
    .groupBy('year_month')
    .count()
)

display(_check2)

# COMMAND ----------

# See if patients with rows incomplete for _AGE_AT_ADMISSION in most recent MINAP batch (24/04/2025) are populated in previous batch

# COMMAND ----------

tab(old_not_new, 'archived_on') 
tab(new_not_old, 'archived_on') 

# COMMAND ----------

