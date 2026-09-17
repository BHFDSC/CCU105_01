# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_06a_outcome_mortality
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of mortality outocme. This will be joined with the cohort in CCU105_05c
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
# MAGIC # 2. Load the dataset saved in part 3 (for Patient IDs)

# COMMAND ----------

df_part_3 = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_3')

# COMMAND ----------

display(df_part_3.limit(20))

# COMMAND ----------

count_var(df_part_3, "person_id")

# COMMAND ----------

df_part_3.columns

# COMMAND ----------

# All deaths
death_ids = df_part_3.filter(f.col("death_flag")==1).select(["person_id", "mi_index_date", "date_of_death", "study_end_date"])

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Date QC control
# MAGIC
# MAGIC - Add a flag indicating whether the death date is recorded before MI index
# MAGIC - Add a flag indicating whether deaths is after study end date

# COMMAND ----------

# anyone with recorded date_of_death before mi_index_date
display(death_ids.filter(f.col("date_of_death")<f.col("mi_index_date")))

# COMMAND ----------

# Make a flag to show death before mi
death_ids_3 = death_ids.withColumn("flag_death_before_mi", f.when(f.col("date_of_death")<f.col("mi_index_date"), f.lit(1)).otherwise(f.lit(0)))
display(death_ids_3.groupBy("flag_death_before_mi").agg(f.count("*")))

# COMMAND ----------

# Make a flag to show death after study end date
death_ids_3 = death_ids_3.withColumn("flag_death_after_study_end", f.when(f.col("date_of_death")>f.col("study_end_date"), f.lit(1)).otherwise(f.lit(0)))
display(death_ids_3.groupBy("flag_death_after_study_end").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. All cause and cause specific mortality flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. All cause mortality
# MAGIC
# MAGIC All death events are all-cause mortality, add an explicit flag. 

# COMMAND ----------

death_ids_4 = death_ids_3.withColumn("flag_mortality_all_cause", f.lit("1")) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Cause specific
# MAGIC
# MAGIC - CKD: ICD-10 codes for CKD
# MAGIC - ESRD: ICD-10 codes for ESRD
# MAGIC - CVD I00-I99

# COMMAND ----------

ckd_codelist = spark.table(f'{proj_params.dbc}.kdsc_codelist_ckd_bhfdsc_20250612')
display(ckd_codelist.groupBy("terminology", "name").agg(f.count("*")))

# COMMAND ----------

bhf_ckd_icd_codelist = (ckd_codelist
                    .filter(f.col("terminology")=="ICD10")
                    .filter(f.col("name").isin(["ckd", "dialysis", "transplant"]))
                    .withColumn("name_2", f.when(f.col("name")!="ckd", f.lit("esrd")).otherwise(f.col("name")))
                    )

# COMMAND ----------

display(bhf_ckd_icd_codelist.groupBy("terminology", "name_2", "name").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3. Load long death table

# COMMAND ----------

death_long = spark.table(f'{proj_params.path_cur_deaths_long}')
death_long = death_long.withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
display(death_long.limit(100))

# COMMAND ----------

count_var(death_ids_3, "person_id")

# COMMAND ----------

cohort_deaths = death_ids_3.join(death_long, on= "person_id", how="left")

# COMMAND ----------

count_var(cohort_deaths, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4. CKD and ESRD-specific mortality

# COMMAND ----------

ckd_codelist = bhf_ckd_icd_codelist.filter(f.col("name_2")=="ckd").select("name_2", "code").withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
esrd_codelist = bhf_ckd_icd_codelist.filter(f.col("name_2")=="esrd").select("name_2", "code").withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))

# COMMAND ----------

display(ckd_codelist)

# COMMAND ----------

display(esrd_codelist)

# COMMAND ----------

ckd_deaths = cohort_deaths.join(ckd_codelist, on= "code", how="inner")
ckd_deaths_unique = ckd_deaths.select(["person_id"]).distinct().withColumn("flag_mortality_ckd", f.lit(1))
count_var(ckd_deaths_unique, "person_id")

# COMMAND ----------

esrd_deaths = cohort_deaths.join(esrd_codelist, on= "code", how="inner")
esrd_deaths_unique = esrd_deaths.select(["person_id"]).distinct().withColumn("flag_mortality_esrd", f.lit(1))
count_var(esrd_deaths_unique, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 5. CVD specific

# COMMAND ----------

cvd_deaths = cohort_deaths.filter(f.col("code").startswith("i"))
cvd_deaths_unique = cvd_deaths.select(["person_id"]).distinct().withColumn("flag_mortality_cvd", f.lit(1))
count_var(cvd_deaths_unique, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Join 

# COMMAND ----------

death_output = (death_ids_4.select(["person_id", "flag_mortality_all_cause"])
                .join(ckd_deaths_unique, on= "person_id", how="left")
                .join(esrd_deaths_unique, on= "person_id", how="left")
                .join(cvd_deaths_unique, on= "person_id", how="left")
                .fillna(0)
                )

# COMMAND ----------

death_output.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # 6.  Save

# COMMAND ----------

# death_output.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_mortality_outocmes')
save_table(death_output, out_name=f'ccu105_mortality_outcomes', save_previous=True)

# COMMAND ----------

death_output_reload = spark.table(f'{proj_params.dbc}.ccu105_mortality_outocmes')

# COMMAND ----------

display(death_output_reload.groupBy(["flag_mortality_all_cause","flag_mortality_ckd", "flag_mortality_esrd", "flag_mortality_cvd" ]).agg(f.count("*")))

# COMMAND ----------

count_var(death_output_reload, "person_id")

# COMMAND ----------

