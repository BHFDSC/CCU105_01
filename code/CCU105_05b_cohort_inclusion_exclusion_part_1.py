# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_05b_cohort_inclusion_exclusion_part_1
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the base cohort from MINAP. Note: the MINAP has been curated in the previous botebook from the last two batches (2017 records are extracted from the older batch)
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
# MAGIC # 2. Load MINAP, patient characteristics, and token ID table

# COMMAND ----------

minap_curated = spark.table(f'{proj_params.dbc}.ccu105_minap_curated')

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

display(minap_curated.limit(10))

# COMMAND ----------

proj_params.path_demographics

# COMMAND ----------

demo = spark.table(f'{proj_params.path_demographics}')
display(demo.limit(10))

# COMMAND ----------

display(token.groupBy("archived_on").agg(f.count("*")))

# COMMAND ----------

display(token.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. Link MINAP with both tables

# COMMAND ----------

demo_sel = (demo.select(
    [
        "person_id",
        "date_of_birth",
        "sex",
        "ethnicity_18_group",
        "ethnicity_5_group",
        "death_flag",
        "date_of_death",
        "in_gdppr",
        "gdppr_min_date",
    ]
)
.withColumn(
    "in_gdppr",
    f.when(f.col("in_gdppr").isNull(), f.lit(0)).otherwise(f.col("in_gdppr")))
.withColumn(
    "death_flag",
    f.when(f.col("death_flag").isNull(), f.lit(0)).otherwise(f.col("death_flag"))
)
.withColumn("isin_demo", f.lit(1))
)

# COMMAND ----------

display(demo_sel.groupBy("in_gdppr").agg(f.count("*")))

# COMMAND ----------

display(demo_sel.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

token_sel = (token
             .select([ 'pseudo_id',
                      'valid_nhs_number',
                       'mps_id',

                      'single_use_id'])
             .withColumnRenamed('pseudo_id', 'person_id')
             .withColumn("isin_token", f.lit(1))
             )

# COMMAND ----------

display(token_sel.groupBy("single_use_id", "valid_nhs_number", "mps_id").agg(f.count("*")))

# COMMAND ----------

count_var(minap_curated, "person_id")

# COMMAND ----------

print(proj_params.study_start_date)
print(proj_params.study_end_date)
proj_params.mi_period_end_date

# COMMAND ----------

# check dates
df_tmp_1 = (minap_curated
          .withColumn("study_start_date", f.to_date(f.lit(proj_params.study_start_date)))
          .withColumn("study_end_date", f.to_date(f.lit(proj_params.study_end_date)))
          .withColumn("mi_index_date", f.to_date(f.col("ARRIVAL_AT_HOSPITAL")))
          .withColumn("mi_period_end_date", f.to_date(f.lit(proj_params.mi_period_end_date)))
        
          )
# Check mi index dates
df_tmp_1 = df_tmp_1.select(["person_id","study_start_date", "study_end_date",  "ARRIVAL_AT_HOSPITAL", "mi_index_date", "mi_period_end_date"])
df_tmp_1 = df_tmp_1.withColumn("year", f.year(f.col("mi_index_date"))).withColumn("month", f.month(f.col("mi_index_date"))).withColumn("year_month", f.concat(f.col("year"), f.lit("_"), f.col("month")))
display(df_tmp_1.groupBy("year").agg(f.count("*")))

# COMMAND ----------

# minap_curated already has year and months columns baed on arrival at hospital (extraxted in the prvious notebook)
df_2_1_temp =(minap_curated 
         .join(token_sel, on='person_id', how='left'))
df_2_1_temp = df_2_1_temp.fillna({"isin_token":0})
         

# COMMAND ----------

display(df_2_1_temp.groupBy("year").agg(f.count("*")))

# COMMAND ----------

# Note: there are lots of null IDs
display(df_2_1_temp.groupBy("isin_token").agg(f.count("*")))

# COMMAND ----------

display(df_2_1_temp.groupBy("single_use_id", "valid_nhs_number", "mps_id").agg(f.count("*")))

# COMMAND ----------

display(df_2_1_temp.filter(f.col("valid_nhs_number").isNull()))

# COMMAND ----------

df_2_1 =(df_2_1_temp
         .join(demo_sel, on='person_id', how='left')
         )
df_2_1 = df_2_1.fillna({"isin_demo":0})

# COMMAND ----------

display(df_2_1.groupBy("isin_demo").agg(f.count("*")))

# COMMAND ----------

display(df_2_1.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

display(df_2_1.groupBy("in_gdppr").agg(f.count("*")))

# COMMAND ----------

count_var(df_2_1, "person_id")

# COMMAND ----------

display(df_2_1.limit(10))

# COMMAND ----------



# COMMAND ----------

display(df_2_1.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.2. Add index dates and age at MI index date

# COMMAND ----------

print(f'study_start_date = {proj_params.study_start_date}')
print(f'study_end_date = {proj_params.study_end_date}')
print(f'mi_period_end_date = {proj_params.mi_period_end_date}')



# COMMAND ----------

df_2_1.dtypes

# COMMAND ----------

display(df_2_1.select(["person_id", "ARRIVAL_AT_HOSPITAL"]).limit(5))

# COMMAND ----------

df_2_2 = (df_2_1
          .withColumn("study_start_date", f.to_date(f.lit(proj_params.study_start_date)))
          .withColumn("study_end_date", f.to_date(f.lit(proj_params.study_end_date)))
          .withColumn("mi_index_date", f.to_date(f.col("ARRIVAL_AT_HOSPITAL")))
          .withColumn("mi_period_end_date", f.to_date(f.lit(proj_params.mi_period_end_date)))
          .withColumn("mi_index_age", f.floor(f.datediff(f.col("mi_index_date"), f.col("date_of_birth"))/365.25))
          )


# COMMAND ----------

display(df_2_2.select(["person_id","study_start_date", "study_end_date",  "ARRIVAL_AT_HOSPITAL", "mi_index_date", "mi_period_end_date", "mi_index_age"]).limit(5))

# COMMAND ----------

df_2_2.select(["person_id","study_start_date", "study_end_date",  "ARRIVAL_AT_HOSPITAL", "mi_index_date", "mi_period_end_date", "mi_index_age"]).dtypes

# COMMAND ----------

# Check mi index dates
df_tmp = df_2_2.select(["person_id","study_start_date", "study_end_date",  "ARRIVAL_AT_HOSPITAL", "mi_index_date", "mi_period_end_date", "mi_index_age"])
df_tmp = df_tmp.withColumn("year", f.year(f.col("mi_index_date"))).withColumn("month", f.month(f.col("mi_index_date"))).withColumn("year_month", f.concat(f.col("year"), f.lit("_"), f.col("month")))
display(df_tmp.groupBy("year").agg(f.count("*")))

# COMMAND ----------

display(df_tmp.groupBy("year_month").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3. Join with HES APC MI
# MAGIC
# MAGIC - Later, exclude patients with MI before study start date

# COMMAND ----------

hes_apc_pheno = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

display(hes_apc_pheno.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

hes_apc_mi = hes_apc_pheno.filter((f.col("phenotype") == "MI"))
display(hes_apc_mi.limit(20))

# COMMAND ----------

hes_apc_mi_first = (hes_apc_mi
                    .filter(f.col("phenotype_rownum")==1)
                    .select(["person_id", "event_date_curated"])
                    .withColumnRenamed("event_date_curated", "mi_first_hes_apc_date")
                    )

# COMMAND ----------

count_var(hes_apc_mi_first, "person_id")

# COMMAND ----------

# join
df_2_3 = df_2_2.join(hes_apc_mi_first, on="person_id", how="left")


# COMMAND ----------

count_var(df_2_3, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4. Link with LSOA at index MI date 

# COMMAND ----------

df_lsoa = spark.table(proj_params.path_lsoa_multi)
display(df_lsoa.limit(10))

# COMMAND ----------

display(df_lsoa.groupBy("data_source").agg(f.count("*")))

# COMMAND ----------

# Join with IDs and MI index date to find the closest LSOA to the index date
df_join_lsoa = (df_2_3.select(["person_id", "mi_index_date"])
                .join(df_lsoa, on= "person_id", how="left")
                )

# COMMAND ----------

print(df_join_lsoa.filter(f.col("mi_index_date").isNull()).count())
print(df_join_lsoa.filter(f.col("record_date").isNull()).count())

# COMMAND ----------

# Add the absolute value of of date difference to find the closest LSOA (before or after)
df_join_lsoa = df_join_lsoa.withColumn("lsoa_date_diff", f.datediff(f.col("record_date"), f.col("mi_index_date")))
df_join_lsoa = df_join_lsoa.withColumn("lsoa_date_diff_abs", f.abs(f.col(("lsoa_date_diff"))))

# COMMAND ----------

display(df_join_lsoa.limit(100))

# COMMAND ----------

w_lsoa = Window.partitionBy("person_id").orderBy(f.col("lsoa_date_diff_abs").asc())
df_join_lsoa = df_join_lsoa.withColumn("rank", f.row_number().over(w_lsoa))
df_join_lsoa_unique = (df_join_lsoa
                       .filter(f.col("rank") == 1)
                       .withColumnRenamed("data_source", "lsoa_data_source"))

# COMMAND ----------

display(df_join_lsoa_unique.limit(50))

# COMMAND ----------

df_join_lsoa_unique = (df_join_lsoa_unique
                       .select(["person_id",  "lsoa", "lsoa_data_source", "lsoa_date_diff"])
                      )

# COMMAND ----------

df_2_4 = df_2_3.join(df_join_lsoa_unique, on="person_id", how="left")
# Extract the first letter to keep E only (for England)
df_2_4 = df_2_4.withColumn("lsoa_1", f.substring(f.col("lsoa"), 1, 1))


# COMMAND ----------

display(df_2_4
.groupBy("lsoa_1").agg(f.count("*")))

# COMMAND ----------

display(df_2_4.limit(5))

# COMMAND ----------

display(df_2_4.filter(f.col("lsoa_date_diff")>0).limit(20))

# COMMAND ----------

count_var(df_2_4, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.5. Sex
# MAGIC
# MAGIC Add a QC clolumn for sex
# MAGIC - Where sex is not M or F
# MAGIC - Where there is a QC issues with sex-related diagnosis codes

# COMMAND ----------

# Other codes for sex
display(df_2_4.groupBy("sex").agg(f.count("*")))

# COMMAND ----------

# Sex QC
female_codes = spark.table(f'{proj_params.dbc}.kdsc_sex_qc_female_specific_all')
count_var(female_codes, "person_id")


# COMMAND ----------

male_codes = spark.table(f'{proj_params.dbc}.kdsc_sex_qc_male_specific_all')
count_var(male_codes, "person_id")

# COMMAND ----------

df_2_5 = df_2_4.join(male_codes, on="person_id", how="left").fillna({"male_specific": 0})
df_2_5 = df_2_5.join(female_codes, on="person_id", how="left").fillna({"female_specific": 0})

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Save

# COMMAND ----------

# spark.sql(f'drop table if exists {proj_params.dbc}.ccu105_minap_cohort_part_1')

# COMMAND ----------

# df_2_5.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_minap_cohort_part_1')
save_table(df_2_5, out_name=f'ccu105_minap_cohort_part_1', save_previous=True)

# COMMAND ----------

df_2_5_reload = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_1')

# COMMAND ----------

df_2_5_reload.columns

# COMMAND ----------

display(df_2_5_reload.groupBy(["sex", "female_specific", "male_specific"]).agg(f.count("*")))

# COMMAND ----------

count_var(df_2_5_reload, "person_id")

# COMMAND ----------

display(df_2_5_reload)

# COMMAND ----------

