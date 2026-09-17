# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_06c_smoking
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of smoking categories from DOB to MI index date. This will be joined with the cohort in CCU105_05c
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

# MAGIC
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

cohort_ids = df_part_3.select(["person_id", "date_of_birth", "mi_index_date"])


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Load extracted phenotypes
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

gdppr_pheno= spark.table(f'{proj_params.dbc}.ccu105_phenotypes_gdppr_all_events')

# COMMAND ----------

display(gdppr_pheno.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

hes_apc_pheno= spark.table(f'{proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

display(hes_apc_pheno.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Keep smoking categories
# MAGIC
# MAGIC

# COMMAND ----------

pheno_list = ["smoking_current", "smoking_ex", "smoking_non", "smoking_unsp"]

# COMMAND ----------

gdppr_3_1 = gdppr_pheno.filter(f.col("phenotype").isin(pheno_list))
display(gdppr_3_1.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

hes_apc_3_1 = hes_apc_pheno.filter(f.col("phenotype").isin(pheno_list))
display(hes_apc_3_1.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Union GDPPR and HES APC 

# COMMAND ----------

display(gdppr_3_1.limit(10))

# COMMAND ----------

display(hes_apc_3_1.limit(10))

# COMMAND ----------

gdppr_3_1.columns

# COMMAND ----------

hes_apc_3_1.columns

# COMMAND ----------

df_union = (
    gdppr_3_1.select(["person_id", "phenotype", "event_date_curated", 'phenotype_source'])
    .union(
        hes_apc_3_1.select(["person_id", "phenotype", "event_date_curated", 'phenotype_source'])
    )

)

# COMMAND ----------

# Deduplicate event dates per phenotype (prioritise dates in gdppr)
w_date = (
  Window
  .partitionBy("person_id","phenotype",  "event_date_curated")
  .orderBy(f.col("phenotype_source").asc())
          )
df_3_4 = df_union.withColumn("rownum", f.row_number().over(w_date)).filter(f.col("rownum")==1).drop("rownum")

# COMMAND ----------


display(df_union.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

display(df_3_4.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3.5. Join with cohort_ids,  add time endpots, limit dates to DOB and mi_index_date

# COMMAND ----------

print(cohort_ids.columns)
print(df_3_4.columns)

# COMMAND ----------

df_3_5_pre = cohort_ids.join(df_3_4, on= "person_id", how="left")


# COMMAND ----------

count_var(df_3_5_pre, "person_id")

# COMMAND ----------

df_3_5 = (
    df_3_5_pre
    .filter(
        (f.col("event_date_curated")>=f.col("date_of_birth"))
        &
        (f.col("event_date_curated")<f.col("mi_index_date"))
            )
    )

# COMMAND ----------

count_var(df_3_5, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Note: after joining with the cohort in CCU105_05c, fill NA with missing category specified below. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6. Prepare smoking event for the rules for smoking status
# MAGIC
# MAGIC "Missing smoking status: no record of any codes related to smoking status exist for that individual (at any time point up until date of MI)
# MAGIC
# MAGIC Never smoker: the individual has one or more codes from the 'non-smoking' codelist without any ex-/current smoking codes recorded until the date of MI.
# MAGIC
# MAGIC Ex-smoker: the most recent smoking-related code recorded for the individual prior to MI was from the 'ex-smoking' codelist, or was from the 'non-smoking' codelist, with a 'current smoking' code used prior to the 'non-smoking code'
# MAGIC
# MAGIC Current smoker: the most recent smoking-related code used for an individual prior to the date of MI was from the 'current smoking' codelist"

# COMMAND ----------

df_3_5.columns

# COMMAND ----------

display(df_3_5.orderBy("person_id"))

# COMMAND ----------

# Find the earlierst and most recent evet dates in each category
df_wide_first=df_3_5.groupBy("person_id").pivot("phenotype").agg(f.min("event_date_curated"))
df_wide_last=df_3_5.groupBy("person_id").pivot("phenotype").agg(f.max("event_date_curated"))

# COMMAND ----------

list_first = list(df_wide_first.columns)
print(list_first)

# COMMAND ----------

list_last = list(df_wide_last.columns)
print(list_last)

# COMMAND ----------

for item in list_first[1:]:
  df_wide_first = df_wide_first.withColumnRenamed(item, f'{item}_first')
  df_wide_last = df_wide_last.withColumnRenamed(item, f'{item}_last')

# COMMAND ----------

print(df_wide_first.columns)
print(df_wide_last.columns)


# COMMAND ----------

display(df_wide_first.orderBy("person_id"))

# COMMAND ----------

display(df_wide_last.orderBy("person_id"))

# COMMAND ----------

df_3_6 = cohort_ids.join(df_wide_first, on="person_id", how="left").join(df_wide_last, on="person_id", how="left")

# COMMAND ----------



# COMMAND ----------

print(df_3_6.count())
print(df_3_6.select("person_id").distinct().count())

# COMMAND ----------

# DBTITLE 1,_var
# MAGIC %md
# MAGIC ## 3.7. Apply smoking rules
# MAGIC
# MAGIC "Missing smoking status: no record of any codes related to smoking status exist for that individual (at any time point up until date of MI)
# MAGIC
# MAGIC Never smoker: the individual has one or more codes from the 'non-smoking' codelist without any ex-/current smoking codes recorded until the date of MI.
# MAGIC
# MAGIC Ex-smoker: the most recent smoking-related code recorded for the individual prior to MI was from the 'ex-smoking' codelist, or was from the 'non-smoking' codelist, with a 'current smoking' code used prior to the 'non-smoking code'
# MAGIC
# MAGIC Current smoker: the most recent smoking-related code used for an individual prior to the date of MI was from the 'current smoking' codelist"
# MAGIC
# MAGIC

# COMMAND ----------

df_3_6.columns

# COMMAND ----------

display(df_3_6.limit(200))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7.1. Missing smoking status
# MAGIC
# MAGIC "Missing smoking status: no record of any codes related to smoking status exist for that individual (at any time point up until date of MI)
# MAGIC

# COMMAND ----------

df_3_7_1 = (
    df_3_6.withColumn("smoking_missing", 
                      f.when(
                          (f.col("smoking_current_first").isNull()) &
                          (f.col("smoking_current_last").isNull()) &
                          (f.col("smoking_ex_first").isNull()) &
                          (f.col("smoking_ex_last").isNull()) &
                          (f.col("smoking_non_first").isNull()) &
                          (f.col("smoking_non_last").isNull()) &
                          (f.col("smoking_unsp_first").isNull()) &
                          (f.col("smoking_unsp_last").isNull())
                            , f.lit(1)
                      ).otherwise(f.lit(0))
                    
                      )
    )


# COMMAND ----------

display(df_3_7_1.groupBy("smoking_missing").agg(f.count("*")))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7.2
# MAGIC
# MAGIC Never smoker: the individual has one or more codes from the 'non-smoking' codelist without any ex-/current smoking codes recorded until the date of MI.
# MAGIC

# COMMAND ----------

df_3_7_2 = (
    df_3_7_1.withColumn("smoking_never", 
                      f.when(((f.col("smoking_non_first").isNotNull())
                              |
                              (f.col("smoking_non_last").isNotNull())) 
                             & 
                             ((f.col("smoking_current_first").isNull())&
                              (f.col("smoking_current_last").isNull())&
                              (f.col("smoking_ex_first").isNull())&
                              (f.col("smoking_ex_last").isNull())
                             )
                                                  
                            , f.lit(1)
                      ).otherwise(f.lit(0))
                    
                      )
    )
    
 

# COMMAND ----------

display(df_3_7_2.groupBy("smoking_missing", "smoking_never").agg(f.count("*")))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7.3. Ex smoker
# MAGIC
# MAGIC Ex-smoker: the most recent smoking-related code recorded for the individual prior to MI was from the 'ex-smoking' codelist, or was from the 'non-smoking' codelist, with a 'current smoking' code used prior to the 'non-smoking code'
# MAGIC

# COMMAND ----------

df_3_7_2.dtypes

# COMMAND ----------

df_3_7_3 = (
    df_3_7_2
    .withColumn(
        "smoking_ex",
        f.when(
            # most recent code is ex-smoker
            (
                f.col("smoking_ex_last").isNotNull() &
                (f.col("smoking_current_last").isNull() | (f.col("smoking_ex_last") > f.col("smoking_current_last"))) &
                (f.col("smoking_non_last").isNull()     | (f.col("smoking_ex_last") > f.col("smoking_non_last"))) &
                (f.col("smoking_unsp_last").isNull()    | (f.col("smoking_ex_last") > f.col("smoking_unsp_last")))
            )
            |
            # most recent code is non-smoker, with past current 
            (
                f.col("smoking_non_last").isNotNull() &
                f.col("smoking_current_first").isNotNull() &
                (f.col("smoking_current_first") < f.col("smoking_non_last")) &
                (f.col("smoking_ex_last").isNull()      | (f.col("smoking_non_last") > f.col("smoking_ex_last"))) &
                (f.col("smoking_current_last").isNull() | (f.col("smoking_non_last") > f.col("smoking_current_last"))) &
                (f.col("smoking_unsp_last").isNull()    | (f.col("smoking_non_last") > f.col("smoking_unsp_last")))
            ),
            f.lit(1)
        ).otherwise(f.lit(0))
    )
)

# COMMAND ----------

display(df_3_7_3.groupBy("smoking_missing", "smoking_never", "smoking_ex").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7.4. Current smoker (and unspecified)
# MAGIC Current smoker: the most recent smoking-related code used for an individual prior to the date of MI was from the 'current smoking' codelist"
# MAGIC
# MAGIC Apply the same logic for unspecified
# MAGIC

# COMMAND ----------

df_3_7_3.dtypes

# COMMAND ----------

df_3_7_4 = (
    df_3_7_3
    .withColumn(
        "smoking_current",
        f.when(
            f.col("smoking_current_last").isNotNull() &
            (f.col("smoking_ex_last").isNull()  | (f.col("smoking_current_last") > f.col("smoking_ex_last"))) &
            (f.col("smoking_non_last").isNull() | (f.col("smoking_current_last") > f.col("smoking_non_last"))) &
            (f.col("smoking_unsp_last").isNull() | (f.col("smoking_current_last") > f.col("smoking_unsp_last"))),
            f.lit(1)
        ).otherwise(f.lit(0))
    )
)

df_3_7_4 = (
    df_3_7_4
    .withColumn(
        "smoking_unspecified",
        f.when(
            f.col("smoking_unsp_last").isNotNull() &
            (f.col("smoking_ex_last").isNull()  | (f.col("smoking_unsp_last") > f.col("smoking_ex_last"))) &
            (f.col("smoking_non_last").isNull() | (f.col("smoking_unsp_last") > f.col("smoking_non_last"))) &
            (f.col("smoking_current_last").isNull() | (f.col("smoking_unsp_last") > f.col("smoking_current_last"))),
            f.lit(1)
        ).otherwise(f.lit(0))
    )
)

# COMMAND ----------


display(df_3_7_4.groupBy("smoking_missing", "smoking_never", "smoking_ex", "smoking_current", "smoking_unspecified").agg(f.count("*")))

# COMMAND ----------

df_3_7_4.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.  Save

# COMMAND ----------

df_smoking_out = df_3_7_4.select(["person_id", "smoking_missing", "smoking_never", "smoking_ex", "smoking_current", "smoking_unspecified"])

# COMMAND ----------

# df_smoking_out.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_smoking')
save_table(df_smoking_out, out_name=f'ccu105_smoking', save_previous=True)

# COMMAND ----------

