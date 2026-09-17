# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC # CCU105_05c_cohort_inclusion_exclusion_part_3
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the base cohort from MINAP and applying incluison and exclusion criteria, final step
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
# MAGIC # 2. Load the dataset saved in part 3

# COMMAND ----------

cohort = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_3')

# COMMAND ----------

display(cohort.limit(10))

# COMMAND ----------

count_var(cohort, "person_id")

# COMMAND ----------

# First limit these datses to patient IDs in MINAP Cphort
from pyspark.sql.functions import broadcast
mi_ids = cohort.select(["person_id", "mi_index_date"])  
broadcast_ids = broadcast(mi_ids)

# COMMAND ----------

gdppr_krt = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_krt_gdppr_all_events')
hes_apc_krt = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_krt_hes_apc_all_events')
hes_apc_proc_krt = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_krt_hes_apc_proc_all_events')

# COMMAND ----------

gdppr_krt_sel = gdppr_krt.select(["person_id", "phenotype", "subphenotype", "event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")
hes_apc_krt_sel = hes_apc_krt.select(["person_id", "phenotype", "subphenotype", "event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")
hes_apc_proc_krt_sel = hes_apc_proc_krt.select(["person_id", "phenotype", "subphenotype","event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")

# We need first event event dates across all datasets
union_krt = gdppr_krt_sel.union(hes_apc_krt_sel).union(hes_apc_proc_krt_sel)

# COMMAND ----------

# Add study start date and rename phenotypes based on pre- (inclusive) and post-
union_krt = (union_krt
             .withColumn("study_start_date", f.to_date(f.lit(proj_params.study_start_date)))
             .withColumn("mi_period_end_date", f.to_date(f.lit(proj_params.mi_period_end_date)))
             .withColumn("study_end_date", f.to_date(f.lit(proj_params.study_end_date)))
             )
display(union_krt.limit(10))
                                

# COMMAND ----------

union_krt_2 = (union_krt
               .withColumn('phenotype_name', f.col('phenotype'))
            .withColumn ("phenotype_name", 
                         f.when(f.col("event_date_curated")<=f.col("mi_index_date"), 
                                f.concat(f.col("phenotype_name"), f.lit("_"), f.col("subphenotype"), f.lit("_previous")))
                         .when((f.col("event_date_curated")> f.col("mi_index_date")) & (f.col("event_date_curated")<=f.col("study_end_date")), f.concat(f.col("phenotype_name"), f.lit("_"), f.col("subphenotype"), f.lit("_outcome")))
                                  .otherwise(f.concat(f.col("phenotype_name"), f.lit("_"), f.col("subphenotype"), f.lit("_out_of_range"))))
            )

# COMMAND ----------

display(union_krt_2.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

# Drop out of range phenotyeps
union_krt_3 = union_krt_2.filter(~f.col("phenotype_name").endswith("_out_of_range"))
display(union_krt_3.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

# We need first event event dates across all datasets
w = Window.partitionBy("person_id", "phenotype_name").orderBy(f.col("event_date_curated").asc_nulls_last())
union_krt_long = union_krt_3.withColumn("rank", f.row_number().over(w)).filter(f.col("rank")==1  ).drop("rank")
display(union_krt_long.limit(5))

# COMMAND ----------

# Long to wide
krt_wide =union_krt_long.groupBy("person_id").pivot("phenotype_name").agg(f.first("event_date_curated"))

# COMMAND ----------

# Join back to cohort
cohort_final = (
    cohort
    .join(krt_wide, 'person_id', how='left')
)

display(cohort_final)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Check

# COMMAND ----------

count_var(cohort, 'person_id')
count_var(cohort_final, 'person_id')
count_var(cohort_final, 'renal_replacement_therapy_dialysis_outcome')
count_var(cohort_final, 'renal_replacement_therapy_transplant_outcome')
count_var(cohort_final, 'dialysis_outcome')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 4. Save

# COMMAND ----------

save_table(cohort_final, out_name=f'ccu105_minap_cohort_part_4', save_previous=True)

# COMMAND ----------

df_3_6_reload = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_4')

# COMMAND ----------

display(df_3_6_reload.limit(10))

# COMMAND ----------

