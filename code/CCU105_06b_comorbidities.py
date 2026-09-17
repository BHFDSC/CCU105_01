# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_06b_comorbidities
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of comorbidities from DOB to MI index date. This will be joined with the cohort in CCU105_05c
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

cohort_ids = df_part_3.select(["person_id", "date_of_birth", "mi_index_date"])


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Load extracted phenotypes
# MAGIC
# MAGIC
# MAGIC Keep the first event date in GDPPR or HES APC from DOB to mi_index_date

# COMMAND ----------

gdppr_pheno= spark.table(f'{proj_params.dbc}.ccu105_phenotypes_gdppr_all_events')

# COMMAND ----------

display(gdppr_pheno.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------



# COMMAND ----------

hes_apc_pheno= spark.table(f'{proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

display(hes_apc_pheno.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Keep diabetes, hypertension, and obesity
# MAGIC
# MAGIC

# COMMAND ----------

pheno_list = ["diabetes", "hypertension", "obesity"]

# COMMAND ----------

gdppr_3_1 = gdppr_pheno.filter(f.col("phenotype").isin(pheno_list))
display(gdppr_3_1.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

hes_apc_3_1 = hes_apc_pheno.filter(f.col("phenotype").isin(pheno_list))
display(hes_apc_3_1.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Keep the first event date in GDPPR and HES APC

# COMMAND ----------

display(gdppr_3_1.limit(10))

# COMMAND ----------

display(hes_apc_3_1.limit(10))

# COMMAND ----------

gdppr_3_2 = gdppr_3_1.filter(f.col("phenotype_rownum")==1)
display(gdppr_3_2.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# check uniqueness
count_var(gdppr_3_2.filter(f.col("phenotype")=="hypertension"), "person_id")

# COMMAND ----------

hes_apc_3_2 = hes_apc_3_1.filter(f.col("phenotype_rownum")==1)
display(hes_apc_3_2.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# check uniqueness
count_var(hes_apc_3_2.filter(f.col("phenotype")=="hypertension"), "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3. Keep first event date perphenotypes in a long table format

# COMMAND ----------

gdppr_3_2.columns

# COMMAND ----------

hes_apc_3_2.columns

# COMMAND ----------

gdppr_3_3 = gdppr_3_2.select(["person_id", "phenotype", "event_date_curated"])
hes_apc_3_3 = hes_apc_3_2.select(["person_id", "phenotype", "event_date_curated"])

# COMMAND ----------

pheno_union= gdppr_3_3.union(hes_apc_3_3)

# COMMAND ----------

w = Window.partitionBy(["person_id", "phenotype"]).orderBy(f.col("event_date_curated").asc_nulls_last())
pheno_union= pheno_union.withColumn("rownum", f.row_number().over(w))


# COMMAND ----------

display(pheno_union.limit(20))

# COMMAND ----------

pheno_3_3 = pheno_union.filter(f.col("rownum")==1).drop("rownum")

# COMMAND ----------

display(pheno_3_3.groupBy("phenotype").agg(f.count("*")))

# COMMAND ----------

count_var(pheno_3_3.filter(f.col("phenotype")=="hypertension"), "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4. Keep event dates >=DOB and < MI index date

# COMMAND ----------

display(cohort_ids.limit(10))

# COMMAND ----------

count_var(cohort_ids, "person_id")

# COMMAND ----------

df_pheno_cohort = cohort_ids.join(pheno_3_3, on= "person_id", how="left")

# COMMAND ----------

display(df_pheno_cohort.limit(10))

# COMMAND ----------

count_var(df_pheno_cohort, "person_id")

# COMMAND ----------

df_3_4 = (df_pheno_cohort
          .filter(
              (f.col("event_date_curated")>=f.col("date_of_birth"))
              &
              (f.col("event_date_curated")<f.col("mi_index_date")))
)

# COMMAND ----------

display(df_3_4.limit(10))

# COMMAND ----------

count_var(df_3_4, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5. Make a wide table

# COMMAND ----------

df_3_5 =df_3_4.groupBy("person_id").pivot("phenotype").agg(f.first("event_date_curated"))

# COMMAND ----------

display(df_3_5.limit(10))

# COMMAND ----------

count_var(df_3_5, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.  Save

# COMMAND ----------

# df_3_5.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_comorbidities')
save_table(df_3_5, out_name=f'ccu105_comorbidities', save_previous=True)

# COMMAND ----------

