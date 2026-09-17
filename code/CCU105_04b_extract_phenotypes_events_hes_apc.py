# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_04b_extract_phenotypes_events_hes_apc
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all diagnostic phenotype events from HES APC 
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
# MAGIC # 2. Load clean HES APC and codelists

# COMMAND ----------

proj_params.path_hes_apc_preprocessed

# COMMAND ----------

hes_apc_clean = spark.table(f'{proj_params.path_hes_apc_preprocessed}')

# COMMAND ----------

count_var(hes_apc_clean, "person_id")

# COMMAND ----------

codelist_union = spark.table(f'{proj_params.dbc}.ccu105_codelist_union_20250918')



# COMMAND ----------

display(codelist_union.limit(1))

# COMMAND ----------

display(codelist_union.groupBy("phenotype", "terminology").agg(f.count("*")).orderBy(f.col("phenotype")))

# COMMAND ----------

# Keep diagnostic ICD10 codes
diag_codelist = codelist_union.filter(f.col("terminology")=="icd10").withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
display(diag_codelist.groupBy("phenotype", "terminology").agg(f.count("*")).orderBy(f.col("phenotype")))

# COMMAND ----------

print(diag_codelist.filter(f.col("code").contains(".")).count())

# COMMAND ----------

diag_codelist.columns

# COMMAND ----------

display(diag_codelist.limit(5))

# COMMAND ----------

display(hes_apc_clean.limit(5))

# COMMAND ----------

hes_apc_clean = hes_apc_clean.withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))

# COMMAND ----------

display(hes_apc_clean.limit(5))

# COMMAND ----------

diag_codelist = diag_codelist.drop("terminology").drop("description")

# COMMAND ----------

# All  events in HES APC
hes_apc_pheno = hes_apc_clean.join(f.broadcast(diag_codelist), on=["code"], how="inner")



# COMMAND ----------

count_var(hes_apc_pheno, "person_id")

# COMMAND ----------

hes_apc_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe

hes_apc_pheno_sel = hes_apc_pheno.select(
    [
        "person_id",
        "phenotype",
        "code",
        "event_date_curated",
 'epikey',
 'epistart',
 'epiend',
 'admidate',
 'disdate',
 'diag_column',
 'diag_digits',
 'diag_position',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD',
    ]
).withColumn("phenotype_source", f.lit("HES_APC"))

# COMMAND ----------

display(hes_apc_pheno_sel.groupBy("phenotype").agg(f.count("*")))

# COMMAND ----------

# Mark the order of event dates and save
w = Window.partitionBy("person_id", "phenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
hes_apc_pheno_sel = (hes_apc_pheno_sel
                        .withColumn("phenotype_rownum", f.row_number().over(w))
                        .withColumn("phenotype_rank", f.rank().over(w))
                        .withColumn("phenotype_dense_rank", f.dense_rank().over(w))

                        )


# COMMAND ----------

spark.sql(f'drop table {proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

hes_apc_pheno_sel.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

hes_apc_pheno_sel_relaod = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_hes_apc_all_events')

# COMMAND ----------

display(hes_apc_pheno_sel_relaod.limit(20))

# COMMAND ----------

display(hes_apc_pheno_sel_relaod.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

