# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_04a_extract_phenotype_events_gdppr
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all diagnostic phenotype events from GDPPR 
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

import os

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
# MAGIC # 2. Load clean GDPPR and codelists

# COMMAND ----------

proj_params.path_gdppr_preprocessed

# COMMAND ----------

gdppr_clean = spark.table(f'{proj_params.path_gdppr_preprocessed}')

# COMMAND ----------

count_var(gdppr_clean, "person_id")

# COMMAND ----------

codelist_krt_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/CCU105/codelist_krt.csv'

codelist_krt = pd.read_csv(codelist_krt_path, keep_default_na=False)
codelist_krt = spark.createDataFrame(codelist_krt)

# COMMAND ----------

display(codelist_krt.limit(1))

# COMMAND ----------

# check 
tmpt = tab(codelist_krt, 'phenotype', 'terminology')

# COMMAND ----------

# Keep diagnostic SNOMED codes
diag_codelist = codelist_krt.filter(f.col("terminology")=="snomed")
display(diag_codelist.groupBy("phenotype", "terminology").agg(f.count("*")).orderBy(f.col("phenotype")))

# COMMAND ----------

diag_codelist.columns

# COMMAND ----------

display(diag_codelist.limit(5))

# COMMAND ----------

display(gdppr_clean.limit(5))

# COMMAND ----------

gdppr_clean = gdppr_clean.withColumnRenamed("CODD", "code")
diag_codelist = diag_codelist.drop("terminology").drop("description")

# COMMAND ----------

# All KRT events in GDPPR
gdppr_pheno = gdppr_clean.join(f.broadcast(diag_codelist), on=["code"], how="inner")

# COMMAND ----------

count_var(gdppr_pheno, "person_id")

# COMMAND ----------

gdppr_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe

gdppr_pheno_sel = gdppr_pheno.select(
    [
        "person_id",
        "phenotype",
        "subphenotype",
        "code",
        "event_date_curated",
        "PRACTICE",
        "GP_SYSTEM_SUPPLIER",
        "PROCESSED_TIMESTAMP",
        "REPORTING_PERIOD_END_DATE",
        "DATE",
        "RECORD_DATE",
        "EPISODE_CONDITION",
        "EPISODE_PRESCRIPTION",
        "VALUE1_CONDITION",
        "VALUE2_CONDITION",
        "date_of_birth",
        "death_flag",
        "date_of_death",
        "flag_death_and_event_date",
        "date_diff_curated_evdt_DOD",
    ]
).withColumn("phenotype_source", f.lit("GDPPR"))

# COMMAND ----------

display(gdppr_pheno_sel.groupBy("phenotype").agg(f.count("*")))

# COMMAND ----------

# Mark the order of event dates and save
w = Window.partitionBy("person_id", "phenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
gdppr_pheno_sel = (gdppr_pheno_sel
                        .withColumn("phenotype_rownum", f.row_number().over(w))
                        .withColumn("phenotype_rank", f.rank().over(w))
                        .withColumn("phenotype_dense_rank", f.dense_rank().over(w))

                        )


# COMMAND ----------

save_table(gdppr_pheno_sel, f'ccu105_phenotypes_krt_gdppr_all_events', save_previous=False)

# COMMAND ----------

gdppr_pheno_sel_relaod = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_krt_gdppr_all_events')

# COMMAND ----------

display(gdppr_pheno_sel_relaod.limit(20))

# COMMAND ----------

display(gdppr_pheno_sel_relaod.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

