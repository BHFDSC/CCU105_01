# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_04e_extract_phenotypes_events_krt_hes_apc_procedures
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all procedure phenotype events from HES APC 
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC **Reviewers** DY & GC
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
# MAGIC # 2. Load clean HES APC Procedures and codelists

# COMMAND ----------

proj_params.path_hes_apc_procedure_preprocessed

# COMMAND ----------

hes_apc_proc_clean = spark.table(f'{proj_params.path_hes_apc_procedure_preprocessed}')

# COMMAND ----------

count_var(hes_apc_proc_clean, "person_id")

# COMMAND ----------

codelist_krt_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/CCU105/codelist_krt.csv'

codelist_krt = pd.read_csv(codelist_krt_path, keep_default_na=False)
codelist_krt = spark.createDataFrame(codelist_krt)

# COMMAND ----------

display(codelist_krt.limit(20))

# COMMAND ----------

# Keep  OPCS4 codes
proc_codelist = codelist_krt.filter(f.col("terminology")=="opcs4").withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
display(proc_codelist.groupBy("phenotype", "terminology").agg(f.count("*")).orderBy(f.col("phenotype")))

# COMMAND ----------

print(proc_codelist.filter(f.col("code").contains(".")).count())

# COMMAND ----------

proc_codelist.columns

# COMMAND ----------

display(proc_codelist.limit(5))

# COMMAND ----------



display(hes_apc_proc_clean.limit(5))

# COMMAND ----------

hes_apc_proc_clean = hes_apc_proc_clean.withColumn("code", f.lower(f.col("code"))).withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))

# COMMAND ----------

display(hes_apc_proc_clean.limit(5))

# COMMAND ----------

proc_codelist = proc_codelist.drop("terminology").drop("description")

# COMMAND ----------

# All  events in HES APC
hes_apc_proc_pheno = hes_apc_proc_clean.join(f.broadcast(proc_codelist), on=["code"], how="inner")



# COMMAND ----------

count_var(hes_apc_proc_pheno, "person_id")

# COMMAND ----------

hes_apc_proc_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe
hes_apc_pheno_proc_sel = hes_apc_proc_pheno.select(
    [
        "person_id",
        "phenotype",
        "subphenotype",
        #"subphenotype_text",
        #"phenotype",
        "code",
        "event_date_curated",
 'procedure_date',
 'epikey',
 'epistart',
 'epiend',
 'admidate',
 'disdate',
 'code_digits',
 'position',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD'
    ]
).withColumn("phenotype_source", f.lit("HES_APC_PROC"))

# COMMAND ----------

display(hes_apc_pheno_proc_sel.groupBy("phenotype").agg(f.count("*")))

# COMMAND ----------

display(hes_apc_pheno_proc_sel.groupBy("phenotype", "subphenotype").agg(f.count("*")))

# COMMAND ----------

# Mark the order of event dates and save

w = Window.partitionBy("person_id", "phenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
hes_apc_pheno_proc_sel = (hes_apc_pheno_proc_sel
                        .withColumn("combined_pheno_rownum", f.row_number().over(w))
                        .withColumn("combined_pheno_rank", f.rank().over(w))
                        .withColumn("combined_pheno_dense_rank", f.dense_rank().over(w))
               

                        )


# w = Window.partitionBy("person_id", "phenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
# w_sub = Window.partitionBy("person_id", "phenotype", "subphenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
# hes_apc_pheno_proc_sel = (hes_apc_pheno_proc_sel
 #                        .withColumn("phenotype_rownum", f.row_number().over(w))
 #                        .withColumn("phenotype_rank", f.rank().over(w))
#                         .withColumn("phenotype_dense_rank", f.dense_rank().over(w))
#                         .withColumn("subphenotype_rownum", f.row_number().over(w_sub))
#                         .withColumn("subphenotype_rank", f.rank().over(w_sub))
#                         .withColumn("subphenotype_dense_rank", f.dense_rank().over(w_sub))
# 
#                         )


# COMMAND ----------

hes_apc_pheno_proc_sel.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_phenotypes_krt_hes_apc_proc_all_events')

# COMMAND ----------

hes_apc_pheno_proc_sel_relaod = spark.table(f'{proj_params.dbc}.ccu105_phenotypes_krt_hes_apc_proc_all_events')

# COMMAND ----------

display(hes_apc_pheno_proc_sel_relaod.limit(20))

# COMMAND ----------

display(hes_apc_pheno_proc_sel_relaod.groupBy("phenotype", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

display(hes_apc_pheno_proc_sel_relaod.groupBy("phenotype", "subphenotype_text").agg(f.count("*")))

# COMMAND ----------

