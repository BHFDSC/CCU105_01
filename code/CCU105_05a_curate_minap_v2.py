# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_05a_curate_minap_v2
# MAGIC
# MAGIC
# MAGIC **Description** Add records from 2017 from the previous Archived_on date
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**
# MAGIC Update as of 15/01/2026. This v2 notebook largely uses the previous MINAP batch (archived_on == 2023-12-27) to develop cohort (study period: 2017-2023), supplemented with ~9 months of data in 2023 from the most recent MINAP batch (archived_on == 2025-04-24). This is because a lot of the MINAP variables (in particular the discharge medication variables) in the most recent batch are 100% missing. This is currently being investigated by the NHS data wrangler team.
# MAGIC **Data Output**
# MAGIC - **``** :  

# COMMAND ----------

# MAGIC %md
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
# MAGIC # 2. Load MINAPs

# COMMAND ----------

# check if minap is filtered for the correct archived_on_date
# Note: There should be only one row in the outputs and the date must be checked against the archived_on date in the protocol
display(minap.groupBy("archived_on").agg(f.count("*")))

# COMMAND ----------

display(minap_old.groupBy("archived_on").agg(f.count("*")))

# COMMAND ----------

minap = (minap
         .withColumnRenamed("NHS_NUMBER_DEID", "person_id")
         .drop("ProductionDate").drop("BatchId")
         )
minap_old = (minap_old
         .withColumnRenamed("NHS_NUMBER_DEID", "person_id")
         .drop("ProductionDate").drop("BatchId")
         )

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

count_var(minap, "person_id")

# COMMAND ----------

count_var(minap_old, "person_id")

# COMMAND ----------


# Check mi index dates
minap = minap.withColumn("year", f.year(f.col("ARRIVAL_AT_HOSPITAL"))).withColumn("month", f.month(f.col("ARRIVAL_AT_HOSPITAL"))).withColumn("year_month", f.concat(f.col("year"), f.lit("_"), f.col("month")))
display(minap.groupBy("year").agg(f.count("*")).orderBy(f.col("year").asc()))

# COMMAND ----------


# Check mi index dates
minap_old = minap_old.withColumn("year", f.year(f.col("ARRIVAL_AT_HOSPITAL"))).withColumn("month", f.month(f.col("ARRIVAL_AT_HOSPITAL"))).withColumn("year_month", f.concat(f.col("year"), f.lit("_"), f.col("month")))
display(minap_old.groupBy("year").agg(f.count("*")).orderBy(f.col("year").asc()))
         

# COMMAND ----------

minap_old.dtypes

# COMMAND ----------

# check whether IDs in 2017 in the older minap appear in subsequent years in the most recent minap
minap_2017 = minap_old.filter(f.col("year")==2017)
id_2017 = minap_2017.select("person_id").distinct()
id_later = minap.select("person_id", "year").distinct()
id_overlap = id_2017.join(id_later, on= "person_id", how="inner")



# COMMAND ----------

count_overlap = (
    id_overlap.select(f.countDistinct("person_id"))
)
display(count_overlap)

# COMMAND ----------

# patients in 2017 that have a record in other years
counts_by_year = (
    id_overlap.groupBy("year")
.agg(f.countDistinct("person_id"))
.orderBy("year")
)
display(counts_by_year)

# COMMAND ----------

per_patient_years = (
    id_overlap.groupBy("person_id")
    .agg(f.collect_set("year").alias("years_later"))
    .withColumn("n_years_later", f.size(f.col("years_later")))
    .orderBy(f.col('n_years_later').desc())
)
display(per_patient_years)

# COMMAND ----------

# Check how far minap_old extends to
display(minap_old.select(f.max(f.col('ARRIVAL_AT_HOSPITAL')), f.min(f.col('ARRIVAL_AT_HOSPITAL'))))

# COMMAND ----------

# Check when coverage drops: ~ March 2023
minap_old_coverage = (
    minap_old
    .groupBy('year_month')
    .agg(f.count(f.lit(1)))
)

display(minap_old_coverage)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Combine 2017 - early 2023 records (in minap_old) with remaining months of 2023 in minap  

# COMMAND ----------

# Check null values
count_var(minap_old, "ARRIVAL_AT_HOSPITAL")
count_var(minap_old, "person_id")

count_var(minap, "ARRIVAL_AT_HOSPITAL")
count_var(minap, "person_id")

# COMMAND ----------

# Remove null person_id fields
minap_old_no_null_id = (
    minap_old
    .filter(f.col('person_id').isNotNull())
)

minap_no_null_id = (
    minap
    .filter(f.col('person_id').isNotNull())
)

# COMMAND ----------

# Check for duplicates
dupes_minap_old = (
    minap_old_no_null_id
    .groupBy(minap_old_no_null_id.columns)
    .count()
    .filter("count > 1")
)

dupes_minap = (
    minap_no_null_id
    .groupBy(minap_no_null_id.columns)
    .count()
    .filter("count > 1")
)

display(dupes_minap_old)
display(dupes_minap)

# COMMAND ----------

# Remove duplicates
minap_old_no_null_id_no_dupe = (
    minap_old_no_null_id
    .dropDuplicates()
)

minap_no_null_id_no_dupe = (
    minap_no_null_id
    .dropDuplicates()
)

# check
display(minap_no_null_id_no_dupe.groupBy(minap_no_null_id_no_dupe.columns).count().filter("count > 1"))
display(minap_old_no_null_id_no_dupe.groupBy(minap_old_no_null_id_no_dupe.columns).count().filter("count > 1"))

# COMMAND ----------

# Filter minap_old from 2017 to early 2023
minap_old_filtered = (
    minap_old_no_null_id_no_dupe
    .filter(~f.col('year_month').isin(['2023_4', '2023_5', '2023_6', '2023_7', '2023_8', '2023_9', '2023_10', '2023_11', '2023_12']))
    .withColumn('minap_source', f.lit('minap_old'))
)

tab(minap_old_filtered, 'year_month')

# COMMAND ----------

# Filter most recent minap (April 2023 to Dec 2023)
minap_filtered = (
    minap_no_null_id_no_dupe
    .filter(f.col('year_month').isin(['2023_4', '2023_5', '2023_6', '2023_7', '2023_8', '2023_9', '2023_10', '2023_11', '2023_12']))
    .withColumn('minap_source', f.lit('minap'))
)

tab(minap_filtered, 'year_month')

# COMMAND ----------

curated_minap = minap_old_filtered.union(minap_filtered)

# COMMAND ----------

display(curated_minap.groupBy("year").agg(f.count("*")).orderBy(f.col("year").asc()))

# COMMAND ----------

display(minap_old_filtered.groupBy("year").agg(f.count("*")).orderBy(f.col("year").asc()))

# COMMAND ----------

display(minap_filtered.groupBy("year_month").agg(f.count("*")).orderBy(f.col("year_month").asc()))

# COMMAND ----------

count_var(curated_minap, "person_id")

# COMMAND ----------

curated_minap.columns

# COMMAND ----------

tab(curated_minap, 'minap_source')

# COMMAND ----------

# check for any duplicated rows
# curated_minap_dupe = curated_minap.drop('minap_source')

_dup = (
    curated_minap
    .groupBy(curated_minap.columns)
    .count()
    .filter("count > 1")
)
display(_dup)

# COMMAND ----------

display(curated_minap.filter(f.col('person_id') == '').groupBy(curated_minap.columns).count()) #.filter("count > 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Save

# COMMAND ----------

# spark.sql(f'drop table if exists {proj_params.dbc}.ccu105_minap_curated')

# COMMAND ----------

save_table(curated_minap, out_name = f'ccu105_minap_curated', save_previous=True)

# COMMAND ----------

display(minap.filter(f.col('person_id') == ''))

# COMMAND ----------

curated_minap = spark.table(f'{proj_params.dbc}.ccu105_minap_curated')

# COMMAND ----------

 curated_minap.columns

# COMMAND ----------

