# Databricks notebook source
# MAGIC %md 
# MAGIC # CCU105_03a_codelist_comorbidities
# MAGIC
# MAGIC **Project** CCU105
# MAGIC
# MAGIC **Description** This notebook creates the codelist uploaded by the CCU105 team.
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Data Output**
# MAGIC - **``** : 

# COMMAND ----------

# MAGIC %md # 0. Setup

# COMMAND ----------

# DBTITLE 1,Libraries
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

# DBTITLE 1,Common functions
# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md # 1. Parameters

# COMMAND ----------

# DBTITLE 1,Common functions
# MAGIC %run "./CCU105_01_parameters"

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. Project Codelists
# MAGIC

# COMMAND ----------

list_table = spark.sql(f'''show tables in  {proj_params.dbc}''')
ccu105_tables = list_table.filter(f.col("tableName").startswith("ccu105"))
display(ccu105_tables)

# COMMAND ----------

ccu105_codelists = ['ccu105_diabetes_codelist_v1_20250612',
 'ccu105_hypertension_codelist_v1_20250612',
 'ccu105_mi_codelist_v1_20250612',
 'ccu105_obesity_codelist_v1_20250612',
 'ccu105_smoking_current_codelist_v1_20250612',
 'ccu105_smoking_ex_codelist_v1_20250612',
 'ccu105_smoking_non_codelist_v1_20250612',
 'ccu105_smoking_unsp_codelist_v1_20250612']

# COMMAND ----------

# Check diabetes codes for QOF codes
diab_temp = spark.table(f'{proj_params.dbc}.ccu105_diabetes_codelist_v1_20250612')
display(diab_temp.limit(2))

# COMMAND ----------

display(diab_temp.filter(f.lower(f.col("description")).contains('quality')))

# COMMAND ----------

diab_no_qof = diab_temp.filter(~f.col("code").isin(["1110921000000100","143401000000102"])).drop("name").withColumn("name", f.lit("diabetes_no_qof"))
display(diab_no_qof.filter(f.lower(f.col("description")).contains('quality')))

# COMMAND ----------

print(diab_temp.count())
print(diab_no_qof.count())


# COMMAND ----------




# COMMAND ----------

 # Check tables
 
for item in ccu105_codelists:
    spark.table(f'''{proj_params.dbc}.{item}''').limit(3).show()

# COMMAND ----------

# Check phenotype names  (Any sub_phenotypes?)

for item in ccu105_codelists:
    print(f'''Table = {item}''')
    df = spark.table(f'''{proj_params.dbc}.{item}''')
    unique_phenos = [row['name'] for row in df.select('name').distinct().collect()]
    print(unique_phenos)
    print("\n")
    

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Union of codelists

# COMMAND ----------

# DBTITLE 1,Declare Prostate Cancer SNOMED codes
codelist_union = spark.table(f'''{proj_params.dbc}.{ccu105_codelists[0]}''')
for item in ccu105_codelists[1:]:
    codelist_union = codelist_union.union(spark.table(f'''{proj_params.dbc}.{item}'''))
codelist_union= codelist_union.union(diab_no_qof)



# COMMAND ----------

display(codelist_union)

# COMMAND ----------

display(codelist_union.groupBy(["code_type", "name"]).agg(f.count("*")))

# COMMAND ----------

# Check overlapping smoking codes
code_smoke = codelist_union.filter(f.col("name").startswith("smoking"))
display(code_smoke.groupBy(["code_type", "name"]).agg(f.count("*")))

# COMMAND ----------

display(code_smoke.filter(f.col("name")=="smoking_unsp"))

# COMMAND ----------

display(code_smoke.filter(f.col("name")=="smoking_non"))

# COMMAND ----------

display(code_smoke.filter(f.col("name")=="smoking_ex"))

# COMMAND ----------

display(code_smoke.filter(f.col("name")=="smoking_current"))

# COMMAND ----------

# Drop generic QOF code 1109921000000106

smoke_current_no_qof = code_smoke.filter(f.col("name")=="smoking_current").filter(f.col("code")!="1109921000000106").drop("name").withColumn("name", f.lit("smoking_current_no_qof_1"))
print(code_smoke.filter(f.col("name")=="smoking_current").count())
print(smoke_current_no_qof.count())


# COMMAND ----------

codelist_union = codelist_union.union(smoke_current_no_qof)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Quality check

# COMMAND ----------

# DBTITLE 1,Declare Pregnancy/Birth SNOMED codes
# check 
tmpt = tab(codelist_union, 'name', 'code_type')

# COMMAND ----------

codelist_union =(codelist_union.withColumnRenamed('name', 'phenotype')
                 .withColumn('terminology', f.when(f.col("code_type")=="ICD-10", f.lit("icd10")).otherwise(f.lit("snomed"))).drop("code_type"))

# COMMAND ----------

# check 
tmpt = tab(codelist_union, 'phenotype', 'terminology')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1. Check SNOMED code duplication

# COMMAND ----------

# Are snomed codes unique?
snomed_grouped = codelist_union.filter(f.col("terminology") == "snomed").groupBy("phenotype", "code").agg(f.count("*").alias("count")).orderBy(f.col("count").desc())
display(snomed_grouped)

# COMMAND ----------

duplicated_codes = snomed_grouped.filter(f.col("count") > 1).select("code").collect()
duplicated_codes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Check ICD10 code duplication

# COMMAND ----------

# Are ICD10 codes unique?
icd_grouped = codelist_union.filter(f.col("terminology") == "icd10").groupBy("phenotype", "code").agg(f.count("*").alias("count")).orderBy(f.col("count").desc())
display(icd_grouped)

# COMMAND ----------

duplicated_codes = icd_grouped.filter(f.col("count") > 1).select("code").collect()
duplicated_codes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3. ICD-10 code format

# COMMAND ----------

# any codes with a lengh not = 3 or 4
display(codelist_union.filter(f.col("terminology") == "icd10").filter(~f.length(f.col("code")).isin([3, 4])))

# COMMAND ----------

# any codes with .
display(codelist_union.filter(f.col("terminology") == "icd10").filter(f.col("code").contains(".")))

# COMMAND ----------

# MAGIC %md ## 4 How many of the SNOMED codes exist in the SNOMED refset 

# COMMAND ----------

print(proj_params.path_gdppr_refset)
df_refset = spark.table(proj_params.path_gdppr_refset)

# COMMAND ----------

# DBTITLE 1,Declare COCP BNF codes
df_refset_2 = df_refset.withColumnRenamed("ConceptId", "CODE").withColumn("isin_refset", f.lit(1))
df_snomed = codelist_union.filter(f.col("terminology")=="snomed").withColumn("isin_snomed_codelist", f.lit(1))
df_refset_join = df_snomed.join(df_refset_2, on= "CODE", how="left").fillna({"isin_snomed_codelist":0, "isin_refset":0})
            
display(df_refset_join.groupby("isin_snomed_codelist", "isin_refset").agg(f.count("*")))

# COMMAND ----------

display(df_refset_join.filter(f.col("isin_refset")==0))

# COMMAND ----------

# MAGIC %md # 5. Save

# COMMAND ----------

spark.sql(f'drop table if exists {proj_params.dbc}.ccu105_codelist_union_20250918')

# COMMAND ----------

codelist_union.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_codelist_union_20250918')

# COMMAND ----------

