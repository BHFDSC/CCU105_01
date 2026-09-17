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
# MAGIC # 2. Load the dataset saved in part 2

# COMMAND ----------

df_2 = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_2')

# COMMAND ----------

display(df_2.limit(10))

# COMMAND ----------


count_var(df_2, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Inclusion/Exclusion 

# COMMAND ----------

df_2.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1. MI definition based on diagnosis discharge 
# MAGIC
# MAGIC Inluce only category 1 and category 4

# COMMAND ----------

display(df_2.groupBy("DISCHARGE_DIAGNOSIS").agg(f.count("*")))

# COMMAND ----------

# How many not with 1. and 4. categories?
df_temp = df_2.filter(~f.col("DISCHARGE_DIAGNOSIS").rlike(r"^(1\.|4\.)"))

count_var(df_temp, "person_id")

# COMMAND ----------

# How many  with 1. and 4. categories?
df_temp = df_2.filter(f.col("DISCHARGE_DIAGNOSIS").rlike(r"^(1\.|4\.)"))

count_var(df_temp, "person_id")

# COMMAND ----------


df_3_1 = df_2.filter(f.col("DISCHARGE_DIAGNOSIS").rlike(r"^(1\.|4\.)"))

# COMMAND ----------



display(df_3_1.groupBy("DISCHARGE_DIAGNOSIS").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_1, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 First MI between 2017-01-01 and 2023-12-31
# MAGIC (inclusive)
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------

display(df_3_1.select(['study_start_date','study_end_date', 'mi_index_date','mi_period_end_date']).limit(10))

# COMMAND ----------

# Rank MI events and find the maximum number of MI events per patient
w_mi = Window.partitionBy("person_id").orderBy(f.col("mi_index_date").asc_nulls_last())
df_3_1_a = df_3_1.withColumn("mi_rank", f.row_number().over(w_mi)).withColumn("mi_count", f.max(f.col("mi_rank")).over(w_mi))

# COMMAND ----------

display(df_3_1_a.groupBy("mi_count").agg(f.count("*")))

# COMMAND ----------


# How many patients outside this period
df_temp = (df_3_1_a.filter(f.col("mi_rank")==1)
           .filter((f.col("mi_index_date")<f.col("study_start_date"))
                    | 
                    (f.col("mi_index_date")>f.col("mi_period_end_date"))))
count_var(df_temp, "person_id")

# COMMAND ----------

#df_3_2_a = (df_3_1
#.filter(
#    (f.col("mi_index_date")>=f.col("study_start_date"))
#    & 
#    (f.col("mi_index_date")<=f.col("mi_period_end_date"))
#    )
#    )

# COMMAND ----------

# Keep only the first MI event between study_start_date and mi_period_end_date
df_3_2 = (df_3_1_a.filter(f.col("mi_rank")==1)
.filter(
    (f.col("mi_index_date")>=f.col("study_start_date"))
    & 
    (f.col("mi_index_date")<=f.col("mi_period_end_date"))
    )
    .drop("mi_rank").drop("mi_count")
    )

# COMMAND ----------

print(proj_params.study_start_date)
print(proj_params.study_end_date)
print(proj_params.mi_period_end_date)

# COMMAND ----------

count_var(df_3_2, "person_id")

# COMMAND ----------


df_3_2.filter(f.col("mi_index_date").isNull()).count()

# COMMAND ----------

display(df_3_2.groupBy("year").agg(f.count("*")).orderBy(f.col("year").asc()))

# COMMAND ----------

# Rank MI events and find the maximum number of MI events per patient
#w_mi = Window.partitionBy("person_id").orderBy(f.col("mi_index_date").asc_nulls_last())
#df_3_2_b = df_3_2_a.withColumn("mi_rank", f.row_number().over(w_mi)).withColumn("mi_count", f.max#(f.col("mi_rank")).over(w_mi))

# COMMAND ----------

#display(df_3_2_b.limit(100))

# COMMAND ----------

#display(df_3_2_b.groupBy("mi_count").agg(f.count("*")))

# COMMAND ----------

# Keep the first MI event

#df_3_2 = df_3_2_b.filter(f.col("mi_rank")==1).drop("mi_rank").drop("mi_count")   


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3. Age 18 or above at time of index MI

# COMMAND ----------

display(df_3_2.select(["person_id", "date_of_birth", "mi_index_date", "mi_index_age"]).limit(10))

# COMMAND ----------

# How many <18
count_var(df_3_2.filter(f.col("mi_index_age")<18), "person_id")


# COMMAND ----------

df_3_3 = df_3_2.filter(f.col("mi_index_age")>=18)
count_var(df_3_3, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4. Alive on 2017/01/01
# MAGIC

# COMMAND ----------

display(df_3_3.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# double check
df_3_3.filter((f.col("date_of_death").isNotNull()) & (f.col("death_flag")==0)).count()

# COMMAND ----------

display(df_3_3.select("study_start_date").limit(1))

# COMMAND ----------

# How many not alive on 2017/01/01
df_temp = df_3_3.filter((f.col("death_flag")==1)&(f.col("date_of_death")<=f.col("study_start_date")))
count_var(df_temp, "person_id")

# COMMAND ----------



# COMMAND ----------

display(df_temp.select(["person_id", "death_flag", "date_of_death", "study_start_date"]))

# COMMAND ----------


display(df_temp.select(["person_id", "death_flag", "date_of_death", "study_start_date", "mi_index_date"]))

# COMMAND ----------

df_3_4_a = (df_3_3
          .filter(
              (f.col("death_flag")==0)|
              ( (f.col("death_flag")==1) &
                (f.col("date_of_death")>f.col("study_start_date"))
               )
              ))

# COMMAND ----------

count_var(df_3_4_a, "person_id")

# COMMAND ----------

display(df_3_4_a.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# Death before mi Index date

df_mi_death = df_3_4_a.filter(f.col("death_flag")==1).withColumn("pre_mi_death", f.when(f.col("date_of_death")<f.col("mi_index_date"), f.lit("death_pre_mi")).otherwise(f.lit("death_post_mi"))) 
display(df_mi_death.groupby("pre_mi_death").agg(f.count("*")))

# COMMAND ----------

# Add day difference and assess more than 2 weeks difference

df_mi_death = df_mi_death.withColumn("day_diff_death_mi", f.datediff(f.col("date_of_death"), f.col("mi_index_date")))


# COMMAND ----------

display(df_mi_death.select("person_id", "mi_index_date", "date_of_death", "day_diff_death_mi", "pre_mi_death"))


# COMMAND ----------

display(df_mi_death.filter(f.col("pre_mi_death")=="death_pre_mi").select("person_id", "mi_index_date", "date_of_death", "day_diff_death_mi", "pre_mi_death"))

# COMMAND ----------

df_mi_death.filter(f.col("pre_mi_death")=="death_pre_mi").filter(f.col("day_diff_death_mi")<-14).count()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

count_var(df_3_4_a, "person_id")

# COMMAND ----------

# Link back to the cohort and remobe those with a death date diff <-14 days
df_mi_death_sel = df_mi_death.select(["person_id", 'pre_mi_death',
 'day_diff_death_mi'])
df_3_4_b = df_3_4_a.join(df_mi_death_sel, on="person_id", how="left").fillna({"pre_mi_death":"alive"})
count_var(df_3_4_b, "person_id")

# COMMAND ----------

display(df_3_4_b.groupby("pre_mi_death").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_4_b.filter(f.col("day_diff_death_mi")<-14), "person_id")



# COMMAND ----------

df_3_4 = (df_3_4_b.filter
          (
              (f.col("pre_mi_death")=="alive")
              |(f.col("day_diff_death_mi")>=-14)
              ))
count_var(df_3_4, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5. Create a column showing date difference between MI diagnosis in Minap and hes apc

# COMMAND ----------

display(df_3_4.select((["person_id", "mi_index_date", "mi_first_hes_apc_date", "study_start_date"])).limit(10))

# COMMAND ----------


# Any mi_first_hes_apc_date before DOB?
display(df_3_4.filter(f.col("mi_first_hes_apc_date")<f.col("date_of_birth")))

# COMMAND ----------

df_3_5  = df_3_4.withColumn("day_diff_hes_mi_and_mi_index", f.datediff(f.col("mi_first_hes_apc_date"), f.col("mi_index_date")))

#df_3_5 = df_3_4.withColumn("day_diff_hes_mi_and_index_date", f.datediff(f.col("mi_first_hes_apc_date"), f.col("mi_index_date")))

# COMMAND ----------

# How many with previous HES APC MI?


count_var(df_3_5.filter(f.col("day_diff_hes_mi_and_mi_index")<0), "person_id")

# COMMAND ----------

# Check date difference (in days)

display(df_3_5.select(["person_id", "mi_first_hes_apc_date", "mi_index_date",  "day_diff_hes_mi_and_mi_index"]))

# COMMAND ----------

display(df_3_5.filter(f.col("day_diff_hes_mi_and_mi_index")<0).groupBy("day_diff_hes_mi_and_mi_index").agg(f.count("*")).orderBy(f.col("day_diff_hes_mi_and_mi_index").desc()))

# COMMAND ----------

display(df_3_5.filter(f.col("day_diff_hes_mi_and_mi_index")>=0).groupBy("day_diff_hes_mi_and_mi_index").agg(f.count("*")).orderBy(f.col("day_diff_hes_mi_and_mi_index").asc()))

# COMMAND ----------

display(df_3_5.filter(f.col("day_diff_hes_mi_and_mi_index")<0).select(["person_id", "mi_first_hes_apc_date", "mi_index_date", "day_diff_hes_mi_and_mi_index", "study_start_date"]).orderBy(f.col("day_diff_hes_mi_and_mi_index").desc()))

# COMMAND ----------

# How many with less than two weeks of difference
df_3_5 = (df_3_5
           .withColumn("summary_date_diff", 
                       f.when((f.col("day_diff_hes_mi_and_mi_index")<0) & (f.col("day_diff_hes_mi_and_mi_index")>=-14), f.lit("two_weeks"))
                       .when( (f.col("day_diff_hes_mi_and_mi_index")<-14), f.lit("more_than_two_weeks"))
                       .when(f.col("day_diff_hes_mi_and_mi_index").isNull(), f.lit("no HES MI"))
                       .otherwise(f.lit("valid"))
                      )
)
display(df_3_5.groupBy("summary_date_diff").agg(f.count("*")))

# COMMAND ----------

# How many with less than two weeks
count_var(df_3_5.filter(f.col("summary_date_diff")=="two_weeks"), "person_id")

# COMMAND ----------

# How many with more than two weeks
count_var(df_3_5.filter(f.col("summary_date_diff")=="more_than_two_weeks"), "person_id")

# COMMAND ----------

# How many with more HES MI after MINAP MI
count_var(df_3_5.filter(f.col("summary_date_diff")=="valid"), "person_id")

# COMMAND ----------

count_var(df_3_5, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 6. Exclude individuals with CKD+Dialysis+Transplant before MI index date
# MAGIC
# MAGIC Extract both _prevoius and _outcome phenotypes for CKD and ESRD (Dialysi+Tranplant)
# MAGIC
# MAGIC - Previous CKD+ESRD for exclusion: DOB to MI index date (No need to keep these variable as we exclude all with previous CKD+ESRD)
# MAGIC - CKD_outcome, ESRD_outcome: from index MI to study end date

# COMMAND ----------

gdppr_ckd= spark.table(f'{proj_params.dbc}.kdsc_ckd_diagnostic_phenotypes_gdppr_all_events')
display(gdppr_ckd.groupBy("phenotype_name", "phenotype_source").agg(f.count("*")))


# COMMAND ----------


hes_apc_ckd = spark.table(f'{proj_params.dbc}.kdsc_ckd_diagnostic_phenotypes_hes_apc_all_events')
display(hes_apc_ckd.groupBy("phenotype_name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

hes_apc_proc_ckd = spark.table(f'{proj_params.dbc}.kdsc_ckd_procedure_phenotypes_hes_apc_all_events')
display(hes_apc_proc_ckd.groupBy("phenotype_name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

display(gdppr_ckd.limit(5))

# COMMAND ----------




display(hes_apc_ckd.limit(5))


# COMMAND ----------

display(hes_apc_proc_ckd.limit(5))

# COMMAND ----------

# First limit these datses to patient IDs in MINAP Cphort
from pyspark.sql.functions import broadcast
mi_ids = df_3_5.select(["person_id", "mi_index_date"])  
broadcast_ids = broadcast(mi_ids)

# COMMAND ----------

gdppr_ckd_sel = gdppr_ckd.select(["person_id", "phenotype_name", "event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")
hes_apc_ckd_sel = hes_apc_ckd.select(["person_id", "phenotype_name", "event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")
hes_apc_proc_ckd_sel = hes_apc_proc_ckd.select(["person_id", "phenotype_name", "event_date_curated", "phenotype_source"]).join(broadcast_ids, on="person_id", how="inner")



# COMMAND ----------

# We need first event event dates across all datasets
union_ckd = gdppr_ckd_sel.union(hes_apc_ckd_sel).union(hes_apc_proc_ckd_sel)

# COMMAND ----------

# Add study start date and rename phenotypes based on pre- (inclusive) and post-

union_ckd = (union_ckd
             .withColumn("study_start_date", f.to_date(f.lit(proj_params.study_start_date)))
             .withColumn("mi_period_end_date", f.to_date(f.lit(proj_params.mi_period_end_date)))
             .withColumn("study_end_date", f.to_date(f.lit(proj_params.study_end_date)))
             )
display(union_ckd.limit(10))
                                

# COMMAND ----------

# Any mi_idnex_date on study start date? 
df_3_5.filter(f.col("study_start_date")==f.col("mi_index_date")).count()

# COMMAND ----------

display(union_ckd.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC CCU105 Team: Please confirm this decision
# MAGIC
# MAGIC For exclusion of pre-MI CKD: Use <= MI index date
# MAGIC For outcome CKD: Use > MI index  and <= study end date

# COMMAND ----------


union_ckd_b =(union_ckd
            .withColumn ("phenotype_name", 
                         f.when(f.col("event_date_curated")<=f.col("mi_index_date"), 
                                f.concat(f.col("phenotype_name"), f.lit("_previous")))
                         .when((f.col("event_date_curated")> f.col("mi_index_date")) & (f.col("event_date_curated")<=f.col("study_end_date")), f.concat(f.col("phenotype_name"), f.lit("_outcome")))
                                  .otherwise(f.concat(f.col("phenotype_name"), f.lit("_out_of_range"))))
            )

# COMMAND ----------

display(union_ckd_b.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

display(union_ckd_b.filter(f.col("phenotype_name")=="ckd_previous").limit(10))

# COMMAND ----------

display(union_ckd_b.filter(f.col("phenotype_name")=="ckd_outcome").limit(10))

# COMMAND ----------

display(union_ckd_b.filter(f.col("phenotype_name")=="ckd_out_of_range").limit(10))

# COMMAND ----------

# Drop out of range phenotyeps
union_ckd_c = union_ckd_b.filter(~f.col("phenotype_name").endswith("_out_of_range"))
display(union_ckd_c.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

# We need first event event dates across all datasets
w = Window.partitionBy("person_id", "phenotype_name").orderBy(f.col("event_date_curated").asc_nulls_last())
union_ckd_long = union_ckd_c.withColumn("rank", f.row_number().over(w)).filter(f.col("rank")==1  ).drop("rank")
display(union_ckd_long.limit(5))

# COMMAND ----------

display(union_ckd_long.groupBy("phenotype_name").agg(f.count("*")))

# COMMAND ----------

# Long to wide
ckd_wide =union_ckd_long.groupBy("person_id").pivot("phenotype_name").agg(f.first("event_date_curated"))

# COMMAND ----------

display(ckd_wide.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ESRD and CKD+ESRD (for excluding pre-study CKD)

# COMMAND ----------



# COMMAND ----------



ckd_wide = (ckd_wide
            .withColumn("esrd_previous", f.least(f.col("dialysis_previous"), f.col("transplant_previous")))
            .withColumn("esrd_outcome", f.least(f.col("dialysis_outcome"), f.col("transplant_outcome")))
            .withColumn("ckd_dialysis_transplant_previous", f.least(f.col("ckd_previous"), f.col("dialysis_previous"), f.col("transplant_previous")))
            .withColumn("ckd_dialysis_transplant_outcome", f.least(f.col("ckd_outcome"), f.col("dialysis_outcome"), f.col("transplant_outcome")))

            )

# COMMAND ----------

display(ckd_wide.limit(100))

# COMMAND ----------

count_var(ckd_wide, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join with the MI cohort

# COMMAND ----------

ckd_wide.columns

# COMMAND ----------

df_3_6_temp = (df_3_5
               .join(ckd_wide
                     , on="person_id"
                     , how="left")
               )

# COMMAND ----------

# Any event dates before DOB?
print(df_3_6_temp.filter(f.col("ckd_outcome")<f.col("date_of_birth")).count())
print(df_3_6_temp.filter(f.col("dialysis_outcome")<f.col("date_of_birth")).count())
print(df_3_6_temp.filter(f.col("transplant_outcome")<f.col("date_of_birth")).count())
print(df_3_6_temp.filter(f.col("ckd_previous")<f.col("date_of_birth")).count())
print(df_3_6_temp.filter(f.col("dialysis_previous")<f.col("date_of_birth")).count())
print(df_3_6_temp.filter(f.col("transplant_previous")<f.col("date_of_birth")).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Any combined CKD before MI index date?

# COMMAND ----------

count_var(df_3_6_temp, "person_id")

# COMMAND ----------

df_exclude = df_3_6_temp.filter(f.col("ckd_dialysis_transplant_previous")<=f.col("mi_index_date"))
count_var(df_exclude, "person_id")

# COMMAND ----------

# This should retuwn the same value
df_exclude_check = df_3_6_temp.filter(f.col("ckd_dialysis_transplant_previous").isNotNull())
count_var(df_exclude_check, "person_id")

# COMMAND ----------

df_3_6 = df_3_6_temp.filter((f.col("ckd_dialysis_transplant_previous").isNull()) | (f.col("ckd_dialysis_transplant_previous")>f.col("mi_index_date")))
count_var(df_3_6, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Note: We have not applied any filtering based on the maximum date of CKD or ESRD outcome. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 4. Save

# COMMAND ----------

# spark.sql(f'drop table {proj_params.dbc}.ccu105_minap_cohort_part_3')

# COMMAND ----------

# df_3_6.write.mode("overwrite").saveAsTable(f'{proj_params.dbc}.ccu105_minap_cohort_part_3')
save_table(df_3_6, out_name=f'ccu105_minap_cohort_part_3', save_previous=True)

# COMMAND ----------

df_3_6_reload = spark.table(f'{proj_params.dbc}.ccu105_minap_cohort_part_3')

# COMMAND ----------

display(df_3_6_reload.limit(10))

# COMMAND ----------

display(df_3_6_reload.filter(f.col("esrd_outcome").isNotNull()).limit(10))

# COMMAND ----------

for item in list(np.sort(df_3_6_reload.columns)):
    print(item)

# COMMAND ----------

