# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # CCU105_01_parameters
# MAGIC
# MAGIC **Description** The parameters notebook defines a set of parameters, which is loaded in each notebook in the data curation pipeline to create datasets for the CCU105 project. 
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**

# COMMAND ----------

# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

import pyspark.sql.functions as f
import pandas as pd
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 Datasets, Paths and Variables

# COMMAND ----------

class params_class:
    # -----------------------------------------------------------------------------
    # Project
    # -----------------------------------------------------------------------------
    proj = "ccu105"
    # -----------------------------------------------------------------------------
    # Project specific
    # -----------------------------------------------------------------------------
    tmp_year = "2025"
    tmp_month = "04"
    tmp_day = "24"
    tmp_archived_on = "-".join([tmp_year, tmp_month, tmp_day])
    tmp_archived_on_str = "_".join([tmp_year, tmp_month, tmp_day])
    old_minap_archived_on = "2023-12-27"
    old_minap_archived_on_str = "2023_12_27"
    study_start_date = "2017-01-01"
    study_end_date = "2024-12-31"
    #mi_period_end_date = "2023-03-31"
    mi_period_end_date = "2023-12-31"
    # -----------------------------------------------------------------------------
    db = ""
    dbc_old = f""
    dbc = ""
    dss = "reference_data.dss_corporate"
    list_datasets = [
        [
            "deaths",
            dbc_old,
            f"deaths__archive",
            tmp_archived_on,
            "DEC_CONF_NHS_NUMBER_CLEAN_DEID",
            "REG_DATE_OF_DEATH",
        ],
        [
            "gdppr",
            dbc_old,
            f"gdppr__archive",
            tmp_archived_on,
            "NHS_NUMBER_DEID",
            "DATE",
        ],
        [
            "hes_apc",
            dbc_old,
            f"hes_apc_all_years_archive",
            tmp_archived_on,
            "PERSON_ID_DEID",
            "EPISTART",
        ],
        [
            "pmeds",
            dbc_old,
            f"primary_care_meds_{db}_archive",
            tmp_archived_on,
            "PERSON_ID_DEID",
            "ProcessingPeriodDate",
        ],
        [   "vacc",
            dbc_old,
            f'vaccine_status_{db}_archive',
            tmp_archived_on,
            'PERSON_ID_DEID', 
            'DATE_AND_TIME'
        ],
        [
            "minap",
            dbc_old, 
            f'nicor_minap__archive',
            tmp_archived_on,
            'NHS_NUMBER_DEID', 
            'ARRIVAL_AT_HOSPITAL'

        ],
        [
            "minap_old",
            dbc_old, 
            f'nicor_minap__archive',
            old_minap_archived_on,
            'NHS_NUMBER_DEID', 
            'ARRIVAL_AT_HOSPITAL'

        ],
        
        [
            "token", 
            dbc_old, 
            f'token_pseudo_id_lookup_archive',
            tmp_archived_on,
            'pseudo_id',
            ''
        ]
    ]
    pd_datasets = pd.DataFrame(
        list_datasets, columns=["dataset", "database", "table", "archived_on", "idVar", "dateVar"])
    
    path_gdppr_arch = f"{dbc_old}.gdppr__archive"
    path_hes_apc_arch = f"{dbc_old}.hes_apc_all_years_archive"
    path_pmeds_arch = f"{dbc_old}.primary_care_meds__archive"
    path_deaths_arch = f"{dbc_old}.deaths__archive"
    path_covid_vacc_arch = f"{dbc_old}.vaccine_status__archive"
    path_minap_arch = f"{dbc_old}.nicor_minap__archive"
    # curated tables
    path_demographics = f"{dbc}.hds_curated_assets__demographics_{tmp_archived_on_str}"
    path_patient_id_type_lookup = f"{db}.token_pseudo_id_lookup"
    path_hes_apc_cips = f"{dbc}.hds_curated_assets__hes_apc_cips_cips_{tmp_archived_on_str}"
    path_hes_apc_epis = (
        f"{dbc}.hds_curated_assets__hes_apc_cips_episodes_{tmp_archived_on_str}")
    path_hes_apc_pspells = (
        f"{dbc}.hds_curated_assets__hes_apc_cips_provider_spells_{tmp_archived_on_str}")
    path_hes_apc_long = f"{dbc}.hds_curated_assets__hes_apc_diagnosis_{tmp_archived_on_str}"
    path_hes_apc_procedure_long = f"{dbc}.hds_curated_assets__hes_apc_procedure_{tmp_archived_on_str}"
    path_cur_deaths_sing = f"{dbc}.hds_curated_assets__deaths_single_{tmp_archived_on_str}"
    path_cur_deaths_long = (
        f"{dbc}.hds_curated_assets__deaths_cause_of_death_{tmp_archived_on_str}")
    path_lsoa_multi = f"{dbc}.hds_curated_assets__lsoa_multisource_{tmp_archived_on_str}"
    path_dob_ind = (
        f"{dbc}.hds_curated_assets__date_of_birth_individual_{tmp_archived_on_str}")
    path_dob_multi = (
        f"{dbc}.hds_curated_assets__date_of_birth_multisource_{tmp_archived_on_str}")
    path_cur_covid = f"{dbc}.hds_curated_assets__covid_positive_{tmp_archived_on_str}"
    # reference tables
    path_mainspef = f"{dss}.hesf_mainspef"
    path_tretspef = f"{dss}.hesf_tretspef"
    path_icd10 = f"{dss}.icd10_codes"
    path_lsoa_2011_imd_lookup = f"{dbc}.hds_cur_lsoa_2011_imd_lookup"
    path_lsoa_2011_reg_lookup = f"{dbc}.hds_cur_lsoa_region_lookup"
    path_gdppr_refset = f"{dss}.gdppr_cluster_refset"
    # Curated within KDSC 
    path_gdppr_preprocessed = f"{dbc}.kdsc_gdppr_preprocessed_mm"
    path_hes_apc_preprocessed = f"{dbc}.kdsc_hes_apc_preprocessed_mm"
    path_hes_apc_procedure_preprocessed = f"{dbc}.kdsc_hes_apc_procedure_preprocessed_mm"
# codelist tables
# path_out_codelist_quality_assurance = f'{dbc}.{proj}_out_codelist_quality_assurance'

proj_params = params_class()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 Functions

# COMMAND ----------

# DBTITLE 1,Extract batch from archive function
# function to extract the batch corresponding to the pre-defined archived_on date from the archive for the specified dataset
from pyspark.sql import DataFrame
def extract_batch_from_archive(_df_datasets: DataFrame, _dataset: str, verbose_counts = False):
  
  # get row from df_archive_tables corresponding to the specified dataset
  _row = _df_datasets[_df_datasets['dataset'] == _dataset]
  
  # check one row only
  assert _row.shape[0] != 0, f"dataset = {_dataset} not found in _df_datasets (datasets = {_df_datasets['dataset'].tolist()})"
  assert _row.shape[0] == 1, f"dataset = {_dataset} has >1 row in _df_datasets"
  
  # create path and extract archived on
  _row = _row.iloc[0]
  _path = _row['database'] + '.' + _row['table']  
  _archived_on = _row['archived_on']  
  print(_path + ' (archived_on = ' + _archived_on + ')')
  
  # check path exists # commented out for runtime
#   _tmp_exists = spark.sql(f"SHOW TABLES FROM {_row['database']}")\
#     .where(f.col('tableName') == _row['table'])\
#     .count()
#   assert _tmp_exists == 1, f"path = {_path} not found"

  # extract batch
  _tmp = spark.table(_path)\
    .where(f.col('archived_on') == _archived_on)  
  
  # check number of records returned
  if verbose_counts:
      _tmp_records = _tmp.count()
      print(f'  {_tmp_records:,} records')
      assert _tmp_records > 0, f"number of records == 0"

  # return dataframe
  return _tmp

# COMMAND ----------

display(proj_params.pd_datasets)

# COMMAND ----------

gdppr = extract_batch_from_archive(proj_params.pd_datasets, 'gdppr')
hes_apc = extract_batch_from_archive(proj_params.pd_datasets, 'hes_apc')
pmeds = extract_batch_from_archive(proj_params.pd_datasets, 'pmeds')
deaths = extract_batch_from_archive(proj_params.pd_datasets, 'deaths')
vacc = extract_batch_from_archive(proj_params.pd_datasets, 'vacc')
minap = extract_batch_from_archive(proj_params.pd_datasets, 'minap')
minap_old = extract_batch_from_archive(proj_params.pd_datasets, 'minap_old')
token = extract_batch_from_archive(proj_params.pd_datasets, 'token')

# COMMAND ----------

# DBTITLE 1,Save Table
# def __save_table(df, out_name:str, save_previous=True, data_base:str=f''):
  
#   # assert that df is a dataframe
#   assert isinstance(df, f.DataFrame), 'df must be of type dataframe' #isinstance(df, pd.DataFrame) | 
#   # if a pandas df then convert to spark
#   #if(isinstance(df, pd.DataFrame)):
#     #df = (spark.createDataFrame(df))
  
#   # save name check
#   if(any(char.isupper() for char in out_name)): 
#     print(f'Warning: {out_name} converted to lowercase for saving')
#     out_name = out_name.lower()
#     print('out_name: ' + out_name)
#     print('')
  
#   # df name
#   df_name = [x for x in globals() if globals()[x] is df][0]
  
#   # ------------------------------------------------------------------------------------------------
#   # save previous version for comparison purposes
#   if(save_previous):
#     tmpt = (
#       spark.sql(f"""SHOW TABLES FROM {data_base}""")
#       .select('tableName')
#       .where(f.col('tableName') == out_name)
#       .collect()
#     )
#     if(len(tmpt)>0):
#       # save with production date appended
#       _datetimenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#       out_name_pre = f'{out_name}_pre{_datetimenow}'.lower()
#       print(f'saving (previous version):')
#       print(f'  {out_name}')
#       print(f'  as')
#       print(f'  {out_name_pre}')
#       spark.table(f'{data_base}.{out_name}').write.mode('overwrite').saveAsTable(f'{data_base}.{out_name_pre}')
#       #spark.sql(f'ALTER TABLE {data_base}.{out_name_pre} OWNER TO {data_base}')
#       print('saved')
#       print('') 
#     else:
#       print(f'Warning: no previous version of {out_name} found')
#       print('')
#   # ------------------------------------------------------------------------------------------------  
  
#   # save new version
#   print(f'saving:')
#   print(f'  {df_name}')
#   print(f'  as')
#   print(f'  {out_name}')
#   df.write.mode('overwrite').option("overwriteSchema", "True").saveAsTable(f'{data_base}.{out_name}')
#   #spark.sql(f'ALTER TABLE {data_base}.{out_name} OWNER TO {data_base}')
#   print('saved')

# COMMAND ----------

