import pandas as pd
import numpy as np
import duckdb
import os
import shutil
from etl_utils import descompactar_e_mover,geotiff_to_dataframe,encontrar_coordenadas_mais_proximas,calcular_intersecoes_estacoes_vizinhas
from tqdm.notebook import tqdm

## ------------------------------------------------------------------------------------------------ ##
## --------------------------------------- UNZIPPING LANDING -------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

# GPM Final Run
gpm_final_run_zipped_files_path = 'landing/zipados/GPM-final-run'
gpm_final_run_unzipped_files_path = 'landing/unzipados/GPM-final-run'
for year_zip in os.listdir(gpm_final_run_zipped_files_path):
    year = year_zip.split('.zip')[0]
    descompactar_e_mover(f'{gpm_final_run_zipped_files_path}/{year_zip}',f'{gpm_final_run_unzipped_files_path}/{year}')

# GPM Late Run
gpm_final_run_zipped_files_path = 'landing/zipados/GPM-late-run'
gpm_final_run_unzipped_files_path = 'landing/unzipados/GPM-late-run'
for year_zip in os.listdir(gpm_final_run_zipped_files_path):
    year = year_zip.split('.zip')[0]
    descompactar_e_mover(f'{gpm_final_run_zipped_files_path}/{year_zip}',f'{gpm_final_run_unzipped_files_path}/{year}')

# CPC
cpc_zipped_files_path = 'landing/zipados/CPC'
cpc_unzipped_files_path = 'landing/unzipados/CPC'
for folder_zip in os.listdir(cpc_zipped_files_path):
    print(folder_zip)
    folder = folder_zip.split('.zip')[0]
    descompactar_e_mover(
        f'{cpc_zipped_files_path}/{folder_zip}',
        f'{cpc_unzipped_files_path}/{folder}')
    
# POWER
power_zipped_files_path = 'landing/zipados/POWER'
power_unzipped_files_path = 'landing/unzipados/POWER'
for folder_zip in os.listdir(power_zipped_files_path):
    print(folder_zip)
    folder = folder_zip.split('.zip')[0]
    descompactar_e_mover(
        f'{power_zipped_files_path}/{folder_zip}',
        f'{power_unzipped_files_path}/{folder}')
    
# CHIRPS
chirps_zipped_files_path = 'landing/zipados/chirps'
chirps_unzipped_files_path = 'landing/unzipados/chirps'
for folder_zip in os.listdir(chirps_zipped_files_path):
    print(folder_zip)
    folder = folder_zip.split('.zip')[0]
    descompactar_e_mover(
        f'{chirps_zipped_files_path}/{folder_zip}',
        f'{chirps_unzipped_files_path}/{folder}')
    

## ------------------------------------------------------------------------------------------------ ##
## --------------------------------------- LANDING TO BRONZE -------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

bronze_conn = duckdb.connect("bronze_db")

# Estações
estacoes_bronze_tables_dict = {
    'dim_estacoes':'bronze/dim_estacoes.csv',
    'fato_estacoes_precipitacao':'bronze/fato_estacoes_precipitacao.csv',
    'fato_estacoes_temperatura_maxima':'bronze/fato_estacoes_temperatura_maxima.csv',
    'fato_estacoes_temperatura_media':'bronze/fato_estacoes_temperatura_media.csv',
    'fato_estacoes_temperatura_minima':'bronze/fato_estacoes_temperatura_minima.csv',
    'fato_estacoes_umidade_relativa_maxima':'bronze/fato_estacoes_umidade_relativa_maxima.csv',
    'fato_estacoes_umidade_relativa_media':'bronze/fato_estacoes_umidade_relativa_media.csv',
    'fato_estacoes_umidade_relativa_minima':'bronze/fato_estacoes_umidade_relativa_minima.csv',
    'fato_estacoes_velocidade_vento_2m_maxima':'bronze/fato_estacoes_velocidade_vento_2m_maxima.csv',
    'fato_estacoes_velocidade_vento_2m_media':'bronze/fato_estacoes_velocidade_vento_2m_media.csv',
    'fato_estacoes_velocidade_vento_10m_media':'bronze/fato_estacoes_velocidade_vento_10m_media.csv',
}

for table_name,table_path in estacoes_bronze_tables_dict.items():
    bronze_conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
            SELECT
                *
            FROM read_csv('{table_path}')
""")

## PRODUTOS - GPM FINAL RUN
gpm_final_run_table_name = 'fato_produto_gpm_final_run_precipitacao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
dataframes = []
base_path = 'landing/unzipados/GPM-final-run'
years = os.listdir(base_path)

for year in years:
    print(year)
    year_path = os.path.join(base_path, year, year)
    if not os.path.isdir(year_path):
        continue
    months = os.listdir(year_path)
    
    for month in months:
        print(f'> {month}')
        month_path = os.path.join(year_path, month)
        
        for filename in os.listdir(month_path):
            if not filename.endswith('.tif'):
                continue
            try:
                day = filename.split('.tif')[0][-2:]
                path = os.path.join(month_path, filename)
                df = geotiff_to_dataframe(path, min_lon, max_lon, min_lat, max_lat, band, 'vl_precipitacao')
                df['dt_medicao'] = f'{year}-{month}-{day}'
                dataframes.append(df)
            except Exception as e:
                print(f'Erro ao processar {filename}: {e}')

gpm_final_run_df = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {gpm_final_run_table_name} AS
    SELECT
        *
    FROM gpm_final_run_df
""")


## PRODUTOS - GPM LATE RUN
gpm_late_run_table_name = 'fato_produto_gpm_late_run_precipitacao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
dataframes = []
base_path = 'landing/unzipados/GPM-late-run'
years = os.listdir(base_path)

for year in years:
    print(year)
    year_path = os.path.join(base_path, year, year)
    if not os.path.isdir(year_path):
        continue
    months = os.listdir(year_path)
    
    for month in months:
        print(f'> {month}')
        month_path = os.path.join(year_path, month)
        
        for filename in os.listdir(month_path):
            if not filename.endswith('.tif'):
                continue
            try:
                day = filename.split('.tif')[0][-33:-31]
                path = os.path.join(month_path, filename)
                df = geotiff_to_dataframe(path, min_lon, max_lon, min_lat, max_lat, band, 'vl_precipitacao')
                df['dt_medicao'] = f'{year}-{month}-{day}'
                dataframes.append(df)
            except Exception as e:
                print(f'Erro ao processar {filename}: {e}')

gpm_late_run_df = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {gpm_late_run_table_name} AS
    SELECT
        *
    FROM gpm_late_run_df
""")

## PRODUTOS - CPC (PRECIPITAÇÃO)
cpc_precipitacao_table_name = 'fato_produto_cpc_precipitacao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
cpc_root_folder = os.path.join('landing','unzipados','CPC')
cpc_precip_folders = [folder for folder in os.listdir(cpc_root_folder) if folder.startswith('precip')]
cpc_file_paths = []
dataframes = []
for folder in cpc_precip_folders:
    cpc_file_paths+=[os.path.join(cpc_root_folder,folder,file) for file in  os.listdir(os.path.join(cpc_root_folder,folder))]

for file in cpc_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_precipitacao')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

cpc_precipitacao = pd.concat(dataframes,ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {cpc_precipitacao_table_name} AS
    SELECT
        *
    FROM cpc_precipitacao
""")

## PRODUTOS - CPC (TEMPERATURA MÁXIMA)
cpc_tmax_table_name = 'fato_produto_cpc_temperatura_maxima'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
cpc_root_folder = os.path.join('landing','unzipados','CPC')
cpc_precip_folders = [folder for folder in os.listdir(cpc_root_folder) if folder.startswith('tmax')]
cpc_file_paths = []
dataframes = []
for folder in cpc_precip_folders:
    cpc_file_paths+=[os.path.join(cpc_root_folder,folder,file) for file in  os.listdir(os.path.join(cpc_root_folder,folder))]


for file in cpc_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_maxima')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

cpc_tmax = pd.concat(dataframes,ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {cpc_tmax_table_name} AS
    SELECT
        *
    FROM cpc_tmax
""")

## PRODUTOS - CPC (TEMPERATURA MÍNIMA)
cpc_tmin_table_name = 'fato_produto_cpc_temperatura_minima'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
cpc_root_folder = os.path.join('landing','unzipados','CPC')
cpc_precip_folders = [folder for folder in os.listdir(cpc_root_folder) if folder.startswith('tmin')]
cpc_file_paths = []
dataframes = []
for folder in cpc_precip_folders:
    cpc_file_paths+=[os.path.join(cpc_root_folder,folder,file) for file in  os.listdir(os.path.join(cpc_root_folder,folder))]

for file in cpc_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_minima')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

cpc_tmin = pd.concat(dataframes,ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {cpc_tmin_table_name} AS
    SELECT
        *
    FROM cpc_tmin
""")

## PRODUTOS - POWER (IRRADIANCIA ALL SKY)
power_irradiancia_allsky_table_name = 'fato_produto_power_irradiancia_allsky'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing','unzipados','POWER')
years = os.listdir(power_root_folder)
power_irradiancia_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder,year,year))
    for month in months:
        power_irradiancia_file_paths+=[os.path.join(power_root_folder,year,year,month,file) for file in os.listdir(os.path.join(power_root_folder,year,year,month)) if 'ALLSKY_SFC_SW_DWN' in file]

dataframes = []
for file in power_irradiancia_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_irradiancia_allsky')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_irradiancia = pd.concat(dataframes,ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_irradiancia_allsky_table_name} AS
    SELECT
        *
    FROM power_irradiancia
""")

## PRODUTOS - POWER (PRECIPITAÇÃO TOTAL CORRIGIDA)

power_precipitacao_table_name = 'fato_produto_power_precipitacao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_precipitacao_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_precipitacao_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'PRECTOTCORR' in file
        ]

dataframes = []
for file in power_precipitacao_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_precipitacao')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_precipitacao = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_precipitacao_table_name} AS
    SELECT
        *
    FROM power_precipitacao
""")

## PRODUTOS - POWER (PRESSÃO ATMOSFÉRICA NA SUPERFÍCIE)power_pressao_table_name = 'fato_produto_power_pressao'
power_pressao_table_name = 'fato_produto_power_pressao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_pressao_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_pressao_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'PS' in file
        ]

dataframes = []
for file in power_pressao_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_pressao_nivel_superficie')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_pressao = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_pressao_table_name} AS
    SELECT
        *
    FROM power_pressao
""")

## PRODUTOS - POWER (TEMPERATURA MÁXIMO A 2 METROS DE ALTURA)
power_tempmax_table_name = 'fato_produto_power_tempmax'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_tempmax_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_tempmax_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'T2M_MAX' in file
        ]

dataframes = []
for file in power_tempmax_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_maxima_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_tempmax = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_tempmax_table_name} AS
    SELECT
        *
    FROM power_tempmax
""")

## PRODUTOS - POWER (TEMPERATURA MÍNIMA A 2 METROS DE ALTURA)
power_tempmin_table_name = 'fato_produto_power_tempmin'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_tempmin_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_tempmin_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'T2M_MIN' in file
        ]

dataframes = []
for file in power_tempmin_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_minima_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_tempmin = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_tempmin_table_name} AS
    SELECT
        *
    FROM power_tempmin
""")

## PRODUTOS - POWER (TEMPERATURA MÉDIA A 2 METROS DE ALTURA)
power_tempmedia_table_name = 'fato_produto_power_tempmedia'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_tempmedia_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_tempmedia_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'T2M' in file and 'MAX' not in file and 'MIN' not in file and 'DEW' not in file
        ]

dataframes = []
for file in power_tempmedia_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_media_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_tempmedia = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_tempmedia_table_name} AS
    SELECT
        *
    FROM power_tempmedia
""")

## PRODUTOS - POWER (TEMPERATURA ORVALHO)
power_temp_orvalho_table_name = 'fato_produto_power_temp_orvalho'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_temp_orvalho_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_temp_orvalho_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'T2MDEW' in file
        ]

dataframes = []
for file in power_temp_orvalho_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_temperatura_orvalho_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_temp_orvalho = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_temp_orvalho_table_name} AS
    SELECT
        *
    FROM power_temp_orvalho
""")

## PRODUTOS - POWER (UMIDADE RELATIVA 2M)

power_umidade_2m_table_name = 'fato_produto_power_umidade_2m'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_umidade_2m_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_umidade_2m_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'RH2M' in file
        ]

dataframes = []
for file in power_umidade_2m_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_umidade_relativa_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_umidade_2m = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_umidade_2m_table_name} AS
    SELECT
        *
    FROM power_umidade_2m
""")

## PRODUTOS - POWER (VENTO MÉDIO 2M)

power_vento_2m_table_name = 'fato_produto_power_vento_2m'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_vento_2m_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_vento_2m_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WS2M' in file and 'MAX' not in file
        ]

dataframes = []
for file in power_vento_2m_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_vento_medio_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_vento_2m = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_vento_2m_table_name} AS
    SELECT
        *
    FROM power_vento_2m
""")

## PRODUTOS - POWER (VENTO MÁXIMO 2M)

power_vento_2m_max_table_name = 'fato_produto_power_vento_2m_max'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_vento_2m_max_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_vento_2m_max_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WS2M_MAX' in file
        ]

dataframes = []
for file in power_vento_2m_max_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_vento_maximo_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_vento_2m_max = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_vento_2m_max_table_name} AS
    SELECT
        *
    FROM power_vento_2m_max
""")

## PRODUTOS - POWER (VENTO MÉDIO 10M)

power_vento_10m_table_name = 'fato_produto_power_vento_10m'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_vento_10m_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_vento_10m_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WS10M' in file and 'MAX' not in file
        ]

dataframes = []
for file in power_vento_10m_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_vento_10m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_vento_10m = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_vento_10m_table_name} AS
    SELECT
        *
    FROM power_vento_10m
""")

## PRODUTOS - POWER (VENTO MÁXIMO 10M)

power_vento_10m_max_table_name = 'fato_produto_power_vento_10m_max'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_vento_10m_max_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_vento_10m_max_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WS10M_MAX' in file
        ]

dataframes = []
for file in power_vento_10m_max_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_vento_maximo_10m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_vento_10m_max = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_vento_10m_max_table_name} AS
    SELECT
        *
    FROM power_vento_10m_max
""")

## PRODUTOS - POWER (DIREÇÃO DO VENTO 2M)

power_direcao_vento_2m_table_name = 'fato_produto_power_direcao_vento_2m'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_direcao_vento_2m_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_direcao_vento_2m_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WD2M' in file
        ]

dataframes = []
for file in power_direcao_vento_2m_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_direcao_vento_2m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_direcao_vento_2m = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_direcao_vento_2m_table_name} AS
    SELECT
        *
    FROM power_direcao_vento_2m
""")

## PRODUTOS - POWER (DIREÇÃO DO VENTO 10M)

power_direcao_vento_10m_table_name = 'fato_produto_power_direcao_vento_10m'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
power_root_folder = os.path.join('landing', 'unzipados', 'POWER')
years = os.listdir(power_root_folder)
power_direcao_vento_10m_file_paths = []

for year in years:
    months = os.listdir(os.path.join(power_root_folder, year, year))
    for month in months:
        power_direcao_vento_10m_file_paths += [
            os.path.join(power_root_folder, year, year, month, file)
            for file in os.listdir(os.path.join(power_root_folder, year, year, month))
            if 'WD10M' in file
        ]

dataframes = []
for file in power_direcao_vento_10m_file_paths:
    try:
        day = file[-6:-4]
        month = file[-8:-6]
        year = file[-12:-8]
        df = geotiff_to_dataframe(file, min_lon, max_lon, min_lat, max_lat, band, 'vl_direcao_vento_10m')
        df['dt_medicao'] = f'{year}-{month}-{day}'
        dataframes.append(df)
    except Exception as e:
        print(f'Erro ao processar {file}: {e}')

power_direcao_vento_10m = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {power_direcao_vento_10m_table_name} AS
    SELECT
        *
    FROM power_direcao_vento_10m
""")

## PRODUTOS - CHIRPS
chirps_final_run_table_name = 'fato_produto_chirps_precipitacao'
min_lon, max_lon = (-54, -37)
min_lat, max_lat = (-25, -16)
band = 1
dataframes = []
base_path = 'landing/unzipados/chirps'
years = os.listdir(base_path)

for year in years:
    print(year)
    year_path = os.path.join(base_path, year)
    if not os.path.isdir(year_path):
        continue
    months = os.listdir(year_path)
    
    for month in months:
        print(f'> {month}')
        month_path = os.path.join(year_path, month)
        
        for filename in os.listdir(month_path):
            if not filename.endswith('.tif'):
                continue
            try:
                day = filename.split('.tif')[0][-2:]
                path = os.path.join(month_path, filename)
                df = geotiff_to_dataframe(path, min_lon, max_lon, min_lat, max_lat, band, 'vl_precipitacao')
                df['dt_medicao'] = f'{year}-{month}-{day}'
                dataframes.append(df)
            except Exception as e:
                print(f'Erro ao processar {filename}: {e}')

chirps_df = pd.concat(dataframes, ignore_index=True)

bronze_conn.execute(f"""
CREATE OR REPLACE TABLE {chirps_final_run_table_name} AS
    SELECT
        *
    FROM chirps_df
""")

## ------------------------------------------------------------------------------------------------ ##
## ---------------------------------------- BRONZE TO PRATA --------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

bronze_conn = duckdb.connect("bronze_db")
bronze_conn.execute("INSTALL spatial; LOAD spatial")
prata_conn = duckdb.connect("prata_db")
prata_conn.execute("INSTALL spatial; LOAD spatial")

## -------------------- ##
## MATRIZ DE DISTÂNCIAS ##
## -------------------- ##

prata_matriz_distancias_table_name  = "fato_estacoes_distancia"

df_matriz_distancias = bronze_conn.execute(f"""
    WITH estacoes AS (
        SELECT DISTINCT
            id_estacao
            ,ST_POINT(longitude,latitude) AS point
        FROM dim_estacoes)
    SELECT 
        estacoes_1.id_estacao as id_estacao_base
        ,estacoes_2.id_estacao as id_estacao_candidata
        ,CAST(ST_Distance_Sphere(estacoes_1.point,estacoes_2.point)/1000 AS DECIMAL(8,2))  AS vl_distancia_km
    FROM estacoes AS estacoes_1
    CROSS JOIN estacoes AS estacoes_2
    WHERE estacoes_1.id_estacao <> estacoes_2.id_estacao
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_matriz_distancias_table_name} AS
    SELECT
        *
    FROM df_matriz_distancias
""")

## --------------------- ##
## MATRIZ DE INTERSEÇÕES ##
## --------------------- ##

prata_matriz_distancias_table_name  = "fato_estacoes_intersecao"

fato_estacoes = prata_conn.execute("select id_estacao,dt_medicao,vl_precipitacao from fato_estacoes").fetch_df()
fato_estacoes['dt_medicao'] = fato_estacoes['dt_medicao'].astype(str)

datas_por_estacao = {}
print("Pré-calculando datas por estação...")
for id_estacao in tqdm(set(fato_estacoes['id_estacao']), desc="Preparando dados de datas"):
    datas_por_estacao[id_estacao] = set(fato_estacoes.loc[fato_estacoes['id_estacao'] == id_estacao, 'dt_medicao'])
print("Pré-cálculo concluído.")

print("Iniciando o cálculo de interseções...")
intersecoes_dict = {}
for id_estacao_base in tqdm(fato_estacoes['id_estacao'].unique(), desc=f"Calculando interseções", leave=False):
    intersecoes_dict[id_estacao_base] = calcular_intersecoes_estacoes_vizinhas(id_estacao_base,fato_estacoes,datas_por_estacao)

for id_estacao_base in intersecoes_dict.keys():
    del intersecoes_dict[id_estacao_base][id_estacao_base]

df_matriz_intersecoes = pd.DataFrame.from_dict(intersecoes_dict,orient='index').stack().reset_index() \
    .rename(columns = {
        'level_0':'id_estacao_base',
        'level_1':'id_estacao_candidata',
        0:'pct_intersecao_precipitacao'
    })

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_matriz_distancias_table_name} AS
    SELECT
        *
    FROM df_matriz_intersecoes
""")


## ------------- ##
## FATO ESTAÇÕES ##
## ------------- ##

prata_fato_estacoes_table_name = "fato_estacoes"

prata_fato_estacoes = bronze_conn.execute(f"""
SELECT
    precipitacao.dt_medicao
    ,precipitacao.id_estacao
    ,precipitacao.vl_precipitacao
    ,precipitacao.fl_dado_ausente AS fl_precipitacao_dado_ausente
    ,precipitacao.fl_dado_duvidoso AS fl_precipitacao_dado_duvidoso
    ,precipitacao.fl_dado_invalido AS fl_precipitacao_dado_invalido
    ,precipitacao.fl_dado_valido AS fl_precipitacao_dado_valido

    ,temperatura_maxima.vl_temperatura_maxima
    ,temperatura_maxima.fl_dado_ausente AS fl_temperatura_maxima_dado_ausente
    ,temperatura_maxima.fl_dado_duvidoso AS fl_temperatura_maxima_dado_duvidoso
    ,temperatura_maxima.fl_dado_invalido AS fl_temperatura_maxima_dado_invalido
    ,temperatura_maxima.fl_dado_valido AS fl_temperatura_maxima_dado_valido

    ,temperatura_media.vl_temperatura_media
    ,temperatura_media.fl_dado_ausente AS fl_temperatura_media_dado_ausente
    ,temperatura_media.fl_dado_duvidoso AS fl_temperatura_media_dado_duvidoso
    ,temperatura_media.fl_dado_invalido AS fl_temperatura_media_dado_invalido
    ,temperatura_media.fl_dado_valido AS fl_temperatura_media_dado_valido

    ,temperatura_minima.vl_temperatura_minima
    ,temperatura_minima.fl_dado_ausente AS fl_temperatura_minima_dado_ausente
    ,temperatura_minima.fl_dado_duvidoso AS fl_temperatura_minima_dado_duvidoso
    ,temperatura_minima.fl_dado_invalido AS fl_temperatura_minima_dado_invalido
    ,temperatura_minima.fl_dado_valido AS fl_temperatura_minima_dado_valido

    ,umidade_relativa_maxima.vl_umidade_relativa_maxima
    ,umidade_relativa_maxima.fl_dado_ausente AS fl_umidade_relativa_maxima_dado_ausente
    ,umidade_relativa_maxima.fl_dado_duvidoso AS fl_umidade_relativa_maxima_dado_duvidoso
    ,umidade_relativa_maxima.fl_dado_invalido AS fl_umidade_relativa_maxima_dado_invalido
    ,umidade_relativa_maxima.fl_dado_valido AS fl_umidade_relativa_maxima_dado_valido

    ,umidade_relativa_media.vl_umidade_relativa_media
    ,umidade_relativa_media.fl_dado_ausente AS fl_umidade_relativa_media_dado_ausente
    ,umidade_relativa_media.fl_dado_duvidoso AS fl_umidade_relativa_media_dado_duvidoso
    ,umidade_relativa_media.fl_dado_invalido AS fl_umidade_relativa_media_dado_invalido
    ,umidade_relativa_media.fl_dado_valido AS fl_umidade_relativa_media_dado_valido

    ,umidade_relativa_minima.vl_umidade_relativa_minima
    ,umidade_relativa_minima.fl_dado_ausente AS fl_umidade_relativa_minima_dado_ausente
    ,umidade_relativa_minima.fl_dado_duvidoso AS fl_umidade_relativa_minima_dado_duvidoso
    ,umidade_relativa_minima.fl_dado_invalido AS fl_umidade_relativa_minima_dado_invalido
    ,umidade_relativa_minima.fl_dado_valido AS fl_umidade_relativa_minima_dado_valido

    ,velocidade_vento_2m_maxima.vl_velocidade_vento_2m_maxima
    ,velocidade_vento_2m_maxima.fl_dado_ausente AS fl_velocidade_vento_2m_maxima_dado_ausente
    ,velocidade_vento_2m_maxima.fl_dado_duvidoso AS fl_velocidade_vento_2m_maxima_dado_duvidoso
    ,velocidade_vento_2m_maxima.fl_dado_invalido AS fl_velocidade_vento_2m_maxima_dado_invalido
    ,velocidade_vento_2m_maxima.fl_dado_valido AS fl_velocidade_vento_2m_maxima_dado_valido

    ,velocidade_vento_2m_media.vl_velocidade_vento_2m_media
    ,velocidade_vento_2m_media.fl_dado_ausente AS fl_velocidade_vento_2m_media_dado_ausente
    ,velocidade_vento_2m_media.fl_dado_duvidoso AS fl_velocidade_vento_2m_media_dado_duvidoso
    ,velocidade_vento_2m_media.fl_dado_invalido AS fl_velocidade_vento_2m_media_dado_invalido
    ,velocidade_vento_2m_media.fl_dado_valido AS fl_velocidade_vento_2m_media_dado_valido

    ,velocidade_vento_10m_media.vl_velocidade_vento_10m_media
    ,velocidade_vento_10m_media.fl_dado_ausente AS fl_velocidade_vento_10m_media_dado_ausente
    ,velocidade_vento_10m_media.fl_dado_duvidoso AS fl_velocidade_vento_10m_media_dado_duvidoso
    ,velocidade_vento_10m_media.fl_dado_invalido AS fl_velocidade_vento_10m_media_dado_invalido
    ,velocidade_vento_10m_media.fl_dado_valido AS fl_velocidade_vento_10m_media_dado_valido

FROM fato_estacoes_precipitacao AS precipitacao

LEFT JOIN fato_estacoes_temperatura_maxima AS temperatura_maxima
    ON precipitacao.dt_medicao = temperatura_maxima.dt_medicao
    AND precipitacao.id_estacao = temperatura_maxima.id_estacao

LEFT JOIN fato_estacoes_temperatura_media AS temperatura_media
    ON precipitacao.dt_medicao = temperatura_media.dt_medicao
    AND precipitacao.id_estacao = temperatura_media.id_estacao

LEFT JOIN fato_estacoes_temperatura_minima AS temperatura_minima
    ON precipitacao.dt_medicao = temperatura_minima.dt_medicao
    AND precipitacao.id_estacao = temperatura_minima.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_maxima AS umidade_relativa_maxima
    ON precipitacao.dt_medicao = umidade_relativa_maxima.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_maxima.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_media AS umidade_relativa_media
    ON precipitacao.dt_medicao = umidade_relativa_media.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_media.id_estacao

LEFT JOIN fato_estacoes_umidade_relativa_minima AS umidade_relativa_minima
    ON precipitacao.dt_medicao = umidade_relativa_minima.dt_medicao
    AND precipitacao.id_estacao = umidade_relativa_minima.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_2m_maxima AS velocidade_vento_2m_maxima
    ON precipitacao.dt_medicao = velocidade_vento_2m_maxima.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_2m_maxima.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_2m_media AS velocidade_vento_2m_media
    ON precipitacao.dt_medicao = velocidade_vento_2m_media.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_2m_media.id_estacao

LEFT JOIN fato_estacoes_velocidade_vento_10m_media AS velocidade_vento_10m_media
    ON precipitacao.dt_medicao = velocidade_vento_10m_media.dt_medicao
    AND precipitacao.id_estacao = velocidade_vento_10m_media.id_estacao
                    
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_fato_estacoes_table_name} AS
    SELECT
        *
    FROM prata_fato_estacoes
""")

## ------------ ##
## DIM ESTAÇÕES ##
## ------------ ##

prata_dim_estacoes_table_name = "dim_estacoes"

bronze_dim_estacoes = bronze_conn.execute("SELECT * FROM dim_estacoes").fetch_df()

prata_dim_estacoes = bronze_dim_estacoes.copy()

prata_dim_estacoes['nm_instituicao'] = prata_dim_estacoes['nm_instituicao'].astype(str) \
    .str.replace('\n','') \
    .replace(r'\s+', ' ', regex=True)

prata_dim_estacoes['nm_estacao'] = prata_dim_estacoes['nm_estacao'].astype(str).replace(r'\s+', ' ', regex=True)
    
prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_dim_estacoes_table_name} AS 
    SELECT
        *
    FROM prata_dim_estacoes
    """)

## ------------------------ ##
## PRODUTOS - GPM FINAL RUN ##
## ------------------------ ##
prata_gpm_final_run_table_name = 'fato_produto_gpm_final_run'

bronze_gpm_final_run_precipitacao = bronze_conn.execute('SELECT * FROM fato_produto_gpm_final_run_precipitacao').fetch_df()
prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_gpm_final_run_table_name} AS
    SELECT * FROM bronze_gpm_final_run_precipitacao
                   """)

## ----------------------- ##
## PRODUTOS - GPM LATE RUN ##
## ----------------------- ##
prata_gpm_late_run_table_name = 'fato_produto_gpm_late_run'

bronze_gpm_late_run_precipitacao = bronze_conn.execute('SELECT * FROM fato_produto_gpm_late_run_precipitacao').fetch_df()

prata_conn.execute(
f"""
CREATE OR REPLACE TABLE {prata_gpm_late_run_table_name} AS
SELECT 
    dt_medicao
    ,CAST(lon AS DECIMAL(4,2)) AS lon
    ,CAST(lat AS DECIMAL(4,2)) AS lat
    ,vl_precipitacao
FROM bronze_gpm_late_run_precipitacao
                   """)

## ------------ ##
## PRODUTOS CPC ##
## ------------ ##
prata_cpc_table_name = 'fato_produto_cpc'

prata_cpc_df = bronze_conn.execute(
f"""
SELECT
    cpc_precipitacao.dt_medicao
    ,cpc_precipitacao.lon
    ,cpc_precipitacao.lat
    ,cpc_precipitacao.vl_precipitacao
    ,cpc_temperatura_maxima.vl_temperatura_maxima
    ,cpc_temperatura_minima.vl_temperatura_minima
FROM fato_produto_cpc_precipitacao AS cpc_precipitacao
FULL OUTER JOIN fato_produto_cpc_temperatura_maxima AS cpc_temperatura_maxima
    ON cpc_precipitacao.dt_medicao = cpc_temperatura_maxima.dt_medicao
    AND cpc_precipitacao.lon = cpc_temperatura_maxima.lon
    AND cpc_precipitacao.lat = cpc_temperatura_maxima.lat
FULL OUTER JOIN fato_produto_cpc_temperatura_minima AS cpc_temperatura_minima
    ON cpc_precipitacao.dt_medicao = cpc_temperatura_minima.dt_medicao
    AND cpc_precipitacao.lon = cpc_temperatura_minima.lon
    AND cpc_precipitacao.lat = cpc_temperatura_minima.lat
ORDER BY
    cpc_precipitacao.dt_medicao
    ,cpc_precipitacao.lat
    ,cpc_precipitacao.lon
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_cpc_table_name} AS 
    SELECT
        *
    FROM prata_cpc_df
    """)

## -------------- ##
## PRODUTOS POWER ##
## -------------- ##

prata_power_table_name = 'fato_produto_power'

prata_power_df = bronze_conn.execute(
f"""
SELECT
    power_precipitacao.dt_medicao
    ,power_precipitacao.lat
    ,power_precipitacao.lon
    ,power_precipitacao.vl_precipitacao
    ,power_tempmax.vl_temperatura_maxima_2m
    ,power_tempmedia.vl_temperatura_media_2m
    ,power_tempmin.vl_temperatura_minima_2m
    ,power_umidaderelativa.vl_umidade_relativa_2m
    ,power_pressao.vl_pressao_nivel_superficie
    ,power_irradiancia.vl_irradiancia_allsky
    ,power_direcao_vento_10m.vl_direcao_vento_10m
    ,power_direcao_vento_2m.vl_direcao_vento_2m
    ,power_temporvalho.vl_temperatura_orvalho_2m
    ,power_vento_10m.vl_vento_10m
    ,power_vento_2m.vl_vento_medio_2m
    ,power_vento_2m_max.vl_vento_maximo_2m
    ,power_vento_10m_max.vl_vento_maximo_10m
FROM fato_produto_power_precipitacao AS power_precipitacao
FULL OUTER JOIN fato_produto_power_tempmax AS power_tempmax 
    ON power_precipitacao.dt_medicao = power_tempmax.dt_medicao
    AND power_precipitacao.lat = power_tempmax.lat
    AND power_precipitacao.lon = power_tempmax.lon
FULL OUTER JOIN fato_produto_power_tempmedia AS power_tempmedia
    ON power_precipitacao.dt_medicao = power_tempmedia.dt_medicao
    AND power_precipitacao.lat = power_tempmedia.lat
    AND power_precipitacao.lon = power_tempmedia.lon
FULL OUTER JOIN fato_produto_power_tempmin AS power_tempmin
    ON power_precipitacao.dt_medicao = power_tempmin.dt_medicao
    AND power_precipitacao.lat = power_tempmin.lat
    AND power_precipitacao.lon = power_tempmin.lon
FULL OUTER JOIN fato_produto_power_umidade_2m AS power_umidaderelativa
    ON power_precipitacao.dt_medicao = power_umidaderelativa.dt_medicao
    AND power_precipitacao.lat = power_umidaderelativa.lat
    AND power_precipitacao.lon = power_umidaderelativa.lon
FULL OUTER JOIN fato_produto_power_pressao AS power_pressao
    ON power_precipitacao.dt_medicao = power_pressao.dt_medicao
    AND power_precipitacao.lat = power_pressao.lat
    AND power_precipitacao.lon = power_pressao.lon
FULL OUTER JOIN fato_produto_power_irradiancia_allsky AS power_irradiancia
    ON power_precipitacao.dt_medicao = power_irradiancia.dt_medicao
    AND power_precipitacao.lat = power_irradiancia.lat
    AND power_precipitacao.lon = power_irradiancia.lon
FULL OUTER JOIN fato_produto_power_direcao_vento_10m AS power_direcao_vento_10m
    ON power_precipitacao.dt_medicao = power_direcao_vento_10m.dt_medicao
    AND power_precipitacao.lat = power_direcao_vento_10m.lat
    AND power_precipitacao.lon = power_direcao_vento_10m.lon
FULL OUTER JOIN fato_produto_power_direcao_vento_2m AS power_direcao_vento_2m
    ON power_precipitacao.dt_medicao = power_direcao_vento_2m.dt_medicao
    AND power_precipitacao.lat = power_direcao_vento_2m.lat
    AND power_precipitacao.lon = power_direcao_vento_2m.lon
FULL OUTER JOIN fato_produto_power_temp_orvalho AS power_temporvalho
    ON power_precipitacao.dt_medicao = power_temporvalho.dt_medicao
    AND power_precipitacao.lat = power_temporvalho.lat
    AND power_precipitacao.lon = power_temporvalho.lon
FULL OUTER JOIN fato_produto_power_vento_10m AS power_vento_10m
    ON power_precipitacao.dt_medicao = power_vento_10m.dt_medicao
    AND power_precipitacao.lat = power_vento_10m.lat
    AND power_precipitacao.lon = power_vento_10m.lon
FULL OUTER JOIN fato_produto_power_vento_2m AS power_vento_2m
    ON power_precipitacao.dt_medicao = power_vento_2m.dt_medicao
    AND power_precipitacao.lat = power_vento_2m.lat
    AND power_precipitacao.lon = power_vento_2m.lon
FULL OUTER JOIN fato_produto_power_vento_2m_max AS power_vento_2m_max
    ON power_precipitacao.dt_medicao = power_vento_2m_max.dt_medicao
    AND power_precipitacao.lat = power_vento_2m_max.lat
    AND power_precipitacao.lon = power_vento_2m_max.lon
FULL OUTER JOIN fato_produto_power_vento_10m_max AS power_vento_10m_max
    ON power_precipitacao.dt_medicao = power_vento_10m_max.dt_medicao
    AND power_precipitacao.lat = power_vento_10m_max.lat
    AND power_precipitacao.lon = power_vento_10m_max.lon
"""
).fetch_df()


prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_power_table_name} AS 
    SELECT
        *
    FROM prata_power_df
    """)

## --------------- ##
## PRODUTOS CHIRPS ##
## --------------- ##

prata_chirps_table_name = 'fato_produto_chirps'

prata_chirps_df = bronze_conn.execute(
f"""
SELECT
    chirps.dt_medicao
    ,chirps.lat
    ,chirps.lon
    ,chirps.vl_precipitacao
FROM fato_produto_chirps_precipitacao AS chirps
""").fetch_df()

prata_conn.execute(f"""
    CREATE OR REPLACE TABLE {prata_chirps_table_name} AS 
    SELECT
        *
    FROM prata_chirps_df
    """)

## ------------------------------------------------------------------------------------------------ ##
## ---------------------------------------- PRATA TO GOLD ----------------------------------------- ##
## ------------------------------------------------------------------------------------------------ ##

prata_conn = duckdb.connect("prata_db")
prata_conn.execute("INSTALL spatial; LOAD spatial")
ouro_conn = duckdb.connect("ouro_db")
ouro_conn.execute("INSTALL spatial; LOAD spatial")

## ------------ ##
## DIM ESTAÇÕES ##
## ------------ ##

ouro_dim_estacoes_table_name = 'dim_estacoes'

estacoes_com_medicao_de_precipitacao = prata_conn.execute('select id_estacao from fato_estacoes group by id_estacao having count(vl_precipitacao)>0').fetch_df()['id_estacao'].tolist()

estacoes_com_medicao_de_precipitacao_str = ','.join([str(id_estacao) for id_estacao in estacoes_com_medicao_de_precipitacao])

ouro_dim_estacoes_df = prata_conn.execute(f'select * from dim_estacoes where id_estacao in ({estacoes_com_medicao_de_precipitacao_str})').fetch_df()

ouro_conn.execute(
f"""
CREATE OR REPLACE TABLE {ouro_dim_estacoes_table_name} AS
SELECT * FROM ouro_dim_estacoes_df
""")

## -------------------------- ##
## FATO ESTAÇÕES PRECIPITAÇÃO ##
## -------------------------- ##

ouro_fato_estacoes_precipitacao_table_name = 'fato_estacoes_precipitacao'
ouro_fato_estacoes_df = prata_conn.execute(
f"""
SELECT 
    dt_medicao
    ,id_estacao
    ,vl_precipitacao
FROM fato_estacoes
WHERE 1=1
    AND fl_precipitacao_dado_valido
""").fetch_df()

ouro_conn.execute(
f"""
CREATE OR REPLACE TABLE {ouro_fato_estacoes_precipitacao_table_name} AS
SELECT * FROM ouro_fato_estacoes_df
""")


## ----------------------------- ##
## FATO ESTAÇÕES LATLON PRODUTOS ##
## ----------------------------- ##

ouro_fato_estacoes_latlon_table_name = 'fato_estacoes_latlon_produtos_df'

produtos = ['chirps','cpc','power','gpm_final_run','gpm_late_run']

possible_lon = {produto: [round(v,3) for v in prata_conn.execute(f"SELECT DISTINCT lon FROM fato_produto_{produto} ORDER BY lon").fetch_df()['lon'].tolist()] for produto in produtos}
possible_lat = {produto: [round(v,3) for v in prata_conn.execute(f"SELECT DISTINCT lat FROM fato_produto_{produto} ORDER BY lon").fetch_df()['lat'].tolist()] for produto in produtos}

ouro_dim_estacoes_df = prata_conn.execute('select * from dim_estacoes').fetch_df()

fato_estacoes_latlon_produtos_df = encontrar_coordenadas_mais_proximas(
    ouro_dim_estacoes_df, 
    produtos, 
    possible_lat, 
    possible_lon
)

ouro_conn.execute(
f"""
CREATE OR REPLACE TABLE {ouro_fato_estacoes_latlon_table_name} AS
SELECT * FROM fato_estacoes_latlon_produtos_df
""")


## -------------------- ##
## MATRIZ DE DISTÂNCIAS ##
## -------------------- ##

ouro_matriz_distancias_table_name  = "fato_estacoes_distancia"

estacoes_list = ouro_conn.execute("SELECT DISTINCT id_estacao FROM dim_estacoes").fetch_df()['id_estacao'].tolist()

estacoes_list_str = ','.join([str(id_estacao) for id_estacao in estacoes_list])

ouro_matriz_distancias = prata_conn.execute(f"""
    SELECT * FROM fato_estacoes_distancia
    WHERE id_estacao_base IN ({estacoes_list_str})
    AND id_estacao_candidata IN ({estacoes_list_str})
""").fetch_df()

ouro_conn.execute(
f"""
CREATE OR REPLACE TABLE {ouro_matriz_distancias_table_name} AS
SELECT * FROM ouro_matriz_distancias
""")


## --------------------- ##
## MATRIZ DE INTERSEÇÕES ##
## --------------------- ##

ouro_matriz_intersecao_table_name  = "fato_estacoes_intersecao"

estacoes_list = ouro_conn.execute("SELECT DISTINCT id_estacao FROM dim_estacoes").fetch_df()['id_estacao'].tolist()

ouro_matriz_intersecao = prata_conn.execute(f"""
    SELECT * FROM fato_estacoes_intersecao
    WHERE id_estacao_base IN ({estacoes_list_str})
    AND id_estacao_candidata IN ({estacoes_list_str})
""").fetch_df()

ouro_conn.execute(
f"""
CREATE OR REPLACE TABLE {ouro_matriz_intersecao_table_name} AS
SELECT * FROM ouro_matriz_intersecao
""")


## --------------------- ##
## MATRIZ DE CORRELAÇÕES ## 
## --------------------- ##

ouro_matriz_correlacoes_table_name  = "fato_estacoes_correlacao"

print("Carregando dados da tabela fato_estacoes...")
df = prata_conn.execute("""
    SELECT id_estacao, dt_medicao, vl_precipitacao
    FROM fato_estacoes
""").fetch_df()
print(f"Total de linhas carregadas: {len(df)}")
print(f"Total de estações distintas: {df['id_estacao'].nunique()}")
print(f"Total de datas distintas: {df['dt_medicao'].nunique()}")

print("Pivotando dados...")
pivot_df = df.pivot_table(
    index="dt_medicao",
    columns="id_estacao",
    values="vl_precipitacao"
)
print(f"Dimensões da matriz pivotada: {pivot_df.shape}")

print("Calculando correlações...")
cor_matrix = pivot_df.corr(method="pearson").fillna(0)
print("Correlação calculada.")

print("Convertendo matriz de correlação para formato longo...")

cor_matrix.columns.name = None
cor_matrix.index.name = None

correlacoes_df = (
    cor_matrix
    .stack()
    .reset_index()
    .rename(columns={
        "level_0": "id_estacao_base",
        "level_1": "id_estacao_vizinha",
        0: "correlacao"
    })
)
correlacoes_df = correlacoes_df[
    correlacoes_df["id_estacao_base"] != correlacoes_df["id_estacao_vizinha"]
].reset_index(drop=True)
print(f"Total de pares com correlação: {len(correlacoes_df)}")

print("Calculando número de datas em comum entre pares...")
presence_matrix = ~pivot_df.isna()
n_pontos = presence_matrix.T.dot(presence_matrix).astype(int)

n_pontos_df = (
    n_pontos
    .stack()
    .reset_index()
    .rename(columns={
        "level_0": "id_estacao_base",
        "level_1": "id_estacao_vizinha",
        0: "n_pontos_comuns"
    })
)
n_pontos_df = n_pontos_df[
    n_pontos_df["id_estacao_base"] != n_pontos_df["id_estacao_vizinha"]
].reset_index(drop=True)
print(f"Total de pares com interseção de datas: {len(n_pontos_df)}")

print("Juntando correlação com número de datas em comum...")
ouro_matriz_correlacoes_df = correlacoes_df.merge(
    n_pontos_df,
    on=["id_estacao_base", "id_estacao_vizinha"]
).rename(columns={
    'id_estacao_vizinha':'id_estacao_candidata'
})
print("Tabela final construída com sucesso!")

ouro_matriz_correlacoes_df = ouro_matriz_correlacoes_df.ren

ouro_conn.execute(f"""
    CREATE OR REPLACE TABLE {ouro_matriz_correlacoes_table_name} AS
    SELECT
        *
    FROM ouro_matriz_correlacoes_df
""")

## ---------------------------------- ##
## FATO_ESTAÇÕES_BASE_FILA_PRIORIDADE ## 
## ---------------------------------- ##

ouro_fato_estacoes_fila_prioridade_table_name  = "fato_estacoes_base_fila_prioridade"

fato_estacoes_fila_prioridade_df = ouro_conn.execute(
"""
SELECT
    correlacao.id_estacao_base
    ,correlacao.id_estacao_candidata
    ,correlacao.correlacao
    ,intersecao.pct_intersecao_precipitacao
    ,distancia.vl_distancia_km
FROM fato_estacoes_correlacao AS correlacao
JOIN fato_estacoes_intersecao AS intersecao
    ON correlacao.id_estacao_base = intersecao.id_estacao_base
    AND correlacao.id_estacao_candidata = intersecao.id_estacao_candidata
JOIN fato_estacoes_distancia AS distancia
    ON correlacao.id_estacao_base = distancia.id_estacao_base
    AND correlacao.id_estacao_candidata = distancia.id_estacao_candidata
""").fetch_df()

ouro_conn.execute(f"""
    CREATE OR REPLACE TABLE {ouro_fato_estacoes_fila_prioridade_table_name} AS
    SELECT
        *
    FROM fato_estacoes_fila_prioridade_df
""")



