import zipfile
import os
import shutil
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np

def descompactar_e_mover(caminho_zip, pasta_destino):
    pasta_temporaria = 'temp_extracao'
    os.makedirs(pasta_temporaria, exist_ok=True)

    # Tenta extrair arquivo por arquivo
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        for membro in zip_ref.namelist():
            try:
                zip_ref.extract(membro, pasta_temporaria)
            except Exception as e:
                print(f'❌ Erro ao extrair {membro}: {e}')

    # Move todos os arquivos da pasta temporária para a pasta de destino
    os.makedirs(pasta_destino, exist_ok=True)
    for nome_arquivo in os.listdir(pasta_temporaria):
        origem = os.path.join(pasta_temporaria, nome_arquivo)
        destino = os.path.join(pasta_destino, nome_arquivo)
        shutil.move(origem, destino)

    # Remove a pasta temporária (se estiver vazia)
    try:
        os.rmdir(pasta_temporaria)
    except OSError:
        print('⚠️ Pasta temporária não vazia ou erro ao remover. Você pode limpar manualmente.')

def geotiff_to_dataframe(filepath, min_lon, max_lon, min_lat, max_lat, band=1, column_name='valor'):
    """
    Lê um arquivo GeoTIFF e retorna um DataFrame Pandas com as coordenadas e valores filtrados por uma janela geográfica.

    Parâmetros:
    - filepath (str): Caminho para o arquivo GeoTIFF.
    - min_lon, max_lon (float): Longitude mínima e máxima.
    - min_lat, max_lat (float): Latitude mínima e máxima.
    - band (int): Banda do raster a ser lida (padrão = 1).
    - column_name (str): Nome da coluna para os valores dos pixels.

    Retorna:
    - pd.DataFrame com colunas ['lon', 'lat', column_name]
    """
    with rasterio.open(filepath) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        data = src.read(band, window=window)
        transform = src.window_transform(window)

        height, width = data.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        
        df = pd.DataFrame({
            'lon': np.array(xs).flatten(),
            'lat': np.array(ys).flatten(),
            column_name: data.flatten()
        })

        # if src.nodata is not None:
        #     df = df[df[column_name] != src.nodata]

    return df