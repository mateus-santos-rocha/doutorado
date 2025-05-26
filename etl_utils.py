import zipfile
import os
import shutil
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


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

    return df


def encontrar_coordenadas_mais_proximas(df_estacoes, produtos, possible_lat, possible_lon):
    """
    Encontra as coordenadas de latitude e longitude mais próximas de produtos
    para cada estação em um DataFrame.

    Args:
        df_estacoes (pd.DataFrame): DataFrame com colunas 'id_estacao', 'latitude', 'longitude'.
        produtos (list): Lista de nomes de produtos (strings).
        possible_lat (dict): Dicionário onde as chaves são nomes de produtos e os
                             valores são listas de latitudes possíveis para esse produto.
        possible_lon (dict): Dicionário onde as chaves são nomes de produtos e os
                             valores são listas de longitudes possíveis para esse produto.

    Returns:
        pd.DataFrame: DataFrame original com colunas adicionais para as
                      latitudes e longitudes mais próximas de cada produto.
                      Ex: lat_produtoX, lon_produtoX.
    """
    df_resultado = df_estacoes.copy()

    for produto in produtos:
        lat_produto_col = f'lat_{produto}'
        lon_produto_col = f'lon_{produto}'
        
        df_resultado[lat_produto_col] = np.nan
        df_resultado[lon_produto_col] = np.nan

        if produto not in possible_lat or produto not in possible_lon:
            print(f"Aviso: Coordenadas possíveis para o produto '{produto}' não encontradas. Pulando este produto.")
            continue

        latitudes_produto = np.array(possible_lat[produto])
        longitudes_produto = np.array(possible_lon[produto])

        if latitudes_produto.size == 0 or longitudes_produto.size == 0:
            print(f"Aviso: Lista de latitudes ou longitudes vazia para o produto '{produto}'. Pulando este produto.")
            continue
            
        product_lon_grid, product_lat_grid = np.meshgrid(longitudes_produto, latitudes_produto)
        
        product_coords = np.vstack([product_lat_grid.ravel(), product_lon_grid.ravel()]).T

        for index, row in df_estacoes.iterrows():
            est_lat = row['latitude']
            est_lon = row['longitude']
            
            station_coord = np.array([[est_lat, est_lon]])

            diffs = station_coord - product_coords 

            dist_sq = np.sum(diffs**2, axis=1)

            idx_mais_proximo = np.argmin(dist_sq)

            melhor_lat_produto = product_coords[idx_mais_proximo, 0]
            melhor_lon_produto = product_coords[idx_mais_proximo, 1]
            
            df_resultado.loc[index, lat_produto_col] = melhor_lat_produto
            df_resultado.loc[index, lon_produto_col] = melhor_lon_produto
            
    return df_resultado

def calcular_intersecoes_estacoes_vizinhas(id_estacao_base,fato_estacoes,datas_por_estacao):
    estacoes = set(fato_estacoes['id_estacao'])
    intersecoes_dict = {}
    intersecoes_dict[id_estacao_base] = {}
    datas_estacao_base = datas_por_estacao[id_estacao_base] # Pega o set pré-calculado

    # Evita ZeroDivisionError se a estação base não tiver dados
    if not datas_estacao_base:
        for id_estacao_candidata in list(estacoes.difference({id_estacao_base})):
            intersecoes_dict[id_estacao_candidata] = 0.0
        return None # Pula para a próxima estação base

    for id_estacao_candidata in tqdm(list(estacoes.difference({id_estacao_base})), desc=f"Calculando para {id_estacao_base}", leave=False):
        datas_estacao_candidata = datas_por_estacao[id_estacao_candidata] # Pega o set pré-calculado

        pct_intersecao_base_candidata = len(datas_estacao_base.intersection(datas_estacao_candidata)) * 100 / len(datas_estacao_base)
        intersecoes_dict[id_estacao_candidata] = pct_intersecao_base_candidata
    return intersecoes_dict