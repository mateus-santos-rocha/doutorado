import zipfile
import os
import shutil

def descompactar_e_mover(caminho_zip, pasta_destino):
    pasta_temporaria = 'temp_extracao'
    os.makedirs(pasta_temporaria, exist_ok=True)

    # Descompacta o arquivo ZIP
    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
        zip_ref.extractall(pasta_temporaria)

    # Move todos os arquivos da pasta temporária para a pasta de destino
    os.makedirs(pasta_destino, exist_ok=True)
    for nome_arquivo in os.listdir(pasta_temporaria):
        origem = os.path.join(pasta_temporaria, nome_arquivo)
        destino = os.path.join(pasta_destino, nome_arquivo)
        shutil.move(origem, destino)

    # Remove a pasta temporária
    os.rmdir(pasta_temporaria)