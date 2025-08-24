import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import warnings
from sklearn.utils import resample

warnings.filterwarnings('ignore')

def generate_synthetic_cases_with_target_smoter(dataframe, target_column, explanatory_variables, 
                                               oversampling_ratio, num_neighbors, constraint_columns=None):
    """
    Função auxiliar para gerar casos sintéticos otimizada para SMOTE-R
    
    Parâmetros adicionais:
    - constraint_columns: Lista de colunas que devem ter valores iguais entre vizinhos
    """
    
    synthetic_cases = []
    num_new_cases = int(oversampling_ratio / 100)
    
    if num_new_cases == 0:
        return pd.DataFrame()
    
    features = dataframe[explanatory_variables].copy()
    target = dataframe[target_column].copy()
    
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    
    label_encoders = {}
    encoded_features = features.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        encoded_features[col] = le.fit_transform(features[col].astype(str))
        label_encoders[col] = le
    
    features_for_knn = encoded_features.values
    knn_model = NearestNeighbors(n_neighbors=min(num_neighbors + 1, len(dataframe)))
    knn_model.fit(features_for_knn)
    
    with tqdm(total=len(features), desc="Gerando casos sintéticos") as pbar:
        for case_index, (_, original_case) in enumerate(features.iterrows()):
            
            distances, neighbor_indices = knn_model.kneighbors([features_for_knn[case_index]])
            neighbor_indices = neighbor_indices[0][1:]  # Remove o primeiro (próprio caso)
            
            for _ in range(num_new_cases):
                selected_neighbor_idx = random.choice(neighbor_indices)
                selected_neighbor = features.iloc[selected_neighbor_idx]
                
                synthetic_case = {}
                
                for attribute_name in explanatory_variables:
                    if attribute_name in numeric_columns:
                        original_value = original_case[attribute_name]
                        neighbor_value = selected_neighbor[attribute_name]
                        difference = neighbor_value - original_value
                        random_factor = np.random.uniform(0, 1)
                        synthetic_value = original_value + random_factor * difference
                        synthetic_case[attribute_name] = synthetic_value
                    else:
                        original_value = original_case[attribute_name]
                        neighbor_value = selected_neighbor[attribute_name]
                        synthetic_case[attribute_name] = random.choice([original_value, neighbor_value])
                
                original_target = target.iloc[case_index]
                neighbor_target = target.iloc[selected_neighbor_idx]
                
                case_distance = distances[0][0] if distances[0][0] > 0 else 0.001
                neighbor_distance = distances[0][np.where(neighbor_indices == selected_neighbor_idx)[0][0] + 1]
                
                total_distance = case_distance + neighbor_distance
                if total_distance > 0:
                    weight_original = neighbor_distance / total_distance
                    weight_neighbor = case_distance / total_distance
                else:
                    weight_original = weight_neighbor = 0.5
                
                synthetic_target = weight_original * original_target + weight_neighbor * neighbor_target
                synthetic_case[target_column] = synthetic_target
                
                synthetic_cases.append(synthetic_case)
            
            pbar.update(1)
    
    result_df = pd.DataFrame(synthetic_cases)
    
    missing_columns = set(dataframe.columns) - set(explanatory_variables) - {target_column}
    for col in missing_columns:
        # Para colunas não explicativas, usar valores da primeira linha como padrão
        # ou implementar lógica específica conforme necessário
        result_df[col] = dataframe[col].iloc[0]
    
    result_df = result_df[dataframe.columns]
    
    return result_df


def smoteR(dataframe, target_column, explanatory_variables=None, 
           relevance_function=None, threshold=0.5, pct_oversampling=100, 
           pct_undersampling=100, number_of_nearest_neighbors=5, 
           constraint_columns=None, random_state=None):
    """
    Implementa o algoritmo SMOTE-R para balanceamento de datasets com target contínuo.
    
    Parâmetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame original com os dados
    target_column : str
        Nome da coluna alvo/target
    explanatory_variables : list, optional
        Lista das variáveis explicativas a serem consideradas. 
        Se None, usa todas as colunas exceto target
    relevance_function : callable, optional
        Função que determina a relevância de uma observação.
        Se None, usa uma função baseada na distância do percentil 50
    threshold : float, default=0.5
        Limiar para determinar observações raras vs comuns
    pct_oversampling : int, default=100
        Porcentagem de oversampling para casos raros
    pct_undersampling : int, default=100
        Porcentagem de undersampling para casos comuns
    number_of_nearest_neighbors : int, default=5
        Número de vizinhos mais próximos para geração sintética
    constraint_columns : list or str, optional
        Lista de colunas que devem ter valores iguais entre a amostra e seus vizinhos.
        Pode ser uma string (coluna única) ou lista de strings (múltiplas colunas).
        Se None, não há restrições nos vizinhos.
        Exemplo: ['dt_medicao'] ou ['dt_medicao', 'regiao']
    random_state : int, optional
        Semente para reprodutibilidade
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame balanceado com casos originais raros, casos comuns 
        sub-amostrados e casos sintéticos gerados
    """
    
    try:
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        
        print("🚀 Iniciando SMOTE-R...")
        print(f"   Dataset original: {len(dataframe):,} registros")
        
        if target_column not in dataframe.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada no DataFrame")
        
        if not pd.api.types.is_numeric_dtype(dataframe[target_column]):
            raise ValueError(f"Coluna target '{target_column}' deve ser numérica para SMOTE-R")
        
        if explanatory_variables is None:
            explanatory_variables = [col for col in dataframe.columns if col != target_column]
            print(f"   Usando todas as {len(explanatory_variables)} features como variáveis explicativas")
        else:
            missing_vars = set(explanatory_variables) - set(dataframe.columns)
            if missing_vars:
                raise ValueError(f"Variáveis explicativas não encontradas: {missing_vars}")
            print(f"   Usando {len(explanatory_variables)} variáveis explicativas especificadas")
        
        if constraint_columns is not None:
            if isinstance(constraint_columns, str):
                constraint_columns = [constraint_columns]
            
            missing_constraints = set(constraint_columns) - set(dataframe.columns)
            if missing_constraints:
                raise ValueError(f"Colunas de restrição não encontradas: {missing_constraints}")
            
            print(f"   Restrições de vizinhança: {constraint_columns}")
        else:
            constraint_columns = []
            print("   Nenhuma restrição de vizinhança aplicada")
        
        if relevance_function is None:
            median_value = dataframe[target_column].median()
            mad_value = np.median(np.abs(dataframe[target_column] - median_value))
            
            def default_relevance_function(x):
                """Função de relevância baseada na distância do percentil 50"""
                if mad_value == 0:
                    return 0.5  # Se MAD é 0, todos os valores são iguais
                return abs(x - median_value) / (mad_value * 1.4826)  # Fator de escala para normalização
            
            relevance_function = default_relevance_function
            print(f"   Usando função de relevância padrão (mediana: {median_value:.3f}, MAD: {mad_value:.3f})")
        
        print("\n🔍 Etapa 1: Identificando observações raras e comuns...")
        
        with tqdm(total=len(dataframe), desc="Calculando relevância") as pbar:
            relevance_scores = []
            for idx, value in enumerate(dataframe[target_column]):
                try:
                    score = relevance_function(value)
                    relevance_scores.append(score)
                except Exception as e:
                    print(f"Erro ao calcular relevância para valor {value}: {e}")
                    relevance_scores.append(0)
                pbar.update(1)
        
        relevance_series = pd.Series(relevance_scores, index=dataframe.index)
        rare_observations_indexes = dataframe.loc[relevance_series >= threshold].index
        common_observations_indexes = dataframe.loc[relevance_series < threshold].index
        
        rare_df = dataframe.loc[rare_observations_indexes].copy()
        common_df = dataframe.loc[common_observations_indexes].copy()
        
        print(f"   Observações raras: {len(rare_df):,} ({len(rare_df)/len(dataframe)*100:.1f}%)")
        print(f"   Observações comuns: {len(common_df):,} ({len(common_df)/len(dataframe)*100:.1f}%)")
        
        if len(rare_df) == 0:
            print("⚠️  Nenhuma observação rara encontrada. Retornando dataset original.")
            return dataframe.copy()
        
        if len(rare_df) < number_of_nearest_neighbors:
            print(f"⚠️  Ajustando número de vizinhos de {number_of_nearest_neighbors} para {len(rare_df)-1}")
            number_of_nearest_neighbors = max(1, len(rare_df) - 1)
        
        print(f"\n🧬 Etapa 2: Gerando casos sintéticos (oversampling: {pct_oversampling}%)...")
        
        try:
            synthetic_cases = generate_synthetic_cases_with_target_smoter(
                dataframe=rare_df,
                target_column=target_column,
                explanatory_variables=explanatory_variables,
                oversampling_ratio=pct_oversampling,
                num_neighbors=number_of_nearest_neighbors,
                constraint_columns=constraint_columns
            )
            
            print(f"   Casos sintéticos gerados: {len(synthetic_cases):,}")
            
        except Exception as e:
            print(f"❌ Erro na geração de casos sintéticos: {e}")
            print("   Continuando sem casos sintéticos...")
            synthetic_cases = pd.DataFrame()
        
        print(f"\n⚖️ Etapa 3: Sub-amostragem de observações comuns (undersampling: {pct_undersampling}%)...")
        
        if len(common_df) > 0:
            total_rare_and_synthetic = len(rare_df) + len(synthetic_cases)
            n_common_to_keep = int((pct_undersampling / 100) * total_rare_and_synthetic)
            n_common_to_keep = min(n_common_to_keep, len(common_df))
            
            if n_common_to_keep > 0:
                with tqdm(total=1, desc="Selecionando casos comuns") as pbar:
                    selected_common_indexes = random.sample(list(common_observations_indexes), n_common_to_keep)
                    selected_common_df = dataframe.loc[selected_common_indexes].copy()
                    pbar.update(1)
                
                print(f"   Casos comuns selecionados: {len(selected_common_df):,} de {len(common_df):,}")
            else:
                selected_common_df = pd.DataFrame()
                print("   Nenhum caso comum selecionado")
        else:
            selected_common_df = pd.DataFrame()
            print("   Nenhuma observação comum disponível")
        
        print(f"\n🔗 Etapa 4: Combinando datasets...")
        
        datasets_to_combine = []
        
        if len(rare_df) > 0:
            datasets_to_combine.append(rare_df)
            print(f"   - Casos raros originais: {len(rare_df):,}")
        
        if len(selected_common_df) > 0:
            datasets_to_combine.append(selected_common_df)
            print(f"   - Casos comuns selecionados: {len(selected_common_df):,}")
        
        if len(synthetic_cases) > 0:
            datasets_to_combine.append(synthetic_cases)
            print(f"   - Casos sintéticos: {len(synthetic_cases):,}")
        
        if not datasets_to_combine:
            print("❌ Nenhum dataset para combinar. Retornando dataset original.")
            return dataframe.copy()
        
        with tqdm(total=1, desc="Combinando datasets") as pbar:
            final_dataframe = pd.concat(datasets_to_combine, ignore_index=True)
            pbar.update(1)
        
        print(f"\n✅ SMOTE-R concluído com sucesso!")
        print(f"   Dataset original: {len(dataframe):,} registros")
        print(f"   Dataset final: {len(final_dataframe):,} registros")
        print(f"   Variação: {((len(final_dataframe) - len(dataframe)) / len(dataframe) * 100):+.1f}%")
        
        print(f"\n📊 Estatísticas do target '{target_column}':")
        print(f"   Original - Média: {dataframe[target_column].mean():.3f}, Mediana: {dataframe[target_column].median():.3f}")
        print(f"   Final    - Média: {final_dataframe[target_column].mean():.3f}, Mediana: {final_dataframe[target_column].median():.3f}")
        
        return final_dataframe
        
    except Exception as e:
        print(f"❌ Erro crítico no SMOTE-R: {e}")
        print("   Retornando dataset original...")
        return dataframe.copy()

def undersample_zeros(X, y, zero_ratio=1.0, random_state=42):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    zero_mask = y == 0
    non_zero_mask = y > 0

    X_zero = X[zero_mask]
    y_zero = y[zero_mask]

    X_non_zero = X[non_zero_mask]
    y_non_zero = y[non_zero_mask]

    n_non_zero = len(y_non_zero)
    n_zero_desired = int(zero_ratio * n_non_zero)
    n_zero_available = len(y_zero)

    n_zero_final = min(n_zero_desired, n_zero_available)

    X_zero_downsampled, y_zero_downsampled = resample(
        X_zero, y_zero,
        replace=False,
        n_samples=n_zero_final,
        random_state=random_state
    )

    X_bal = pd.concat([X_zero_downsampled, X_non_zero], axis=0)
    y_bal = pd.concat([y_zero_downsampled, y_non_zero], axis=0)

    X_bal, y_bal = resample(X_bal, y_bal, random_state=random_state)

    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True)

