from modelling_utils.model_management import save_model_and_comparison
from modelling_utils.sampling import undersample_zeros,smoteR
from modelling_utils.preprocessing import split_com_sem_vizinha,particao_por_estacao
from comparison_utils import compute_comparison_df
from tqdm.notebook import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os

def generate_X_y_train_test(abt_estacoes_vizinhas, usar_n_estacoes_vizinhas=0, 
                           zero_undersampling_ratio=None, smote_oversampling=False, 
                           smote_threshold=0.5, smote_pct_oversampling=100, 
                           smote_pct_undersampling=100, smote_k_neighbors=5,
                           smote_constraint_columns=None, smote_relevance_function=None,
                           smote_explanatory_variables=None,
                           percent_datetime_partitioning_split=0.7,
                           random_state=None):
    """
    Gera conjuntos de treino e teste a partir de dados de estações meteorológicas.
    
    Esta função processa um DataFrame com dados de estações meteorológicas e suas
    estações vizinhas, criando conjuntos de treino e teste para modelagem preditiva
    de precipitação. Oferece opções para incluir dados de estações vizinhas,
    balanceamento de dados e particionamento temporal.
    
    Parameters
    ----------
    abt_estacoes_vizinhas : pd.DataFrame
        DataFrame contendo os dados das estações meteorológicas e suas vizinhas.
        Deve conter a coluna 'vl_precipitacao' como variável target.
        
    usar_n_estacoes_vizinhas : int, optional, default=0
        Número de estações vizinhas a incluir no dataset. Se 0, não inclui
        dados de estações vizinhas. Deve ser >= 0.
        
    zero_undersampling_ratio : float, optional, default=None
        Proporção de registros com precipitação zero em relação aos registros
        com precipitação > 0 no conjunto de treino. Se None, não aplica undersampling.
        Exemplos:
        - 1.0: mantém mesmo número de zeros e não-zeros (balanceamento 50/50)
        - 0.5: mantém metade de zeros em relação aos não-zeros
        - 2.0: mantém o dobro de zeros em relação aos não-zeros
        Deve ser > 0.
        
    smote_oversampling : bool, optional, default=False
        Se True, aplica técnica SMOTE-R para oversampling de casos raros.
        
    smote_threshold : float, optional, default=0.5
        Limiar para determinar observações raras vs comuns no SMOTE-R.
        
    smote_pct_oversampling : int, optional, default=100
        Porcentagem de oversampling para casos raros no SMOTE-R.
        
    smote_pct_undersampling : int, optional, default=100
        Porcentagem de undersampling para casos comuns no SMOTE-R.
        
    smote_k_neighbors : int, optional, default=5
        Número de vizinhos mais próximos para geração sintética no SMOTE-R.
        
    smote_constraint_columns : list or str, optional, default=None
        Lista de colunas que devem ter valores iguais entre a amostra e seus vizinhos
        no SMOTE-R. Exemplo: ['dt_medicao'] ou ['dt_medicao', 'regiao'].
        
    smote_relevance_function : callable, optional, default=None
        Função customizada que determina a relevância de uma observação no SMOTE-R.
        Se None, usa função padrão baseada na distância da mediana.
        
    percent_datetime_partitioning_split : float, optional, default=0.7
        Percentual dos dados para treino na partição temporal.
        Deve estar entre 0 e 1.
        
    random_state : int, optional, default=None
        Semente para reprodutibilidade em operações aleatórias.
    
    Returns
    -------
    tuple
        Tupla contendo (X_train, X_test, y_train, y_test):
        - X_train : pd.DataFrame - Features de treino
        - X_test : pd.DataFrame - Features de teste  
        - y_train : pd.Series - Target de treino
        - y_test : pd.Series - Target de teste
    
    Raises
    ------
    ValueError
        Se os parâmetros estiverem fora dos limites válidos ou se colunas
        obrigatórias estiverem ausentes.
        
    KeyError
        Se colunas esperadas não existirem no DataFrame de entrada.
        
    TypeError
        Se os tipos dos parâmetros não forem os esperados.
    
    Examples
    --------
    >>> # Uso básico sem estações vizinhas
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(df_estacoes)
    
    >>> # Balanceamento 50/50 (mesmo número de zeros e não-zeros)
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(
    ...     df_estacoes, 
    ...     zero_undersampling_ratio=1.0,
    ...     random_state=42
    ... )
    
    >>> # Incluindo 3 estações vizinhas com menos zeros que não-zeros
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(
    ...     df_estacoes, 
    ...     usar_n_estacoes_vizinhas=3,
    ...     zero_undersampling_ratio=0.5,  # metade de zeros em relação aos não-zeros
    ...     smote_oversampling=True,
    ...     smote_threshold=0.6,
    ...     smote_constraint_columns=['dt_medicao'],
    ...     random_state=42
    ... )
    
    >>> # Mantendo mais zeros que não-zeros
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(
    ...     df_estacoes, 
    ...     zero_undersampling_ratio=2.0,  # dobro de zeros em relação aos não-zeros
    ...     random_state=42
    ... )
    
    Notes
    -----
    - A função assume que existe uma função `particao_por_estacao` disponível
    - A função assume que existe uma função `undersample_zeros` disponível
    - A função assume que existe uma função `smoteR` disponível quando smote_oversampling=True
    - O undersampling é aplicado apenas no conjunto de treino, não afetando o conjunto de teste
    """
    
    
    try:
        if not isinstance(usar_n_estacoes_vizinhas, int) or usar_n_estacoes_vizinhas < 0:
            raise ValueError("usar_n_estacoes_vizinhas deve ser um inteiro >= 0")
        
        if zero_undersampling_ratio is not None:
            if not isinstance(zero_undersampling_ratio, (int, float)) or zero_undersampling_ratio <= 0:
                raise ValueError("zero_undersampling_ratio deve ser None ou um número > 0")
        
        if not isinstance(percent_datetime_partitioning_split, (int, float)) or not (0 < percent_datetime_partitioning_split < 1):
            raise ValueError("percent_datetime_partitioning_split deve ser um número entre 0 e 1")
        
        if not isinstance(smote_threshold, (int, float)) or not (0 <= smote_threshold <= 1):
            raise ValueError("smote_threshold deve ser um número entre 0 e 1")
        
        if not isinstance(smote_pct_oversampling, int) or smote_pct_oversampling < 0:
            raise ValueError("smote_pct_oversampling deve ser um inteiro >= 0")
        
        if not isinstance(smote_pct_undersampling, int) or smote_pct_undersampling < 0:
            raise ValueError("smote_pct_undersampling deve ser um inteiro >= 0")
        
        if not isinstance(smote_k_neighbors, int) or smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors deve ser um inteiro >= 1")
        
        if abt_estacoes_vizinhas.empty:
            raise ValueError("DataFrame de entrada não pode estar vazio")
        
        if 'vl_precipitacao' not in abt_estacoes_vizinhas.columns:
            raise KeyError("Coluna 'vl_precipitacao' não encontrada no DataFrame")
        
        print(f"📊 Iniciando processamento com {len(abt_estacoes_vizinhas)} registros...")
        
        abt = abt_estacoes_vizinhas[[c for c in abt_estacoes_vizinhas.columns if 'vizinha' not in c]].copy()
        print(f"🏭 Dataset base criado com {abt.shape[1]} colunas")
        
        if usar_n_estacoes_vizinhas > 0:
            print(f"🌐 Incluindo dados de {usar_n_estacoes_vizinhas} estação(ões) vizinha(s)...")
            
            vizinhas_columns_prefix = [
                'vl_correlacao_estacao_vizinha_{i_vizinha}',
                'pct_intersecao_precipitacao_vizinha_{i_vizinha}',
                'vl_distancia_km_vizinha_{i_vizinha}',
                'vl_prioridade_vizinha_{i_vizinha}',
                'vl_precipitacao_vizinha_{i_vizinha}'
            ]
            
            for i in tqdm(range(1, usar_n_estacoes_vizinhas + 1), desc="Adicionando estações vizinhas"):
                vizinha_columns = [col.format(i_vizinha=i) for col in vizinhas_columns_prefix]
                
                missing_cols = [col for col in vizinha_columns if col not in abt_estacoes_vizinhas.columns]
                if missing_cols:
                    print(f"⚠️  Aviso: Colunas não encontradas para estação vizinha {i}: {missing_cols}")
                    continue
                
                for col in vizinha_columns:
                    try:
                        abt.loc[:, col] = abt_estacoes_vizinhas[col]
                    except KeyError as e:
                        print(f"⚠️  Erro ao adicionar coluna {col}: {e}")
            
            print(f"✅ Dataset expandido para {abt.shape[1]} colunas")

        print(f"🔄 Realizando partição temporal ({percent_datetime_partitioning_split:.1%} treino)...")
        try:
            training_abt, validation_abt = particao_por_estacao(abt, percent_datetime_partitioning_split)
            print(f"📈 Treino: {len(training_abt)} registros | Teste: {len(validation_abt)} registros")
        except Exception as e:
            raise RuntimeError(f"Erro na partição dos dados: {e}")
        
        try:
            X_train, y_train = training_abt.drop('vl_precipitacao', axis=1), training_abt['vl_precipitacao']
            X_test, y_test = validation_abt.drop('vl_precipitacao', axis=1), validation_abt['vl_precipitacao']
        except KeyError as e:
            raise KeyError(f"Erro ao separar features e target: {e}")

        if zero_undersampling_ratio is not None:
            print(f"⚖️  Aplicando undersampling com ratio {zero_undersampling_ratio}...")
            print(f"    💡 Isso significa: {zero_undersampling_ratio} zeros para cada 1 não-zero")
            try:
                original_size = len(X_train)
                zeros_before = (y_train == 0).sum()
                non_zeros_before = (y_train > 0).sum()
                
                X_train, y_train = undersample_zeros(X_train, y_train, zero_ratio=zero_undersampling_ratio, random_state=random_state)
                
                zeros_after = (y_train == 0).sum()
                non_zeros_after = (y_train > 0).sum()
                actual_ratio = zeros_after / non_zeros_after if non_zeros_after > 0 else 0
                
                print(f"📉 Antes: {zeros_before:,} zeros, {non_zeros_before:,} não-zeros")
                print(f"📊 Depois: {zeros_after:,} zeros, {non_zeros_after:,} não-zeros")
                print(f"📈 Ratio real: {actual_ratio:.2f} | Tamanho: {original_size} → {len(X_train)}")
                
            except Exception as e:
                raise RuntimeError(f"Erro no undersampling: {e}")

        if smote_oversampling:
            print(f"🧬 Aplicando SMOTE-R com threshold={smote_threshold}...")
            try:
                training_combined = pd.concat([X_train, y_train], axis=1)
                
                balanced_training = smoteR(
                    dataframe=training_combined,
                    target_column='vl_precipitacao',
                    explanatory_variables=smote_explanatory_variables,
                    relevance_function=smote_relevance_function,
                    threshold=smote_threshold,
                    pct_oversampling=smote_pct_oversampling,
                    pct_undersampling=smote_pct_undersampling,
                    number_of_nearest_neighbors=smote_k_neighbors,
                    constraint_columns=smote_constraint_columns,
                    random_state=random_state
                )
                
                X_train = balanced_training.drop('vl_precipitacao', axis=1)
                y_train = balanced_training['vl_precipitacao']
                
                print(f"✅ SMOTE-R aplicado com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro na aplicação do SMOTE-R: {e}")
                print("   Continuando com dataset não balanceado...")
        
        print(f"\n📋 Resumo final:")
        print(f"   • Features de treino: {X_train.shape}")
        print(f"   • Features de teste: {X_test.shape}")
        print(f"   • Target treino - valores únicos: {y_train.nunique()}")
        print(f"   • Target teste - valores únicos: {y_test.nunique()}")
        
        if smote_oversampling or zero_undersampling_ratio is not None:
            print(f"\n📊 Estatísticas do target após processamento:")
            print(f"   • Treino - Média: {y_train.mean():.3f}, Mediana: {y_train.median():.3f}")
            print(f"   • Teste  - Média: {y_test.mean():.3f}, Mediana: {y_test.median():.3f}")
            print(f"   • Zeros no treino: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
            print(f"   • Zeros no teste: {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
        
    except (ValueError, KeyError, TypeError) as e:
        print(f"❌ Erro de validação: {e}")
        raise
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        raise RuntimeError(f"Erro inesperado durante o processamento: {e}")
    

def train_model(abt_estacoes_vizinhas, Model, model_number, usar_n_estacoes_vizinhas,
                zero_undersampling_ratio=None, smote_oversampling=False, 
                smote_threshold=0.5, smote_pct_oversampling=100,
                smote_pct_undersampling=100, smote_k_neighbors=5,
                smote_constraint_columns=None, smote_random_state=None,
                use_bi_model=False, threshold_prioridade=0.5, 
                percent_datetime_partitioning_split=0.7,
                truncate_to_non_negative_target=True):
    """
    Treina modelo(s) de machine learning para previsão de precipitação.
    
    Esta função treina modelos preditivos usando dados de estações meteorológicas,
    oferecendo duas abordagens: modelo único ou modelo duplo (bi-model). O modelo
    duplo separa os dados baseado na prioridade das estações vizinhas e treina
    modelos específicos para cada grupo.
    
    Parameters
    ----------
    abt_estacoes_vizinhas : pd.DataFrame
        DataFrame contendo os dados das estações meteorológicas e suas vizinhas.
        Deve conter as colunas 'id_estacao', 'dt_medicao' e 'vl_precipitacao'.
        
    Model : class
        Classe do modelo de machine learning a ser utilizado (ex: RandomForestRegressor).
        Deve implementar os métodos fit() e predict().
        
    model_number : int or str
        Identificador único do modelo para salvamento dos arquivos.
        
    usar_n_estacoes_vizinhas : int
        Número de estações vizinhas a incluir no dataset. Deve ser >= 0.
        
    zero_undersampling_ratio : float, optional, default=None
        Proporção de registros com precipitação zero em relação aos registros
        com precipitação > 0 no conjunto de treino. Se None, não aplica undersampling.
        Exemplos:
        - 1.0: mantém mesmo número de zeros e não-zeros (balanceamento 50/50)
        - 0.5: mantém metade de zeros em relação aos não-zeros
        - 2.0: mantém o dobro de zeros em relação aos não-zeros
        Deve ser > 0.
        
    smote_oversampling : bool, optional, default=False
        Se True, aplica técnica SMOTE-R para balanceamento de target contínuo.
        
    smote_threshold : float, optional, default=0.5
        Limiar para determinar observações raras vs comuns no SMOTE-R.
        
    smote_pct_oversampling : int, optional, default=100
        Porcentagem de oversampling para casos raros no SMOTE-R.
        
    smote_pct_undersampling : int, optional, default=100
        Porcentagem de undersampling para casos comuns no SMOTE-R.
        
    smote_k_neighbors : int, optional, default=5
        Número de vizinhos mais próximos para geração sintética no SMOTE-R.
        
    smote_constraint_columns : list or str, optional
        Colunas que devem ter valores iguais entre amostras e vizinhos no SMOTE-R.
        Exemplo: ['dt_medicao'] para manter consistência temporal.
        
    smote_random_state : int, optional
        Semente para reprodutibilidade do SMOTE-R.
        
    use_bi_model : bool, optional, default=False
        Se True, treina dois modelos separados baseado no threshold_prioridade.
        Se False, treina um modelo único.
        
    threshold_prioridade : float, optional, default=0.5
        Threshold para separar dados em 'com_vizinha' e 'sem_vizinha' quando
        use_bi_model=True. Deve estar entre 0 e 1.
        
    percent_datetime_partitioning_split : float, optional, default=0.7
        Percentual dos dados para treino na partição temporal.
        Deve estar entre 0 e 1.
        
    truncate_to_non_negative_target : bool, optional, default=True
        Se True, trunca predições negativas para 0 (precipitação não pode ser negativa).
    
    Returns
    -------
    tuple
        Tupla contendo (model, comparison):
        
        Para modelo único (use_bi_model=False):
        - model : objeto do modelo treinado
        - comparison : pd.DataFrame com comparação entre valores reais e preditos
        
        Para bi-model (use_bi_model=True):
        - model : dict com chaves 'com_vizinha' e 'sem_vizinha' contendo os modelos
        - comparison : dict com chaves 'com_vizinha' e 'sem_vizinha' contendo as comparações
    
    Raises
    ------
    ValueError
        Se os parâmetros estiverem fora dos limites válidos ou se colunas
        obrigatórias estiverem ausentes.
        
    TypeError
        Se o Model não implementar os métodos necessários ou se os tipos
        dos parâmetros não forem os esperados.
        
    RuntimeError
        Se ocorrer erro durante o treinamento ou salvamento dos modelos.
        
    FileNotFoundError
        Se os diretórios 'models' ou 'comparisons' não existirem.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> 
    >>> # Modelo único com 2 estações vizinhas
    >>> model, comparison = train_model(
    ...     df_estacoes, 
    ...     RandomForestRegressor, 
    ...     model_number=1,
    ...     usar_n_estacoes_vizinhas=2
    ... )
    
    >>> # Balanceamento 50/50 (mesmo número de zeros e não-zeros)
    >>> model, comparison = train_model(
    ...     df_estacoes, 
    ...     RandomForestRegressor, 
    ...     model_number=2,
    ...     usar_n_estacoes_vizinhas=2,
    ...     zero_undersampling_ratio=1.0
    ... )
    
    >>> # Bi-model com SMOTE-R e menos zeros que não-zeros
    >>> model, comparison = train_model(
    ...     df_estacoes,
    ...     RandomForestRegressor,
    ...     model_number=3,
    ...     usar_n_estacoes_vizinhas=3,
    ...     use_bi_model=True,
    ...     smote_oversampling=True,
    ...     smote_constraint_columns=['dt_medicao'],
    ...     zero_undersampling_ratio=0.5,  # metade de zeros em relação aos não-zeros
    ...     threshold_prioridade=0.6
    ... )
    
    Notes
    -----
    - A função salva automaticamente o modelo e comparação nos diretórios 'models' e 'comparisons'
    - Requer as funções auxiliares: generate_X_y_train_test, split_com_sem_vizinha, 
      compute_comparison_df, save_model_and_comparison
    - Para bi-model, os dados são separados baseado na prioridade das estações vizinhas
    - Predições negativas são truncadas para 0 por padrão (precipitação física)
    - O undersampling é aplicado apenas no conjunto de treino, não afetando o conjunto de teste
    """
    
    try:
        # Validação de parâmetros
        if not hasattr(Model, '__call__'):
            raise TypeError("Model deve ser uma classe instanciável")
        
        if not isinstance(usar_n_estacoes_vizinhas, int) or usar_n_estacoes_vizinhas < 0:
            raise ValueError("usar_n_estacoes_vizinhas deve ser um inteiro >= 0")
        
        if zero_undersampling_ratio is not None:
            if not isinstance(zero_undersampling_ratio, (int, float)) or zero_undersampling_ratio <= 0:
                raise ValueError("zero_undersampling_ratio deve ser None ou um número > 0")
        
        if not isinstance(threshold_prioridade, (int, float)) or not (0 <= threshold_prioridade <= 1):
            raise ValueError("threshold_prioridade deve ser um número entre 0 e 1")
        
        if not isinstance(percent_datetime_partitioning_split, (int, float)) or not (0 < percent_datetime_partitioning_split < 1):
            raise ValueError("percent_datetime_partitioning_split deve ser um número entre 0 e 1")
        
        # Verificar DataFrame de entrada
        if abt_estacoes_vizinhas.empty:
            raise ValueError("DataFrame de entrada não pode estar vazio")
        
        required_columns = ['id_estacao', 'dt_medicao', 'vl_precipitacao']
        missing_cols = [col for col in required_columns if col not in abt_estacoes_vizinhas.columns]
        if missing_cols:
            raise KeyError(f"Colunas obrigatórias não encontradas: {missing_cols}")
        
        # Verificar se diretórios existem
        os.makedirs('models', exist_ok=True)
        os.makedirs('comparisons', exist_ok=True)
        
        print(f"🚀 Iniciando treinamento do modelo {model_number}")
        print(f"📊 Dados de entrada: {abt_estacoes_vizinhas.shape}")
        print(f"🏗️  Modelo: {Model.__name__}")
        print(f"🌐 Estações vizinhas: {usar_n_estacoes_vizinhas}")
        print(f"🔄 Tipo de modelo: {'Bi-model' if use_bi_model else 'Modelo único'}")
        if zero_undersampling_ratio is not None:
            print(f"⚖️  Zero undersampling ratio: {zero_undersampling_ratio} (zeros por não-zero)")
        
        if not use_bi_model:
            print("\n=== TREINAMENTO MODELO ÚNICO ===")
            
            # Gerar dados de treino e teste
            print("📈 Preparando dados de treino e teste...")
            try:
                X_train, X_test, y_train, y_test = generate_X_y_train_test(
                    abt_estacoes_vizinhas,
                    usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,
                    zero_undersampling_ratio=zero_undersampling_ratio,
                    smote_oversampling=smote_oversampling,
                    smote_threshold=smote_threshold,
                    smote_pct_oversampling=smote_pct_oversampling,
                    smote_pct_undersampling=smote_pct_undersampling,
                    smote_k_neighbors=smote_k_neighbors,
                    smote_constraint_columns=smote_constraint_columns,
                    smote_relevance_function=None,
                    percent_datetime_partitioning_split=percent_datetime_partitioning_split,
                    random_state=smote_random_state
                )
            except Exception as e:
                raise RuntimeError(f"Erro na preparação dos dados: {e}")
            
            # Verificar se as colunas necessárias estão presentes para remoção
            train_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_train.columns]
            test_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_test.columns]
            
            if len(train_cols_to_drop) != len(test_cols_to_drop):
                print("⚠️  Aviso: Colunas de identificação diferentes entre treino e teste")
            
            # Treinar modelo
            print("🎯 Treinando modelo...")
            try:
                model = Model()
                X_train_features = X_train.drop(train_cols_to_drop, axis=1)
                print(f"   • Features utilizadas: {X_train_features.shape[1]}")
                print(f"   • Amostras de treino: {len(y_train)}")
                
                model.fit(X_train_features, y_train)
                print("✅ Treinamento concluído!")
            except Exception as e:
                raise RuntimeError(f"Erro durante o treinamento: {e}")
            
            # Fazer predições
            print("🔮 Gerando predições...")
            try:
                X_test_features = X_test.drop(test_cols_to_drop, axis=1)
                y_pred = model.predict(X_test_features)
                
                if truncate_to_non_negative_target:
                    negative_count = np.sum(y_pred < 0)
                    if negative_count > 0:
                        print(f"⚖️  Truncando {negative_count} predições negativas para 0")
                    y_pred = np.clip(y_pred, a_min=0, a_max=None)
                
                print(f"📊 Predições geradas: {len(y_pred)}")
                print(f"   • Valor mín: {np.min(y_pred):.3f}")
                print(f"   • Valor máx: {np.max(y_pred):.3f}")
                print(f"   • Média: {np.mean(y_pred):.3f}")
            except Exception as e:
                raise RuntimeError(f"Erro na geração de predições: {e}")
            
            # Computar comparação
            print("📋 Computando métricas de comparação...")
            try:
                comparison = compute_comparison_df(X_test, y_test, y_pred)
                print("✅ Comparação concluída!")
            except Exception as e:
                raise RuntimeError(f"Erro no cálculo de métricas: {e}")
        
        elif use_bi_model:
            print(f"\n=== TREINAMENTO BI-MODEL (threshold={threshold_prioridade}) ===")
            
            # Inicializar estruturas de dados
            abt, X_train, X_test, y_train, y_test, model, y_pred, comparison = {}, {}, {}, {}, {}, {}, {}, {}
            
            # Separar dados com e sem vizinha
            print("🔀 Separando dados por prioridade de estações vizinhas...")
            try:
                abt['com_vizinha'], abt['sem_vizinha'] = split_com_sem_vizinha(
                    abt_estacoes_vizinhas, threshold_prioridade
                )
                print(f"   • Com vizinha: {len(abt['com_vizinha'])} registros")
                print(f"   • Sem vizinha: {len(abt['sem_vizinha'])} registros")
            except Exception as e:
                raise RuntimeError(f"Erro na separação dos dados: {e}")
            
            # Preparar dados para cada tipo
            print("📈 Preparando dados de treino e teste para cada modelo...")
            for tipo in tqdm(['com_vizinha', 'sem_vizinha'], desc="Preparando dados"):
                try:
                    if len(abt[tipo]) == 0:
                        print(f"⚠️  Aviso: Dataset '{tipo}' está vazio!")
                        continue
                        
                    X_train[tipo], X_test[tipo], y_train[tipo], y_test[tipo] = generate_X_y_train_test(
                        abt[tipo],
                        usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,
                        zero_undersampling_ratio=zero_undersampling_ratio,
                        smote_oversampling=smote_oversampling,
                        smote_threshold=smote_threshold,
                        smote_pct_oversampling=smote_pct_oversampling,
                        smote_pct_undersampling=smote_pct_undersampling,
                        smote_k_neighbors=smote_k_neighbors,
                        smote_constraint_columns=smote_constraint_columns,
                        smote_relevance_function=None,
                        percent_datetime_partitioning_split=percent_datetime_partitioning_split,
                        random_state=smote_random_state
                    )
                except Exception as e:
                    raise RuntimeError(f"Erro na preparação dos dados para '{tipo}': {e}")
            
            # Treinar modelos
            print("🎯 Treinando modelos...")
            model['com_vizinha'], model['sem_vizinha'] = Model(), Model()
            
            for tipo in tqdm(['com_vizinha', 'sem_vizinha'], desc="Treinando modelos"):
                if tipo not in X_train or len(X_train[tipo]) == 0:
                    print(f"⚠️  Pulando treinamento para '{tipo}' - dados insuficientes")
                    continue
                    
                try:
                    cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_train[tipo].columns]
                    X_train_features = X_train[tipo].drop(cols_to_drop, axis=1)
                    
                    print(f"   • {tipo}: {X_train_features.shape[1]} features, {len(y_train[tipo])} amostras")
                    model[tipo].fit(X_train_features, y_train[tipo])
                except Exception as e:
                    raise RuntimeError(f"Erro no treinamento do modelo '{tipo}': {e}")
            
            print("✅ Treinamento de ambos os modelos concluído!")
            
            # Fazer predições e calcular comparações
            print("🔮 Gerando predições e métricas...")
            for tipo in tqdm(['com_vizinha', 'sem_vizinha'], desc="Gerando predições"):
                if tipo not in X_test or len(X_test[tipo]) == 0:
                    print(f"⚠️  Pulando predições para '{tipo}' - dados insuficientes")
                    continue
                    
                try:
                    cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_test[tipo].columns]
                    X_test_features = X_test[tipo].drop(cols_to_drop, axis=1)
                    
                    y_pred[tipo] = model[tipo].predict(X_test_features)
                    
                    if truncate_to_non_negative_target:
                        negative_count = np.sum(y_pred[tipo] < 0)
                        if negative_count > 0:
                            print(f"⚖️  {tipo}: Truncando {negative_count} predições negativas para 0")
                        y_pred[tipo] = np.clip(y_pred[tipo], a_min=0, a_max=None)
                    
                    comparison[tipo] = compute_comparison_df(X_test[tipo], y_test[tipo], y_pred[tipo])
                    
                    print(f"   • {tipo}: {len(y_pred[tipo])} predições (média: {np.mean(y_pred[tipo]):.3f})")
                except Exception as e:
                    raise RuntimeError(f"Erro nas predições para '{tipo}': {e}")
        
        # Salvar modelo e comparação
        print(f"💾 Salvando modelo e comparação...")
        try:
            model_path = f'models/model_{model_number}.pkl'
            comparison_path = f'comparisons/comparison_{model_number}.pkl'
            
            save_model_and_comparison(model, comparison, model_path, comparison_path)
            
            print(f"✅ Arquivos salvos:")
            print(f"   • Modelo: {model_path}")
            print(f"   • Comparação: {comparison_path}")
        except Exception as e:
            raise RuntimeError(f"Erro no salvamento: {e}")
        
        print(f"\n🎉 Processo concluído com sucesso para modelo {model_number}!")
        
        return model, comparison
    
    except (ValueError, KeyError, TypeError) as e:
        print(f"❌ Erro de validação: {e}")
        raise
    except RuntimeError as e:
        print(f"❌ Erro durante execução: {e}")
        raise
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        raise RuntimeError(f"Erro inesperado durante o treinamento: {e}")