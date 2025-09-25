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
                           smote_threshold=0.5, smote_pct_oversampling=0.01, 
                           smote_pct_undersampling=1.0, smote_k_neighbors=5,
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
        
    smote_pct_oversampling : float, optional, default=0.01
        Porcentagem decimal de aumento nos casos raros (0.01 = 1%, 0.50 = 50%)
        Exemplo: 100 casos raros com smote_pct_oversampling=0.01 → 1 caso sintético
        
    smote_pct_undersampling : float, optional, default=1.0
        Multiplicador para casos comuns em relação ao total de casos raros + sintéticos
        (1.0 = mesmo número, 0.5 = metade, 2.0 = dobro)
        
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
    
    >>> # Incluindo 3 estações vizinhas com SMOTE-R (1% de aumento)
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(
    ...     df_estacoes, 
    ...     usar_n_estacoes_vizinhas=3,
    ...     zero_undersampling_ratio=0.5,  # metade de zeros em relação aos não-zeros
    ...     smote_oversampling=True,
    ...     smote_threshold=0.6,
    ...     smote_pct_oversampling=0.01,  # 1% de aumento nos casos raros
    ...     smote_constraint_columns=['dt_medicao'],
    ...     random_state=42
    ... )
    
    >>> # SMOTE-R com 50% de aumento nos casos raros
    >>> X_train, X_test, y_train, y_test = generate_X_y_train_test(
    ...     df_estacoes, 
    ...     zero_undersampling_ratio=2.0,  # dobro de zeros em relação aos não-zeros
    ...     smote_oversampling=True,
    ...     smote_pct_oversampling=0.50,  # 50% de aumento
    ...     smote_pct_undersampling=0.8,   # 80% de casos comuns em relação aos raros+sintéticos
    ...     random_state=42
    ... )
    
    Notes
    -----
    - A função assume que existe uma função `particao_por_estacao` disponível
    - A função assume que existe uma função `undersample_zeros` disponível
    - A função assume que existe uma função `smoteR` disponível quando smote_oversampling=True
    - O undersampling é aplicado apenas no conjunto de treino, não afetando o conjunto de teste
    - smote_pct_oversampling agora é uma porcentagem decimal (0.01 = 1% de aumento)
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
        
        # MUDANÇA: Validação para smote_pct_oversampling como decimal
        if not isinstance(smote_pct_oversampling, (int, float)) or smote_pct_oversampling < 0:
            raise ValueError("smote_pct_oversampling deve ser um número >= 0 (ex: 0.01 para 1%)")
        
        if smote_pct_oversampling > 10.0:
            print(f"⚠️  Aviso: smote_pct_oversampling muito alto ({smote_pct_oversampling*100:.1f}%). Considere usar valores menores.")
        
        # MUDANÇA: Validação para smote_pct_undersampling como multiplicador
        if not isinstance(smote_pct_undersampling, (int, float)) or smote_pct_undersampling < 0:
            raise ValueError("smote_pct_undersampling deve ser um número >= 0")
        
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
            print(f"    📈 Oversampling: {smote_pct_oversampling*100:.2f}% de aumento nos casos raros")
            print(f"    ⚖️  Undersampling: multiplicador {smote_pct_undersampling} para casos comuns")
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
                    random_state=random_state)
                
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
                smote_threshold=0.5, smote_pct_oversampling=0.01,
                smote_pct_undersampling=1.0, smote_k_neighbors=5,smote_explanatory_variables=None,
                smote_constraint_columns=None, smote_random_state=None,
                use_bi_model=False, threshold_prioridade=0.5, 
                percent_datetime_partitioning_split=0.7,
                truncate_to_non_negative_target=True,
                classifier_model=None, classifier_thresholds=None):
    """
    Treina modelo(s) de machine learning para previsão de precipitação.
    
    Esta função treina modelos preditivos usando dados de estações meteorológicas,
    oferecendo três abordagens: modelo único, modelo duplo (bi-model) e/ou 
    modelo híbrido (classificador + regressores por bins). As abordagens podem
    ser combinadas para máxima flexibilidade.
    
    Parameters
    ----------
    abt_estacoes_vizinhas : pd.DataFrame
        DataFrame contendo os dados das estações meteorológicas e suas vizinhas.
        Deve conter as colunas 'id_estacao', 'dt_medicao' e 'vl_precipitacao'.
        
    Model : class
        Classe do modelo de machine learning a ser utilizado para regressão 
        (ex: RandomForestRegressor). Deve implementar os métodos fit() e predict().
        
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
        
    smote_pct_oversampling : float, optional, default=0.01
        Porcentagem decimal de aumento nos casos raros (0.01 = 1%, 0.50 = 50%)
        Exemplo: 100 casos raros com smote_pct_oversampling=0.01 → 1 caso sintético
        
    smote_pct_undersampling : float, optional, default=1.0
        Multiplicador para casos comuns em relação ao total de casos raros + sintéticos
        (1.0 = mesmo número, 0.5 = metade, 2.0 = dobro)
        
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
        
    classifier_model : class, optional, default=None
        Classe do modelo de classificação para abordagem híbrida (ex: XGBClassifier).
        Se fornecido, classifier_thresholds também deve ser fornecido.
        
    classifier_thresholds : array-like, optional, default=None
        Array com thresholds para criação dos bins de classificação.
        Exemplo: [1, 5, 20] cria bins: [0,1), [1,5), [5,20), [20,∞).
        Deve estar em ordem crescente e todos os valores > 0.
    
    Returns
    -------
    tuple
        Tupla contendo (model, comparison):
        
        Para modelo único sem classificador:
        - model : objeto do modelo treinado
        - comparison : pd.DataFrame com comparação entre valores reais e preditos
        
        Para modelo único com classificador:
        - model : dict com chaves 'classifier' e 'regressors' (bins)
        - comparison : pd.DataFrame com comparação incluindo predições híbridas
        
        Para bi-model sem classificador:
        - model : dict com chaves 'com_vizinha' e 'sem_vizinha' contendo os modelos
        - comparison : dict com chaves 'com_vizinha' e 'sem_vizinha' contendo as comparações
        
        Para bi-model com classificador:
        - model : dict aninhado onde cada tipo contém 'classifier' e 'regressors' (bins)
        - comparison : dict aninhado combinando ambas as estruturas
    
    Raises
    ------
    ValueError
        Se os parâmetros estiverem fora dos limites válidos, se colunas
        obrigatórias estiverem ausentes, ou se classifier_model e 
        classifier_thresholds não forem consistentes.
        
    TypeError
        Se o Model ou classifier_model não implementarem os métodos necessários
        ou se os tipos dos parâmetros não forem os esperados.
        
    RuntimeError
        Se ocorrer erro durante o treinamento ou salvamento dos modelos.
        
    FileNotFoundError
        Se os diretórios 'models' ou 'comparisons' não existirem.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from xgboost import XGBClassifier
    >>> 
    >>> # Modelo único tradicional
    >>> model, comparison = train_model(
    ...     df_estacoes, 
    ...     RandomForestRegressor, 
    ...     model_number=1,
    ...     usar_n_estacoes_vizinhas=2
    ... )
    
    >>> # Modelo híbrido (classificador + regressores por bins)
    >>> model, comparison = train_model(
    ...     df_estacoes,
    ...     RandomForestRegressor,
    ...     model_number=2,
    ...     usar_n_estacoes_vizinhas=2,
    ...     classifier_model=XGBClassifier,
    ...     classifier_thresholds=[1, 5, 20]  # Bins: [0,1), [1,5), [5,20), [20,∞)
    ... )
    
    >>> # Bi-model + híbrido
    >>> model, comparison = train_model(
    ...     df_estacoes,
    ...     RandomForestRegressor,
    ...     model_number=3,
    ...     usar_n_estacoes_vizinhas=3,
    ...     use_bi_model=True,
    ...     classifier_model=XGBClassifier,
    ...     classifier_thresholds=[2, 8, 15],  # Bins: [0,2), [2,8), [8,15), [15,∞)
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
    - smote_pct_oversampling agora é uma porcentagem decimal (0.01 = 1% de aumento)
    - smote_pct_undersampling agora é um multiplicador direto (1.0 = mesmo número)
    - Bins são criados como intervalos: [0, t1), [t1, t2), ..., [tn, ∞)
    - A predição híbrida usa o classificador para determinar o bin e depois o regressor correspondente
    - Cada regressor é especializado em seu range específico de precipitação
    """
    
    # Validações básicas
    if not hasattr(Model, '__call__'):
        raise TypeError("Model deve ser uma classe instanciável")
    
    if not isinstance(usar_n_estacoes_vizinhas, int) or usar_n_estacoes_vizinhas < 0:
        raise ValueError("usar_n_estacoes_vizinhas deve ser um inteiro >= 0")
    
    if zero_undersampling_ratio is not None and zero_undersampling_ratio <= 0:
        raise ValueError("zero_undersampling_ratio deve ser None ou um número > 0")
    
    if not isinstance(smote_pct_oversampling, (int, float)) or smote_pct_oversampling < 0:
        raise ValueError("smote_pct_oversampling deve ser um número >= 0 (ex: 0.01 para 1%)")
    
    if not isinstance(threshold_prioridade, (int, float)) or not (0 <= threshold_prioridade <= 1):
        raise ValueError("threshold_prioridade deve ser um número entre 0 e 1")
    
    # Validações do classificador - CORRIGIDAS
    if (classifier_model is None) != (classifier_thresholds is None):
        raise ValueError("classifier_model e classifier_thresholds devem ser fornecidos juntos ou ambos None")
    
    if classifier_model is not None:
        if not hasattr(classifier_model, '__call__'):
            raise TypeError("classifier_model deve ser uma classe instanciável")
        
        if not isinstance(classifier_thresholds, (list, tuple, np.ndarray)):
            raise TypeError("classifier_thresholds deve ser array-like")
        
        classifier_thresholds = np.array(classifier_thresholds)
        
        if len(classifier_thresholds) < 1:
            raise ValueError("classifier_thresholds deve ter pelo menos 1 elemento")
        
        if np.any(classifier_thresholds <= 0):
            raise ValueError("Todos os thresholds devem ser > 0")
        
        if not np.all(classifier_thresholds[:-1] < classifier_thresholds[1:]):
            raise ValueError("classifier_thresholds deve estar em ordem crescente")
    
    if abt_estacoes_vizinhas.empty:
        raise ValueError("DataFrame de entrada não pode estar vazio")
    
    required_columns = ['id_estacao', 'dt_medicao', 'vl_precipitacao']
    missing_cols = [col for col in required_columns if col not in abt_estacoes_vizinhas.columns]
    if missing_cols:
        raise KeyError(f"Colunas obrigatórias não encontradas: {missing_cols}")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('comparisons', exist_ok=True)
    
    use_classifier = classifier_model is not None
    model_type = f"{'Bi-model' if use_bi_model else 'Único'}{' + Híbrido' if use_classifier else ''}"
    
    print(f"Modelo {model_number} | {Model.__name__} | {usar_n_estacoes_vizinhas} estações | {model_type}")
    if use_classifier:
        print(f"  Classificador: {classifier_model.__name__} | Thresholds: {list(classifier_thresholds)}")
    
    def _create_target_classes(y_values, thresholds):
        """Cria classes baseadas nos bins definidos pelos thresholds."""
        classes = np.zeros(len(y_values), dtype=int)
        
        # Bin 0: [0, primeiro_threshold)
        # Bin 1: [primeiro_threshold, segundo_threshold)
        # ...
        # Bin n: [ultimo_threshold, ∞)
        
        for i, threshold in enumerate(thresholds, 1):
            classes[y_values >= threshold] = i
        
        return classes
    
    def _train_hybrid_model(X_train, X_test, y_train, y_test, model_prefix=""):
        """Treina modelo híbrido (classificador + regressores) para um dataset."""
        
        # Preparar dados para classificação
        y_train_classes = _create_target_classes(y_train, classifier_thresholds)
        y_test_classes = _create_target_classes(y_test, classifier_thresholds)
        
        # Remover colunas não-features
        train_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_train.columns]
        test_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_test.columns]
        
        X_train_features = X_train.drop(train_cols_to_drop, axis=1)
        X_test_features = X_test.drop(test_cols_to_drop, axis=1)
        
        # Treinar classificador
        classifier = classifier_model()
        classifier.fit(X_train_features, y_train_classes)
        
        # Predições do classificador
        y_pred_classes = classifier.predict(X_test_features)
        
        # Treinar regressores para cada bin
        regressors = {}
        n_bins = len(classifier_thresholds) + 1
        
        for bin_idx in range(n_bins):
            # Definir limites do bin
            if bin_idx == 0:
                # Bin 0: [0, primeiro_threshold)
                lower_bound = 0
                upper_bound = classifier_thresholds[0]
                mask_train = (y_train >= lower_bound) & (y_train < upper_bound)
                bin_name = f"bin_{bin_idx}"
                bin_desc = f"[{lower_bound}, {upper_bound})"
            elif bin_idx == n_bins - 1:
                # Último bin: [ultimo_threshold, ∞)
                lower_bound = classifier_thresholds[-1]
                mask_train = y_train >= lower_bound
                bin_name = f"bin_{bin_idx}"
                bin_desc = f"[{lower_bound}, ∞)"
            else:
                # Bins intermediários: [threshold_i, threshold_i+1)
                lower_bound = classifier_thresholds[bin_idx - 1]
                upper_bound = classifier_thresholds[bin_idx]
                mask_train = (y_train >= lower_bound) & (y_train < upper_bound)
                bin_name = f"bin_{bin_idx}"
                bin_desc = f"[{lower_bound}, {upper_bound})"
            
            if np.sum(mask_train) == 0:
                print(f"  ⚠️  {model_prefix}Bin {bin_idx} {bin_desc}: Nenhum dado de treino disponível")
                continue
            
            X_train_bin = X_train_features[mask_train]
            y_train_bin = y_train[mask_train]
            
            # Treinar regressor para este bin
            regressor = Model()
            regressor.fit(X_train_bin, y_train_bin)
            regressors[bin_name] = regressor
            
            print(f"  ✓ {model_prefix}Bin {bin_idx} {bin_desc}: {np.sum(mask_train)} amostras de treino")
        
        # Fazer predições híbridas
        y_pred_hybrid = np.zeros(len(y_test))
        
        for i, predicted_class in enumerate(y_pred_classes):
            bin_name = f"bin_{predicted_class}"
            
            if bin_name in regressors:
                # Fazer predição com o regressor do bin correspondente
                sample_features = X_test_features.iloc[i:i+1]
                prediction = regressors[bin_name].predict(sample_features)[0]
            else:
                # Fallback para o regressor do bin 0 se disponível
                if "bin_0" in regressors:
                    sample_features = X_test_features.iloc[i:i+1]
                    prediction = regressors["bin_0"].predict(sample_features)[0]
                else:
                    # Usar qualquer regressor disponível como último recurso
                    if regressors:
                        available_bin = list(regressors.keys())[0]
                        sample_features = X_test_features.iloc[i:i+1]
                        prediction = regressors[available_bin].predict(sample_features)[0]
                    else:
                        prediction = 0.0
            
            y_pred_hybrid[i] = prediction
        
        if truncate_to_non_negative_target:
            y_pred_hybrid = np.clip(y_pred_hybrid, a_min=0, a_max=None)
        
        # Criar estrutura do modelo híbrido
        hybrid_model = {
            'classifier': classifier,
            'regressors': regressors
        }
        
        # Calcular comparação incluindo informações do classificador
        comparison = compute_comparison_df(X_test, y_test, y_pred_hybrid)
        comparison['predicted_class'] = y_pred_classes
        comparison['actual_class'] = y_test_classes
        
        return hybrid_model, comparison
    
    def _train_single_model(X_train, X_test, y_train, y_test):
        """Treina modelo único (regressão tradicional)."""
        train_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_train.columns]
        test_cols_to_drop = [col for col in ['id_estacao', 'dt_medicao'] if col in X_test.columns]
        
        model = Model()
        X_train_features = X_train.drop(train_cols_to_drop, axis=1)
        model.fit(X_train_features, y_train)
        
        X_test_features = X_test.drop(test_cols_to_drop, axis=1)
        y_pred = model.predict(X_test_features)
        
        if truncate_to_non_negative_target:
            y_pred = np.clip(y_pred, a_min=0, a_max=None)
        
        comparison = compute_comparison_df(X_test, y_test, y_pred)
        return model, comparison
    
    if not use_bi_model:
        # Modelo único (com ou sem classificador)
        X_train, X_test, y_train, y_test = generate_X_y_train_test(
            abt_estacoes_vizinhas,
            usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,
            zero_undersampling_ratio=zero_undersampling_ratio,
            smote_oversampling=smote_oversampling,
            smote_threshold=smote_threshold,
            smote_pct_oversampling=smote_pct_oversampling,
            smote_pct_undersampling=smote_pct_undersampling,
            smote_k_neighbors=smote_k_neighbors,
            smote_explanatory_variables=smote_explanatory_variables,
            smote_constraint_columns=smote_constraint_columns,
            smote_relevance_function=None,
            percent_datetime_partitioning_split=percent_datetime_partitioning_split,
            random_state=smote_random_state
        )
        
        if use_classifier:
            model, comparison = _train_hybrid_model(X_train, X_test, y_train, y_test)
        else:
            model, comparison = _train_single_model(X_train, X_test, y_train, y_test)
    
    else:
        # Bi-model (com ou sem classificador)
        abt_com_vizinha, abt_sem_vizinha = split_com_sem_vizinha(
            abt_estacoes_vizinhas, threshold_prioridade
        )
        
        X_train, X_test, y_train, y_test, model, comparison = {}, {}, {}, {}, {}, {}
        
        for tipo in tqdm(['com_vizinha', 'sem_vizinha'], desc="Preparando dados"):
            abt_data = abt_com_vizinha if tipo == 'com_vizinha' else abt_sem_vizinha
            
            if len(abt_data) == 0:
                print(f"  ⚠️  Tipo '{tipo}': Nenhum dado disponível")
                continue
                
            X_train[tipo], X_test[tipo], y_train[tipo], y_test[tipo] = generate_X_y_train_test(
                abt_data,
                usar_n_estacoes_vizinhas=usar_n_estacoes_vizinhas,
                zero_undersampling_ratio=zero_undersampling_ratio,
                smote_oversampling=smote_oversampling,
                smote_threshold=smote_threshold,
                smote_pct_oversampling=smote_pct_oversampling,
                smote_pct_undersampling=smote_pct_undersampling,
                smote_k_neighbors=smote_k_neighbors,
                smote_constraint_columns=smote_constraint_columns,
                smote_relevance_function=None,
                smote_explanatory_variables=smote_explanatory_variables,
                percent_datetime_partitioning_split=percent_datetime_partitioning_split,
                random_state=smote_random_state
            )
        
        for tipo in tqdm(['com_vizinha', 'sem_vizinha'], desc="Treinando modelos"):
            if tipo not in X_train or len(X_train[tipo]) == 0:
                continue
            
            model_prefix = f"[{tipo}] "
            
            if use_classifier:
                model[tipo], comparison[tipo] = _train_hybrid_model(
                    X_train[tipo], X_test[tipo], y_train[tipo], y_test[tipo], model_prefix
                )
            else:
                model[tipo], comparison[tipo] = _train_single_model(
                    X_train[tipo], X_test[tipo], y_train[tipo], y_test[tipo]
                )
    
    model_path = f'models/model_{model_number}.pkl'
    comparison_path = f'comparisons/comparison_{model_number}.pkl'
    
    save_model_and_comparison(model, comparison, model_path, comparison_path)
    
    print(f"✅ Modelo {model_number} salvo")
    
    return model, comparison