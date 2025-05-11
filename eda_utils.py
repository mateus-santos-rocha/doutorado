import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plotar_estacoes_mapa(df, markersize=1, markercolor='blue', 
                         xlim=(-54, -37), ylim=(-25, -16), 
                         figsize=(10, 10), mostrar_borda=True):
    """
    Plota um mapa com estações meteorológicas e divisas estaduais.

    Parâmetros:
    - df: DataFrame com colunas 'latitude' e 'longitude'
    - markersize: tamanho dos pontos das estações
    - markercolor: cor dos pontos das estações
    - xlim: tupla com limites de longitude (min, max)
    - ylim: tupla com limites de latitude (min, max)
    - figsize: tupla com tamanho da figura (largura, altura)
    - mostrar_borda: se True, desenha uma borda pontilhada na área (-53, -24) até (-38, -17)
    """

    # Converter para GeoDataFrame
    gdf_estacoes = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    # Carregar mapa do Brasil com divisões estaduais
    brasil = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    estados = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")

    # Filtrar Brasil e garantir projeção
    brasil = brasil[brasil['name'] == 'Brazil']
    estados = estados.to_crs("EPSG:4326")

    # Plotagem
    fig, ax = plt.subplots(figsize=figsize)
    brasil.plot(ax=ax, color='white', edgecolor='black')
    estados.boundary.plot(ax=ax, color='gray', linewidth=1)
    gdf_estacoes.plot(ax=ax, color=markercolor, markersize=markersize, label='Estações')

    # Borda pontilhada (opcional)
    if mostrar_borda:
        rect = patches.Rectangle(
            xy=(-53, -24), width=15, height=7,  # (lon_min, lat_min), largura, altura
            linewidth=1,
            edgecolor='black',
            facecolor='none',
            linestyle='dotted'
        )
        ax.add_patch(rect)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.title('Estações Meteorológicas na Região R2')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
