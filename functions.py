from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from streamlit import pyplot

# Función para filtrar la data "rango_tiempo" días para atrás
def datos_rang_time(data, rango_tiempo):                            # in: data, rango_tiempo (cantidad de días desde el final en entero)
    data['updated_at'] = pd.to_datetime(data['updated_at'])
    fecha_mas_reciente = data['updated_at'].max()
    fecha_limite = fecha_mas_reciente - pd.Timedelta(days=rango_tiempo)
    return data[data['updated_at'] >= fecha_limite]

# Función que devuelve un dataframe filtrado con solo las transacciones de los usuarios que compran y venden
def user_compra_venta(df):
    usuarios_compras = set(df.loc[df['type'] == 'buy', 'owner_username'])
    usuarios_ventas = set(df.loc[df['type'] == 'sell', 'owner_username'])
    usuarios_compra_y_venta = usuarios_compras & usuarios_ventas

    # Filtrar el DataFrame para incluir solo los usuarios seleccionados
    df_filtrado = df[df['owner_username'].isin(usuarios_compra_y_venta)].copy()

    return list(df_filtrado['owner_username'].unique())

# Hace una lista de candidatos a MM a partir de un número mínimo de transacciones (>=3) consideradas y la data. Devuelve los owner_username
def candidatos_a_mm( data , num_transacciones_minimo ):             # in: base_de_datos_para_una_moneda, número de transacciones mínimo. out: lista de nombres
    candidates_dict = dict(data['owner_username'].value_counts())
    return [nombre for nombre, cantidad in candidates_dict.items() if cantidad > max(num_transacciones_minimo,3)]


# Para dar un número de transacciones mínimas se puede consultar la función que devuelve el promedio entre el máximo número de transacciones y el tercer cuartil
def minimo_de_transacciones_mm(data):                               # in: base de datos para una moneda. out: entero
    return int((data['owner_username'].value_counts().describe()['75%'] + data['owner_username'].value_counts().describe()['max'])/2)


# Pasar a flotante las columnas entendidas como string u objects. Útil para el procesamiento estadístico de las columnas
def numerizar_columnas( data , columnas ):                      # in: data, lista_de_columnas. out: data 
    data = data.copy()
    for columna in columnas:
        data[columna] = pd.to_numeric(data[columna], errors='coerce')
    return data

# Dado un username y la base de datos, devuelve un diccionario con un grupo de estadísticas del username. Útil para hacer un análisis de posibles MM
def datos_market_maker_analisis( data , username ):
   df = numerizar_columnas(data.loc[data['owner_username'] == username], ['amount', 'receive', 'owner_average_rating', 'owner_vip', 'owner_kyc'])
   df_ventas  = df.loc[df['type'] == 'sell'].copy()
   df_compras  = df.loc[df['type'] == 'buy'].copy()
   
   if len(df_compras) != 0:
      df_compras['bid'] = df_compras['receive'] / df_compras['amount']        # Bid = Precio de Compra
      precio_de_compra = df_compras['bid'].describe()['max']
   if len(df_ventas) != 0:
      df_ventas['ask'] = df_ventas['receive'] / df_ventas['amount']           # Ask = Precio de Venta
      precio_de_venta = df_ventas['ask'].describe()['min']
   precio_de_compra = precio_de_compra if len(df_compras) != 0 else None       # Datos_SPREAD
   precio_de_venta = precio_de_venta if len(df_ventas) != 0 else None
   spread = precio_de_venta - precio_de_compra if len(df_ventas) * len(df_compras) != 0 else None
   
   df['updated_at'] = pd.to_datetime(df['updated_at'])
   if len(df) > 1:
      tasa_actividad = (df['updated_at'].max() - df['updated_at'].min()).total_seconds() / len(df)
   else:
      tasa_actividad = None
   if len(df_compras) > 0:
      ratio_ventas_compras = len(df_ventas) / len(df_compras)
   else:
      ratio_ventas_compras = None
   
   #Diccionario con los datos de un mm
   datos_mm = {
      'username': username,
      'num_ofertas': len(df),
      'num_ventas': len(df_ventas),                                               # Número total de ventas
      'num_compras': len(df_compras),                                             # Número total de compras
      'ratio_ventas_compras': ratio_ventas_compras,                               # Razón de ventas sobre compras
      'precio_de_compra': precio_de_compra,                                       # Precio al que compra el dolar (de no existir, False)
      'precio_de_venta': precio_de_venta,                                         # Precio al que vende el dolar (de no existir, False)
      'spread': spread,                                                           # Spread
      'monto_por_vender_USD': df_ventas['amount'].sum(),                          # Cantidad total a vender
      'monto_por_comprar_USD': df_compras['amount'].sum(),                        # Cantidad total a comprar
      'evaluacion_del_usuario': df['owner_average_rating'].describe()['min'],     # Evaluación media del usuario en base a 5
      'vip + kyc': (df['owner_vip'].sum() + df['owner_kyc'].sum())/len(df),       # Si es vip o kyc
      'tasa_actividad_(s)': tasa_actividad,                                       # Tiempo medio entre las transacciones
   }   
   return datos_mm

# Clusteriza con KMeans y Agglomerative para detectar posibles Market_Makers
def kmeans_sobre_mm(data, num_clusters):         # in: dataset y numero de clusters (2), out: dataframe con los clusters agregados + puntos para ploteo
    usuarios = data['owner_username'].unique()  # Extrae usuarios únicos
    analisis_usuarios = [datos_market_maker_analisis(data, user) for user in usuarios]
    df_mm = pd.DataFrame(analisis_usuarios)  # Crea un DataFrame con los datos


    # Seleccionar características para el modelo
    features = ['num_ofertas', 'num_ventas', 'num_compras', 'spread', 'monto_por_vender_USD', 
                'monto_por_comprar_USD', 'evaluacion_del_usuario', 'vip + kyc', 'tasa_actividad_(s)']
    df_features = df_mm[features]

    # Imputar valores faltantes
    df_features = df_features.fillna(df_features.mean())

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Crear y ajustar el modelo
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42, n_init=10)  # Supongamos que hay 2 grupos (MM y no-MM)
    kmeans.fit(X_scaled)

    # Agregar los labels al DataFrame original
    df_mm['cluster_kmeans'] = kmeans.labels_

    # Reducir a dos dimensiones para visualizar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # CLUSTER - JERARQUICO
    # Crear y ajustar el modelo jerárquico
    agg_clustering = AgglomerativeClustering(n_clusters=2)
    clusters = agg_clustering.fit_predict(X_scaled)

    agg_clustering.labels_
    # Agregar clusters al DataFrame
    df_mm['cluster_jerarquico'] = clusters

    return df_mm, X_pca, kmeans.labels_, agg_clustering.labels_

# Función que combina 4 criterios para detectar Market_Makers, compra_y_venta, cantidad de compras, KMeans y clusterización_jerarquica
# Devuelve un diccionario con los usuarios que cumplen alguna de las anteriores y otro con los que cumplen 3 de ellas y la de compra_y_venta
def market_makers_criterio_combinado(df):
    cluster = cluster = kmeans_sobre_mm( df, 2)
    lista_num_compras =  candidatos_a_mm(df, minimo_de_transacciones_mm(df))
    lista_compra_venta = user_compra_venta(df) 
    lista_kmeans = list(cluster[0][cluster[0]['cluster_kmeans'] == 1]['username'])
    lista_jerarquic_cluster = list(cluster[0][cluster[0]['cluster_jerarquico'] == 0]['username'])

    todas_las_listas = [lista_num_compras, lista_compra_venta, lista_kmeans, lista_jerarquic_cluster]

    apariciones = Counter()
    

    for lista in todas_las_listas:
        for nombre in set(lista):  # Usar `set` para evitar duplicados dentro de la misma lista
            apariciones[nombre] += 1


    for nombre in set(lista_compra_venta):
        apariciones[nombre] += 1

    candidatos_todos = dict(sorted(dict(apariciones).items(), key=lambda item: item[1], reverse=True))
    mejores_candidatos = {clave: valor for clave, valor in candidatos_todos.items() if valor >= 4}

    return candidatos_todos, mejores_candidatos

# Dado el mercado de una moneda correr estadísticas principales.
def datos_de_mercado_x_moneda(data, coin):                  #in: data y moneda(s) a tratar (str o lista de str). out: diccionario con datos sobre la moneda
    if type(coin) == list:
        df = numerizar_columnas(data.loc[data['coin'].isin(coin)], ['amount', 'receive'])
    else:
        df = numerizar_columnas(data.loc[data['coin'] == coin], ['amount', 'receive'])

    df_ventas  = df.loc[df['type'] == 'sell'].copy()
    df_compras  = df.loc[df['type'] == 'buy'].copy()
    

    # BID - ASK - SPREAD ---------------
    if len(df_compras) != 0:
        df_compras['bid'] = df_compras['receive'] / df_compras['amount']        # Bid = Precio de Compra
        precio_de_compra = df_compras['bid'].describe()['max'] 
    else:
        precio_de_compra = None
        estimacion_precio_compras = None
        
    if len(df_ventas) != 0:
        df_ventas['ask'] = df_ventas['receive'] / df_ventas['amount']           # Ask = Precio de Venta
        precio_de_venta = df_ventas['ask'].describe()['min']
    else:
        precio_de_venta = None
        estimacion_precio_ventas = None

    spread = precio_de_venta - precio_de_compra if len(df_ventas) * len(df_compras) != 0 else None
    # ---------------------------------------------

    if df_compras['amount'].sum() !=0:
        razon_ofert_demand = df_ventas['amount'].sum()/ df_compras['amount'].sum()
    else:
        razon_ofert_demand = None


    if len(datos_rang_time(df_compras, 7)) != 0:
        data_7_comp = datos_rang_time(df_compras, 7)['bid'].describe()
        estimacion_precio_compras = data_7_comp['50%']
        media_precio_compra = data_7_comp['mean']
        desviacion_compra = data_7_comp['std']
    else:
        estimacion_precio_compras = None
        media_precio_compra = None
        desviacion_compra = None
    

    if len(datos_rang_time(df_ventas, 7)) != 0:
        data_7_venta = datos_rang_time(df_ventas, 7)['ask'].describe()
        estimacion_precio_ventas = data_7_venta['50%']
        media_precio_venta = data_7_venta['mean']
        desviacion_venta = data_7_venta['std']
    else:
        estimacion_precio_ventas = None
        media_precio_venta = None
        desviacion_venta = None


    # Candidatos a Market_Maker
    try:
        potenciales_mm = list(market_makers_criterio_combinado(df)[1])
        df_usuarios = df[df['owner_username'].isin(potenciales_mm)]   
        checked = True 
    except:
        potenciales_mm = candidatos_a_mm(df, minimo_de_transacciones_mm(df))
        df_usuarios = df[df['owner_username'].isin(potenciales_mm)]   
        checked = False

    # Diccionario con los datos de un mm
    datos_coin = {
        'monedas': coin,
        'total_ofertas': len(df),
        'cantidad_de_usuarios': len(list(df['owner_username'].unique())),           # Cantidad de usuarios transaccionando la moneda
        
        'num_compras': len(df_compras),                                             # Número total de compras
        'monto_por_comprar': df_compras['amount'].sum(),                            # Cantidad total a comprar
        'mejor_precio_de_compra': precio_de_compra,                                 # Precio al que compra el dolar (de no existir, False)
        
        'num_ventas': len(df_ventas),                                               # Número total de ventas
        'monto_por_vender': df_ventas['amount'].sum(),                              # Cantidad total a vender
        'mejor_precio_de_venta': precio_de_venta,                                   # Precio al que vende el dolar (de no existir, False)
        
        'razón_oferta/demanda': razon_ofert_demand,                                 # Razón entrecantidad a vender y cantidad a comprar
        'spread': spread,                                                           # Spread = precio_de_venta - precio_de_compra
        
        'precio_compra_estimado_ult_sem': estimacion_precio_compras,                # Estimación del precio de compra por la mediana
        'precio_medio_compra_ult_sem': media_precio_compra,                         # Precio medio de compra durante la última semana
        'desviacion_compra_ult_sem': desviacion_compra,                             # Desviación estándar de compra última semana 
        'precio_venta_estimado_ult_sem': estimacion_precio_ventas,                  # Estimación del precio de venta por la semana
        'precio_medio_venta_ult_sem': media_precio_venta,                           # Precio medio de venta durante la última semana
        'desviacion_venta_ult_sem': desviacion_venta,                               # Desviación estándar de venta última semana
        
        'cantidad_de_potenciales_mm': len(potenciales_mm) if potenciales_mm != None else None,                # Cantidad de potenciales Market Makers 
        'checked_mm': checked,                                                                                # Comprueba si el criterio combinado se ejecutó
        'participacion_de_potenciales_mm': len(df_usuarios)/len(df) if potenciales_mm != None else None       # participación en el mercado de los potenciales Market_Makers
    }
    
    return datos_coin

# Devuelve:
# Cantidad de usuarios transaccionando la moneda
# Número total de compras                   
# Cantidad total a comprar
# Precio al que compra el dolar (de no existir, False)
# Número total de ventas
# Cantidad total a vender
# Precio al que vende el dolar (de no existir, False)
# Razón entrecantidad a vender y cantidad a comprar
# Spread = precio_de_venta - precio_de_compra
# Estimación del precio de compra por la mediana
# Precio medio de compra durante la última semana
# Desviación estándar de compra última semana 
# Estimación del precio de venta por la semana
# Precio medio de venta durante la última semana
# Desviación estándar de venta última semana
# Cantidad de potenciales Market Makers 
# Comprueba si el criterio combinado se ejecutó
# participación en el mercado de los potenciales Market_Makers

############


# Función que genera un dataframe con toda la estadística de la tabla
def data_general(df):
    lista_monedas = [datos_de_mercado_x_moneda(df, coin) for coin in list(df['coin'].unique())]
    ddff = pd.DataFrame(lista_monedas)
    return (ddff.sort_values(by=['total_ofertas'], ascending=False)).set_index('monedas')

# Grafica un spread porcentual de la moneda, primer cuartil de venta - tercer cuartil de compra, para evitar precios de sobrecompra o subventa 
# no ejecutadas.        in: dataframe y moneda. Imprime Gráfica.  out: None
def graficar_spread_approx(df, moneda):
    df = numerizar_columnas(df, ['receive', 'amount'])
    # Filtrar datos por la moneda
    df_moneda = df[df['coin'] == moneda].copy()
    df_moneda['updated_at'] = pd.to_datetime(df_moneda['updated_at'])
    
    # Filtrar ventas y compras
    df_ventas = df_moneda[df_moneda['type'] == 'sell']
    df_compras = df_moneda[df_moneda['type'] == 'buy']
    
    # Validar datos suficientes
    if df_ventas.empty or df_compras.empty:
        print(f"No hay suficientes datos de ventas o compras para {moneda}.")
        return
    
    # Definir puntos equidistantes en el rango de tiempo
    primera_fecha = df_ventas['updated_at'].min()
    ultima_fecha = df_ventas['updated_at'].max()
    puntos_tiempo = pd.date_range(start=primera_fecha, end=ultima_fecha, periods=10)
    
    # Calcular el spread en cada punto
    spreads = []
    for tiempo in puntos_tiempo:
        ventas_filtradas = df_ventas[df_ventas['updated_at'] <= tiempo]
        compras_filtradas = df_compras[df_compras['updated_at'] <= tiempo]
        
        if ventas_filtradas.empty or compras_filtradas.empty:
            spreads.append(None)
        else:
            precio_venta_min = (ventas_filtradas['receive'] / ventas_filtradas['amount']).describe()["25%"]
            precio_compra_max = (compras_filtradas['receive'] / compras_filtradas['amount']).describe()["75%"]
            spreads.append(precio_venta_min - precio_compra_max)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(puntos_tiempo, spreads, marker='o', label='Spread')
    plt.xlabel("Tiempo")
    plt.ylabel("Spread")
    plt.title(f"Spread en puntos equidistantes ({moneda})")
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    pyplot(plt)

# Grafica el spread de la moneda, es posible que sea negativo a partir de transacciones sin ejecutar
# in: dataframe y moneda. Imprime Gráfica.  out: None
def graficar_spread(df, moneda):
    df = numerizar_columnas(df, ['receive', 'amount'])
    # Filtrar datos por la moneda
    df_moneda = df[df['coin'] == moneda].copy()
    df_moneda['updated_at'] = pd.to_datetime(df_moneda['updated_at'])
    
    # Filtrar ventas y compras
    df_ventas = df_moneda[df_moneda['type'] == 'sell']
    df_compras = df_moneda[df_moneda['type'] == 'buy']
    
    # Validar datos suficientes
    if df_ventas.empty or df_compras.empty:
        print(f"No hay suficientes datos de ventas o compras para {moneda}.")
        return
    
    # Definir puntos equidistantes en el rango de tiempo
    primera_fecha = df_ventas['updated_at'].min()
    ultima_fecha = df_ventas['updated_at'].max()
    puntos_tiempo = pd.date_range(start=primera_fecha, end=ultima_fecha, periods=10)
    
    # Calcular el spread en cada punto
    spreads = []
    for tiempo in puntos_tiempo:
        ventas_filtradas = df_ventas[df_ventas['updated_at'] <= tiempo]
        compras_filtradas = df_compras[df_compras['updated_at'] <= tiempo]
        
        if ventas_filtradas.empty or compras_filtradas.empty:
            spreads.append(None)
        else:
            precio_venta_min = (ventas_filtradas['receive'] / ventas_filtradas['amount']).describe()["min"]
            precio_compra_max = (compras_filtradas['receive'] / compras_filtradas['amount']).describe()["max"]
            spreads.append(precio_venta_min - precio_compra_max)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(puntos_tiempo, spreads, marker='o', label='Spread')
    plt.xlabel("Tiempo")
    plt.ylabel("Spread")
    plt.title(f"Spread en puntos equidistantes ({moneda})")
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    pyplot(plt)

############

# Grafica precio estimado de venta y de compra de la moneda en cuestión tomando el comportamiento mediano en la última semana.
# in: dataframe y moneda. Imprime Gráfica.  out: None
def graficar_precios_medianos(df, moneda):
    df = numerizar_columnas(df, ['receive', 'amount'])
    # Filtrar datos por la moneda
    df_moneda = df[df['coin'] == moneda].copy()
    df_moneda['updated_at'] = pd.to_datetime(df_moneda['updated_at'])
    
    # Filtrar ventas y compras
    df_ventas = df_moneda[df_moneda['type'] == 'sell']
    df_compras = df_moneda[df_moneda['type'] == 'buy']
    
    # Validar datos suficientes
    if df_ventas.empty or df_compras.empty:
        print(f"No hay suficientes datos de ventas o compras para {moneda}.")
        return
    
    # Definir puntos equidistantes en el rango de tiempo
    primera_fecha = df_ventas['updated_at'].min()
    ultima_fecha = df_ventas['updated_at'].max()
    puntos_tiempo = pd.date_range(start=primera_fecha, end=ultima_fecha, periods=10)
    
    # Calcular las medianas para cada punto
    precios_compras = []
    precios_ventas = []
    for tiempo in puntos_tiempo:
        inicio_semana = tiempo - pd.Timedelta(weeks=1)
        ventas_semana = df_ventas[(df_ventas['updated_at'] >= inicio_semana) & (df_ventas['updated_at'] <= tiempo)]
        compras_semana = df_compras[(df_compras['updated_at'] >= inicio_semana) & (df_compras['updated_at'] <= tiempo)]
        
        if ventas_semana.empty:
            precios_ventas.append(None)
        else:
            precios_ventas.append((ventas_semana['receive'] / ventas_semana['amount']).median())
        
        if compras_semana.empty:
            precios_compras.append(None)
        else:
            precios_compras.append((compras_semana['receive'] / compras_semana['amount']).median())
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(puntos_tiempo, precios_compras, marker='o', linestyle='--', label='Precio de Compra Mediano')
    plt.plot(puntos_tiempo, precios_ventas, marker='o', linestyle='-', label='Precio de Venta Mediano')
    plt.xlabel("Tiempo")
    plt.ylabel("Precio Mediano")
    plt.title(f"Precios Median (última semana) ({moneda})")
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    pyplot(plt)

############

# Grafica la razon de la oferta y la demanda generada en las últimas 24 horas desde la primera venta.
# in: dataframe y moneda. Imprime Gráfica.  out: None
def graficar_razon_oferta_demanda_24h(df, moneda):
    df = numerizar_columnas(df, ['receive', 'amount'])
    df_moneda = df[df['coin'] == moneda].copy()
    df_moneda['updated_at'] = pd.to_datetime(df_moneda['updated_at'])

    # Filtrar ventas y compras
    df_ventas = df_moneda[df_moneda['type'] == 'sell']
    df_compras = df_moneda[df_moneda['type'] == 'buy']
    
    if df_ventas.empty or df_compras.empty:
        print(f"No hay suficientes datos para {moneda}.")
        return

    # Definir puntos equidistantes
    primera_fecha = df_ventas['updated_at'].min()
    ultima_fecha = df_ventas['updated_at'].max()
    puntos_tiempo = pd.date_range(start=primera_fecha, end=ultima_fecha, periods=10)

    # Calcular razón oferta/demanda
    razones = []
    for tiempo in puntos_tiempo:
        inicio_24h = tiempo - pd.Timedelta(hours=24)
        ventas_24h = df_ventas[(df_ventas['updated_at'] >= inicio_24h) & (df_ventas['updated_at'] <= tiempo)]
        compras_24h = df_compras[(df_compras['updated_at'] >= inicio_24h) & (df_compras['updated_at'] <= tiempo)]

        volumen_venta = ventas_24h['receive'].sum()
        volumen_compra = compras_24h['receive'].sum()
        razon = volumen_venta / volumen_compra if volumen_compra > 0 else None
        razones.append(razon)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(puntos_tiempo, razones, marker='o', label='Razón Oferta/Demanda (24h)')
    plt.xlabel("Tiempo")
    plt.ylabel("Razón Oferta/Demanda")
    plt.title(f"Razón Oferta/Demanda en las últimas 24 horas ({moneda})")
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    pyplot(plt)
    
# Grafica de spread por cada uno de los usuarios potenciales Market_Makers (por el criterio combinado) en una moneda determinada 
# Si no funciona el criterio combinado busca los casos donde haya gente que compra y venda, si no returna Non sin imprimir.
# in: fataframe y moneda. Imprime gráfico. out: None
def graficar_spread_por_usuario_mm_checked(df, coin):
    token = '(seleccion MM robusta)'
    df = numerizar_columnas(df, ['receive', 'amount'])
    # Filtrar datos por la moneda
    df = df[df['coin'] == coin].copy()
    try:
        lista_usuarios = list(market_makers_criterio_combinado(df)[0])
    except:
        lista_usuarios = user_compra_venta(df)
        token = '(selección MM débil)'
        if len(lista_usuarios) < 2:
            return None
    if len(lista_usuarios) < 2:
        lista_usuarios = user_compra_venta(df)
        token = '(selección MM débil)'
        if len(lista_usuarios) < 2:
            return None
    # Crear una lista para guardar los datos de spread
    spreads = []
    nombres = []

    # Calcular el spread para cada usuario
    for username in lista_usuarios:
        datos_mm = datos_market_maker_analisis(df, username)
        spread = datos_mm.get('spread', None)
        if spread is not None:
            spreads.append(spread)
            nombres.append(username)
    
    # Graficar los resultados
    plt.figure(figsize=(12, 6))
    plt.bar(nombres, spreads, color='blue', alpha=0.7)
    plt.xlabel('Usuarios')
    plt.ylabel('Spread')
    plt.title(f'Spread por Usuario potencial Market Maker {token}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    pyplot(plt)


# Gráfica plot de clusterizacion, usando PCA para agrupar por los ejes
def print_df_mm(X_pca, kmeans_labels):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clustering de Usuarios')
    pyplot(plt)
    
# Clusteriza con KMeans y Agglomerative para detectar posibles Market_Makers
def kmeans_sobre_mm(data, num_clusters):         # in: dataset y numero de clusters (2), out: dataframe con los clusters agregados + puntos para ploteo
    usuarios = data['owner_username'].unique()  # Extrae usuarios únicos
    analisis_usuarios = [datos_market_maker_analisis(data, user) for user in usuarios]
    df_mm = pd.DataFrame(analisis_usuarios)  # Crea un DataFrame con los datos


    # Seleccionar características para el modelo
    features = ['num_ofertas', 'num_ventas', 'num_compras', 'spread', 'monto_por_vender_USD', 
                'monto_por_comprar_USD', 'evaluacion_del_usuario', 'vip + kyc', 'tasa_actividad_(s)']
    df_features = df_mm[features]

    # Imputar valores faltantes
    df_features = df_features.fillna(df_features.mean())

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Crear y ajustar el modelo
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42, n_init=10)  # Supongamos que hay 2 grupos (MM y no-MM)
    kmeans.fit(X_scaled)

    # Agregar los labels al DataFrame original
    df_mm['cluster_kmeans'] = kmeans.labels_

    # Reducir a dos dimensiones para visualizar
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # CLUSTER - JERARQUICO
    # Crear y ajustar el modelo jerárquico
    agg_clustering = AgglomerativeClustering(n_clusters=2)
    clusters = agg_clustering.fit_predict(X_scaled)

    agg_clustering.labels_
    # Agregar clusters al DataFrame
    df_mm['cluster_jerarquico'] = clusters

    return df_mm, X_pca, kmeans.labels_, agg_clustering.labels_

# Devuelve el dataframe filtrado, solo con la compra o venta del USDT a cambio de las monedas listadas. Útil para reducir el campo de acción.
def seleccion_de_monedas( data , monedas ):            # in: data, lista de monedas. out: data filtrada
    filtro = (data['coin'].isin(monedas))
    return data.loc[filtro]

# Funcion que dado el dataframe principal y una moneda devuelve volumen de oferta y demanda, precio mediano de venta y mediano de compra
# in: dataframe, moneda, Fecha(str): 'XXXX-MM-DD'
def estadisticas_por_fecha(df, moneda, fecha):
    df = numerizar_columnas(df, ['receive', 'amount'])
    # Convertir a formato datetime
    df['updated_at'] = pd.to_datetime(df['updated_at'])
    fecha_inicio = pd.to_datetime(fecha).tz_localize('UTC')  # Especificar la zona horaria
    fecha_fin = fecha_inicio + pd.Timedelta(days=1)

    # Filtrar datos por la moneda y el rango de la fecha
    df_filtrado = df[(df['coin'] == moneda) & (df['updated_at'] >= fecha_inicio) & (df['updated_at'] < fecha_fin)]
    
    # Separar datos de compra y venta
    df_ventas = df_filtrado[df_filtrado['type'] == 'sell']
    df_compras = df_filtrado[df_filtrado['type'] == 'buy']

    # Calcular oferta, demanda y precios medios
    oferta = df_ventas['amount'].sum()
    demanda = df_compras['amount'].sum()
    precio_medio_venta = (df_ventas['receive'] / df_ventas['amount']).median() if not df_ventas.empty else None
    precio_medio_compra = (df_compras['receive'] / df_compras['amount']).median() if not df_compras.empty else None
    menor_precio_venta = (df_ventas['receive'] / df_ventas['amount']).min() if not df_ventas.empty else None
    mayor_precio_compra = (df_compras['receive'] / df_compras['amount']).max() if not df_compras.empty else None

    # Resultado
    resultado = {
        'fecha': fecha,
        'moneda': moneda,
        'oferta': oferta,
        'demanda': demanda,
        'precio_estimado_venta': precio_medio_venta,
        'precio_estimado_compra': precio_medio_compra,
        'menor_precio_venta': menor_precio_venta,
        'mayor_precio_compra': mayor_precio_compra,
        'spread': menor_precio_venta - mayor_precio_compra
    }
    
    return resultado