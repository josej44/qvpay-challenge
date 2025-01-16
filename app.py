import time
from turtle import pd
import numpy as np
import streamlit as st
import requests
import json
import os
from pandas import json_normalize
from functions import (data_general, datos_de_mercado_x_moneda, graficar_precios_medianos,
                       graficar_razon_oferta_demanda_24h, graficar_spread, graficar_spread_approx,
                       graficar_spread_por_usuario_mm_checked, print_df_mm, kmeans_sobre_mm,
                       seleccion_de_monedas, estadisticas_por_fecha)

HEADERS = {}

# Función para manejar la autenticación
def login_form():
    global HEADERS
    # URL del endpoint para la autenticación
    login_url = "https://qvapay.com/api/auth/login"

    def login(email, password):
        # Datos para autenticación
        login_data = {
            "email": email,
            "password": password
        }

        # Realizar solicitud POST
        response = requests.post(login_url, json=login_data)

        # Verificar respuesta
        if response.status_code == 200:
            data = response.json()
            token = data.get("accessToken")
            return token
        else:
            return None

    # Formulario de login
    st.title("Inicio de Sesión")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")
    button_placeholder = st.empty()
    button_enter = st.button("Entrar", key="enter", on_click=lambda: st.session_state.update({"entered": True}))
    if button_placeholder.button('Iniciar Sesión'):
        token = login(email, password)
        if token:
            st.success("Inicio de sesión exitoso.")
            HEADERS["Authorization"] = f"Bearer {token}"
            st.session_state["headers"] = HEADERS
            st.session_state["authenticated"] = True
            st.session_state["token"] = token
        else:
            st.error("Credenciales inválidas. Intente nuevamente.")

# Función para descargar datos
def download_data():
    ###################################
    # GUARDAR TODA LA DATA EN UN JSON
    ###################################
    base_url = "https://qvapay.com/api/p2p/index"  # Cambia con el URL correcto de la API
    all_data = []
    url = base_url
    retry_attempts = 0

    # Crear un área de logger en la página
    logger = st.empty()

    while url:
        logger.write(f"Obteniendo datos de {url}...")
        response = requests.get(url, headers=st.session_state["headers"])
        time.sleep(np.random.randint(2))
        if response.status_code == 200:
            try:
                page_data = response.json()
                all_data.extend(page_data['data'])  # Añadir datos
                url = page_data.get("next_page_url")  # Próxima página
                logger.write(f"Descargados {len(page_data['data'])} registros.")
                retry_attempts = 0
            except ValueError as e:
                st.error(f"Error al decodificar JSON: {e}")
                break
        elif response.status_code == 429:
            retry_attempts += 1
            wait_time = np.random.randint(4) * retry_attempts
            logger.write(f"Error 429: Demasiadas solicitudes. Esperando {wait_time} segundos...")
            time.sleep(wait_time)
        else:
            st.error(f"Error al obtener datos: {response.status_code}")
            logger.write(response.text)
            break

    # Guardar datos
    if all_data:
        if not os.path.exists("data"):
            os.makedirs("data")
        with open("data/all_data.json", "w") as f:
            json.dump(all_data, f)
        st.success("Datos descargados y guardados correctamente.")
        logger.write("Todos los datos han sido guardados.")
    else:
        st.warning("No se descargaron datos.")
        
def load_data():
    # Lee todos los archivos en la carpeta data y los carga en un solo DataFrame
    data = []
    for file in os.listdir("data"):
        if file.endswith(".json"):
            with open(f"data/{file}", "r") as f:
                data.extend(json.load(f))
    data = json_normalize(data, sep="_", record_path=None)
    data = data.drop_duplicates(subset=['uuid'])
    return data, True
        

# Lógica principal
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "entered" not in st.session_state:
        st.session_state["entered"] = False
    if "selected_coin" not in st.session_state:
        st.session_state["selected_coin"] = None

    if not st.session_state["entered"]:
        login_form()
    else:
        st.title("Descargar Datos")
        st.text("Haz clic en el botón para descargar los datos más recientes, o presiona el botón para cargar datos existentes.")
        data = [None, False]
        if st.button("Descargar", disabled=not st.session_state["authenticated"]):
            download_data()
            data = load_data()
            st.write(f"Se han cargado {len(data[0])} registros en total.")
        elif st.button("Cargar Datos Existentes"):
            data = load_data()
            st.write(f"Se han cargado {len(data[0])} registros.")
        if data[1] or 'data' in st.session_state:
            st.session_state['data'] = data[0] if data[1] else st.session_state['data']
            st.title("Análisis de Datos")
            st.write("A continuación se muestra una tabla con los datos descargados:")
            with st.expander("Ver Datos"):
                st.write(st.session_state['data'])
            with st.expander("Ver Estadísticas"):
                st.session_state['general_data'] = data_general(st.session_state['data']) if 'general_data' not in st.session_state else st.session_state['general_data']
                st.write(st.session_state['general_data'])
            st.title("Análisis de Monedas")
            st.session_state['coins'] = st.session_state['data']['coin'].unique()
            st.session_state["selected_coin"] = st.selectbox('Selecciona una moneda', st.session_state['coins'], index=st.session_state['coins'].tolist().index(st.session_state["selected_coin"]) if st.session_state["selected_coin"] else 4)
            with st.expander("Ver Estadísticas de Mercado"):
                st.session_state['market_data'] = datos_de_mercado_x_moneda(st.session_state['data'], st.session_state["selected_coin"]) 
                #st.write(st.session_state['market_data'])
                for k,v in st.session_state['market_data'].items():
                    st.markdown(f"**{k.capitalize().replace('_', ' ').replace('ult', 'última').replace('sem', 'semana').replace('mm', 'Market Maker')}**: {v}")
            with st.expander("Ver Gráficos"):
                st.write("A continuación se muestran los gráficos de los datos descargados:")
                st.write("Gráfico de Precios Medianos")
                graficar_precios_medianos(st.session_state['data'], st.session_state["selected_coin"])
                st.write("Gráfico de Razon Oferta/Demanda")
                graficar_razon_oferta_demanda_24h(st.session_state['data'], st.session_state["selected_coin"])
                st.write("Gráfico de Spread")
                graficar_spread(st.session_state['data'], st.session_state["selected_coin"])
                st.write("Gráfico de Spread Aproximado")
                graficar_spread_approx(st.session_state['data'], st.session_state["selected_coin"])
                st.write("Gráfico de Spread por Usuario Market Maker")
                graficar_spread_por_usuario_mm_checked(st.session_state['data'], st.session_state["selected_coin"])
            with st.expander("Clustering"):
                new_df = seleccion_de_monedas(st.session_state['data'], [st.session_state["selected_coin"]])
                try:
                    cluster = kmeans_sobre_mm(new_df, 2)
                    st.write("Cluster usando KMeans, devuelve potenciales Market Makers")
                    print_df_mm(cluster[1], cluster[2])
                    st.write("Cluster usando Agglomerative, devuelve potenciales Market Makers")
                    print_df_mm(cluster[1], cluster[3])
                    st.write("Spread global por cada un de los potenciales Market Makers")
                    graficar_spread_por_usuario_mm_checked(st.session_state['data'], st.session_state["selected_coin"])
                except:
                    st.error("No se pudo realizar el clustering")
            with st.expander("Estadísticas por Fecha"):
                st.write("A continuación se muestran las estadísticas de los datos descargados por fecha:")
                date = st.slider('Selecciona una fecha', 1, 31, 1)
                try:
                    st.session_state['date_stats'] = estadisticas_por_fecha(st.session_state['data'], st.session_state["selected_coin"], f'2025-01-{date}')
                except:
                    st.error( "No se pudo realizar el análisis, intenta con otra fecha")
                    st.session_state['date_stats'] = {}
                for k,v in st.session_state['date_stats'].items():
                    st.markdown(f"**{k.capitalize().replace('_', ' ')}**: {v}")

                
                

if __name__ == "__main__":
    main()
