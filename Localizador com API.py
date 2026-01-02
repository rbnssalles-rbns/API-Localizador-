#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from geopy.geocoders import Nominatim
import openrouteservice

# OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# -------------------------------
# Configuração da página
# -------------------------------
st.set_page_config(page_title="Roteirizador Inteligente", layout="wide")

# -------------------------------
# Funções utilitárias
# -------------------------------
def feedback(msg, tipo="info"):
    if tipo == "success":
        st.success(msg)
    elif tipo == "error":
        st.error(msg)
    elif tipo == "warning":
        st.warning(msg)
    else:
        st.info(msg)

def validar_colunas(df, cols):
    faltantes = [c for c in cols if c not in df.columns]
    if faltantes:
        st.error(f"Arquivo inválido. Faltam as colunas: {', '.join(faltantes)}.")
        return False
    return True

# -------------------------------
# Geocodificação com Nominatim
# -------------------------------
@st.cache_data(show_spinner=False)
def geocode_text(endereco, timeout=5):
    geolocator = Nominatim(user_agent="roteirizador_inteligente_app", timeout=timeout)
    try:
        if endereco is None or (isinstance(endereco, float) and np.isnan(endereco)):
            return None, None
        endereco = str(endereco).strip()
        if not endereco:
            return None, None
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        # Não interrompe a execução; retorna None
        return None, None

@st.cache_data(show_spinner=True)
def geocode_dataframe(df, endereco_col="Endereco", sleep_secs=1.0):
    latitudes = []
    longitudes = []
    for _, row in df.iterrows():
        lat, lon = geocode_text(row.get(endereco_col, ""))
        latitudes.append(lat)
        longitudes.append(lon)
        # Respeitar limite de 1 req/s do Nominatim
        time.sleep(sleep_secs)
    return latitudes, longitudes

# -------------------------------
# OpenRouteService - Matriz e Rotas
# -------------------------------
def get_ors_client(api_key: str):
    try:
        if not api_key or len(api_key.strip()) == 0:
            feedback("Informe sua chave da OpenRouteService para otimizar e traçar rotas reais.", "warning")
            return None
        client = openrouteservice.Client(key=api_key.strip())
        return client
    except Exception as e:
        feedback(f"Erro ao inicializar cliente ORS: {e}", "error")
        return None

def calcular_matriz_distancia(client, coords, metric="distance"):
    try:
        # coords: [[lon, lat], ...]
        matriz = client.distance_matrix(
            locations=coords,
            profile="driving-car",
            metrics=[metric],   # 'distance' (km) ou 'duration' (s)
            units="km"
        )
        # retorna matriz de distâncias (km) ou durações (convertidas)
        return matriz.get("distances") if metric == "distance" else matriz.get("durations")
    except Exception as e:
        feedback(f"Falha ao calcular matriz no ORS: {e}", "error")
        return None

def gerar_rota_real(client, coords_lonlat):
    try:
        rota = client.directions(
            coordinates=coords_lonlat,
            profile="driving-car",
            format="geojson"
        )
        caminho = rota["features"][0]["geometry"]["coordinates"]
        return [{"lon": lon, "lat": lat} for lon, lat in caminho]
    except Exception as e:
        feedback(f"Falha ao gerar rota no ORS: {e}", "error")
        return []

# -------------------------------
# Otimização com OR-Tools (TSP)
# -------------------------------
def otimizar_ordem_visita(matriz, metric="distance"):
    """
    matriz: lista de listas (distâncias ou durações). Inclui CD no índice 0.
    metric: 'distance' (km) ou 'duration' (s). OR-Tools espera custo inteiro.
    """
    try:
        n = len(matriz)
        if n == 0:
            return []

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 veículo, começa no nó 0 (CD)
        routing = pywrapcp.RoutingModel(manager)

        # Converter custo para inteiro (evita erros). Se distancia km, usa *1000; se duração em s, usa *1
        multiplier = 1000 if metric == "distance" else 1

        def cost_callback(from_index, to_index):
            f = manager.IndexToNode(from_index)
            t = manager.IndexToNode(to_index)
            val = matriz[f][t] if matriz[f][t] is not None else 0
            return int(val * multiplier)

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5

        solution = routing.SolveWithParameters(search_parameters)
        ordem = []
        if solution:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                ordem.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            return ordem
        else:
            feedback("Otimização não encontrou solução; usando ordem original.", "warning")
            return list(range(n))
    except Exception as e:
        feedback(f"Falha na otimização TSP: {e}. Usando ordem original.", "warning")
        return list(range(len(matriz)))

# -------------------------------
# Sidebar - Configurações
# -------------------------------
st.sidebar.header("📍 Centro de Distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do Centro de Distribuição",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)

st.sidebar.header("🔑 OpenRouteService API Key")
api_key_ors = st.sidebar.text_input(
    "Informe sua chave ORS para otimizar e traçar rotas",
    value="",
    type="password",
    help="Crie uma chave gratuita em https://openrouteservice.org/"
)

st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("🧠 Roteirizador Inteligente (versão estável)")
st.write("Geocodifique clientes, otimize a ordem de visita e gere rotas reais a partir do Centro de Distribuição, com tratamento robusto de erros e interface amigável.")

# -------------------------------
# Geocodificar CD
# -------------------------------
cd_lat, cd_lon = geocode_text(cd_endereco)
if cd_lat and cd_lon:
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.warning("Não foi possível geocodificar o endereço do CD. Você pode continuar e inserir coordenadas manualmente para os clientes.")

# -------------------------------
# Processamento do Excel
# -------------------------------
if arquivo:
    try:
        df = pd.read_excel(arquivo)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        feedback(f"Erro ao ler o Excel: {e}", "error")
        st.stop()

    if not validar_colunas(df, ["Cliente_ID", "Cliente", "Endereco"]):
        st.stop()

    feedback(f"{len(df)} clientes carregados.", "success")

    # Geocodificação de clientes
    with st.spinner("Geocodificando endereços (Nominatim)..."):
        lats, lons = geocode_dataframe(df, endereco_col="Endereco", sleep_secs=1.0)
    df["Latitude"] = lats
    df["Longitude"] = lons

    total = len(df)
    validos = df["Latitude"].notna().sum()
    feedback(f"Coordenadas obtidas para {validos}/{total} clientes.", "info")

    # Inserção manual de coordenadas
    df_faltantes = df[df["Latitude"].isna() | df["Longitude"].isna()]
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente (clientes não geocodificados)")
        for i, row in df_faltantes.iterrows():
            st.markdown(f"**{row['Cliente_ID']} - {row['Cliente']}**")
            lat_in = st.number_input(f"Latitude para {row['Cliente_ID']}", key=f"lat_{i}", value=0.0, format="%.6f")
            lon_in = st.number_input(f"Longitude para {row['Cliente_ID']}", key=f"lon_{i}", value=0.0, format="%.6f")
            if st.button(f"Salvar coordenadas de {row['Cliente_ID']}", key=f"btn_{i}"):
                # Validação simples: latitude [-90, 90], longitude [-180, 180]
                if -90 <= lat_in <= 90 and -180 <= lon_in <= 180:
                    df.at[i, "Latitude"] = lat_in
                    df.at[i, "Longitude"] = lon_in
                    feedback(f"Coordenadas salvas para {row['Cliente_ID']}.", "success")
                else:
                    feedback("Coordenadas inválidas. Verifique os valores.", "error")

    # Filtra clientes com coordenadas válidas
    df_valid = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    if df_valid.empty:
        feedback("Nenhum cliente com coordenadas válidas. Insira coordenadas manualmente ou revise os endereços.", "warning")
        st.stop()

    st.subheader("📥 Visualização e download")
    st.dataframe(df_valid[["Cliente_ID", "Cliente", "Endereco", "Latitude", "Longitude"]])
    st.download_button(
        "Baixar CSV geocodificado",
        data=df_valid.to_csv(index=False).encode("utf-8"),
        file_name="clientes_geocodificados.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Otimização e geração de rota
    # -------------------------------
    st.subheader("🧮 Otimização de rota e traçado")
    client_ors = get_ors_client(api_key_ors)

    # Monta lista de pontos (inclui CD se disponível)
    pontos = []
    if cd_lat is not None and cd_lon is not None:
        pontos.append({"name": "CD", "lat": cd_lat, "lon": cd_lon})
    else:
        feedback("CD sem coordenadas. A otimização começará no primeiro cliente.", "warning")

    # Adiciona clientes válidos
    for _, r in df_valid.iterrows():
        pontos.append({"name": f"{r['Cliente_ID']} - {r['Cliente']}", "lat": r["Latitude"], "lon": r["Longitude"]})

    # Coordenadas no formato [lon, lat]
    coords = [[p["lon"], p["lat"]] for p in pontos]

    ordem_nomes = [p["name"] for p in pontos]  # fallback padrão

    # Se temos cliente ORS, tentamos matriz + TSP
    if client_ors:
        metric_choice = st.radio("Otimizar por", options=["Distância (km)", "Tempo (min)"], index=0)
        metric = "distance" if "Distância" in metric_choice else "duration"

        matriz = calcular_matriz_distancia(client_ors, coords, metric=metric)
        if matriz is not None:
            ordem = otimizar_ordem_visita(matriz, metric=metric)
            # Ajuste para o caso sem CD: manter ordem calculada
            ordem_nomes = [pontos[i]["name"] for i in ordem]
            st.write("Ordem otimizada de visita:", " → ".join(ordem_nomes))
            coords_ordenadas = [coords[i] for i in ordem]
        else:
            feedback("Usando ordem original por falha na matriz.", "warning")
            coords_ordenadas = coords
    else:
        feedback("Sem cliente ORS: usando ordem original.", "warning")
        coords_ordenadas = coords

    # Geração da rota real
    caminho = []
    if client_ors:
        caminho = gerar_rota_real(client_ors, coords_ordenadas)
    else:
        feedback("Sem ORS: a rota será exibida apenas com pontos.", "warning")

    # -------------------------------
    # Mapa interativo
    # -------------------------------
    st.subheader("🗺️ Mapa: clientes e rota")
    # Pontos
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": p["lat"], "lon": p["lon"], "name": p["name"]} for p in pontos],
        get_position='[lon, lat]',
        get_fill_color='[255, 99, 71]',
        get_radius=70,
        pickable=True
    )

    # Rota
    path_data = []
    if caminho:
        path_data = [{"path": [[p["lon"], p["lat"]] for p in caminho], "name": "Rota"}]
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_width=4,
        get_color=[0, 128, 255],
        width_min_pixels=2
    )

    # View inicial
    if pontos:
        center_lat = pontos[0]["lat"]
        center_lon = pontos[0]["lon"]
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11)
    else:
        view_state = pdk.ViewState(latitude=-3.73, longitude=-38.54, zoom=10)

    st.pydeck_chart(pdk.Deck(layers=[scatter, path_layer], initial_view_state=view_state, tooltip={"text": "{name}"}))

else:
    st.info("Importe um arquivo Excel (.xlsx) com as colunas 'Cliente_ID', 'Cliente' e 'Endereco'.")


# In[ ]:





# In[ ]:




