#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

st.set_page_config(page_title="Roteirizador Inteligente", layout="wide")

# -------------------------------
# Configurações e chave da API
# -------------------------------
API_KEY_ORS = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZlZTI5OWZiMGU4MzQ0OTg4ZWU1YzdmMjc5OGMyNWQyIiwiaCI6Im11cm11cjY0In0="
ORS_CLIENT = openrouteservice.Client(key=API_KEY_ORS)

# -------------------------------
# Geocodificação
# -------------------------------
def geocode_osm(endereco):
    geolocator = Nominatim(user_agent="roteirizador_inteligente", timeout=5)
    try:
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.write(f"Erro na geocodificação: {e}")
    return None, None

@st.cache_data
def geocode_dataframe_osm(df, endereco_col="Endereco"):
    results = []
    geolocator = Nominatim(user_agent="roteirizador_inteligente", timeout=5)
    for _, row in df.iterrows():
        addr = str(row.get(endereco_col, "")).strip()
        if not addr:
            results.append((np.nan, np.nan))
            continue
        try:
            location = geolocator.geocode(addr)
            if location:
                results.append((location.latitude, location.longitude))
            else:
                results.append((np.nan, np.nan))
        except Exception as e:
            st.write(f"Erro na geocodificação: {e}")
            results.append((np.nan, np.nan))
        time.sleep(1)  # respeitar limite do Nominatim
    return results

# -------------------------------
# Matriz de distância com ORS
# -------------------------------
def calcular_matriz_distancia(coords):
    # coords: lista de [lon, lat]
    try:
        matriz = ORS_CLIENT.distance_matrix(
            locations=coords,
            profile='driving-car',
            metrics=['distance'],  # pode usar 'duration' se preferir tempo
            units='km'
        )
        return matriz.get('distances')
    except Exception as e:
        st.error(f"Erro ao calcular matriz de distância: {e}")
        return None

# -------------------------------
# Otimização (TSP) com OR-Tools
# -------------------------------
def otimizar_ordem_visita(matriz):
    # matriz: lista de listas com distâncias
    n = len(matriz)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 veículo, começa no nó 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int((matriz[f][t] or 0) * 1000)  # km -> metros inteiros

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5  # ajuste conforme necessidade

    solution = routing.SolveWithParameters(search_parameters)
    ordem = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            ordem.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        # Fechando no depósito (opcional): ordem.append(ordem[0])
    else:
        st.warning("Não foi possível encontrar solução ótima. Usando ordem original.")
        ordem = list(range(n))
    return ordem

# -------------------------------
# Rota real com ORS Directions
# -------------------------------
def gerar_rota_real_ordenada(coords_lonlat):
    # coords_lonlat: [[lon, lat], ...] na ordem desejada
    try:
        rota = ORS_CLIENT.directions(
            coordinates=coords_lonlat,
            profile='driving-car',
            format='geojson'
        )
        caminho = rota['features'][0]['geometry']['coordinates']  # [[lon, lat], ...]
        return [{"lon": lon, "lat": lat} for lon, lat in caminho]
    except Exception as e:
        st.error(f"Erro ao gerar rota real: {e}")
        return []

# -------------------------------
# UI: Sidebar CD + Upload
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do Centro de Distribuição",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)
cd_lat, cd_lon = geocode_osm(cd_endereco)
if cd_lat and cd_lon:
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível geocodificar o endereço do CD.")

st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("🧠 Roteirizador Inteligente")
st.write("Geocodifique clientes, otimize a ordem de visita e gere rotas reais a partir do Centro de Distribuição.")

# -------------------------------
# Processamento de clientes
# -------------------------------
if arquivo:
    df = pd.read_excel(arquivo)
    df.columns = [c.strip() for c in df.columns]

    if "Cliente_ID" not in df.columns or "Endereco" not in df.columns:
        st.error("Arquivo inválido. É necessário conter as colunas 'Cliente_ID' e 'Endereco'.")
        st.stop()

    st.success(f"{len(df)} clientes carregados.")

    with st.spinner("Geocodificando endereços..."):
        coords_res = geocode_dataframe_osm(df, endereco_col="Endereco")
    df["Latitude"], df["Longitude"] = zip(*coords_res)

    total = len(df)
    validos = df["Latitude"].notna().sum()
    st.info(f"Coordenadas obtidas para {validos}/{total} clientes.")

    # Inserção manual de coordenadas
    df_faltantes = df[df["Latitude"].isna() | df["Longitude"].isna()]
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente (clientes não encontrados)")
        for i, row in df_faltantes.iterrows():
            st.markdown(f"**{row['Cliente_ID']} - {row.get('Cliente', '')}**")
            lat_in = st.number_input(f"Latitude para {row['Cliente_ID']}", key=f"lat_{i}", value=0.0, format="%.6f")
            lon_in = st.number_input(f"Longitude para {row['Cliente_ID']}", key=f"lon_{i}", value=0.0, format="%.6f")
            if st.button(f"Salvar coordenadas de {row['Cliente_ID']}", key=f"btn_{i}"):
                df.at[i, "Latitude"] = lat_in
                df.at[i, "Longitude"] = lon_in
                st.success(f"Coordenadas salvas para {row['Cliente_ID']}")

    # Filtra apenas clientes válidos
    df_valid = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    if df_valid.empty:
        st.warning("Nenhum cliente com coordenadas válidas.")
        st.stop()

    st.subheader("📥 Visualização e download")
    st.dataframe(df_valid[["Cliente_ID", "Cliente", "Endereco", "Latitude", "Longitude"]].head(10))
    st.download_button(
        "Baixar CSV geocodificado",
        data=df_valid.to_csv(index=False).encode("utf-8"),
        file_name="clientes_geocodificados.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Otimização de rota (inclui CD como primeiro nó)
    # -------------------------------
    if cd_lat is not None and cd_lon is not None:
        st.subheader("🧮 Otimização de rota (TSP)")
        # Construir lista de pontos com CD na posição 0
        pontos = [{"name": "CD", "lat": cd_lat, "lon": cd_lon}] + [
            {"name": f"{r['Cliente_ID']} - {r.get('Cliente', '')}", "lat": r["Latitude"], "lon": r["Longitude"]}
            for _, r in df_valid.iterrows()
        ]
        coords = [[p["lon"], p["lat"]] for p in pontos]

        matriz = calcular_matriz_distancia(coords)
        if matriz is None:
            st.stop()

        ordem = otimizar_ordem_visita(matriz)  # ordem de índices incluindo 0 (CD)
        ordem_nomes = [pontos[i]["name"] for i in ordem]
        st.write("Ordem otimizada de visita:", " → ".join(ordem_nomes))

        # Sequência de coordenadas para Directions na ordem ótima
        coords_ordenadas = [coords[i] for i in ordem]

        # -------------------------------
        # Geração da rota real
        # -------------------------------
        st.subheader("🗺️ Mapa de clientes e rota real otimizada")
        caminho = gerar_rota_real_ordenada(coords_ordenadas)
        path_data = [{
            "path": [[p["lon"], p["lat"]] for p in caminho],
            "name": "Rota ótima"
        }]

        # Pontos para o mapa (todos)
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": p["lat"], "lon": p["lon"], "name": p["name"]} for p in pontos],
            get_position='[lon, lat]',
            get_fill_color='[255, 99, 71]',
            get_radius=60,
            pickable=True
        )
        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_width=4,
            get_color=[0, 128, 255],
            width_min_pixels=2
        )
        view_state = pdk.ViewState(latitude=cd_lat, longitude=cd_lon, zoom=11)
        st.pydeck_chart(pdk.Deck(layers=[scatter, path_layer], initial_view_state=view_state, tooltip={"text": "{name}"}))
    else:
        st.warning("Defina um endereço válido para o Centro de Distribuição.")
else:
    st.warning("Importe um arquivo Excel (.xlsx) com as colunas 'Cliente_ID', 'Cliente' e 'Endereco'.")


# In[ ]:





# In[ ]:




