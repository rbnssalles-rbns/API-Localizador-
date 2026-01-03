#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
import openrouteservice
import requests
import json

st.set_page_config(page_title="Localizador de Endereços — ORS", layout="wide")

# -------------------------------
# Configurações ORS
# -------------------------------
API_KEY_ORS = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZlZTI5OWZiMGU4MzQ0OTg4ZWU1YzdmMjc5OGMyNWQyIiwiaCI6Im11cm11cjY0In0="
CLIENT_ORS = openrouteservice.Client(key=API_KEY_ORS)

# -------------------------------
# Funções utilitárias
# -------------------------------
@st.cache_data(show_spinner=True)
def load_excel(file_bytes):
    return pd.read_excel(file_bytes)

@st.cache_data(show_spinner=True)
def geocode_ors(endereco):
    """Geocodificação via ORS"""
    try:
        resp = CLIENT_ORS.pelias_search(text=endereco)
        if resp and "features" in resp and len(resp["features"]) > 0:
            coords = resp["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]  # lat, lon
        return None, None
    except Exception:
        return None, None

def geocode_dataframe_ors(df, endereco_col="Endereco"):
    lats, lons = [], []
    for _, row in df.iterrows():
        addr = str(row.get(endereco_col, "")).strip()
        if not addr:
            lats.append(np.nan)
            lons.append(np.nan)
            continue
        lat, lon = geocode_ors(addr)
        lats.append(lat)
        lons.append(lon)
        time.sleep(0.1)
    return lats, lons

# -------------------------------
# Sidebar: Centro de Distribuição
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do CD",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)
cd_lat, cd_lon = geocode_ors(cd_endereco)
if cd_lat is not None and cd_lon is not None:
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível geocodificar o endereço do CD.")

# -------------------------------
# Upload de clientes
# -------------------------------
st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("📍 Localizador de Endereços — versão ORS")
st.write("Geocodificação via ORS, geração automática de JSON de otimização e visualização das rotas otimizadas.")

if arquivo:
    df = load_excel(arquivo)
    df.columns = [c.strip() for c in df.columns]

    if "Cliente_ID" not in df.columns or "Endereco" not in df.columns:
        st.error("Arquivo inválido. Faltam colunas obrigatórias.")
        st.stop()

    st.success(f"{len(df)} clientes carregados.")

    # Geocodificação via ORS
    with st.spinner("Geocodificando endereços com ORS..."):
        lats, lons = geocode_dataframe_ors(df, endereco_col="Endereco")
    df["Latitude"], df["Longitude"] = lats, lons

    validos = df["Latitude"].notna().sum()
    st.info(f"Coordenadas obtidas para {validos}/{len(df)} clientes.")

    # -------------------------------
    # Geração do JSON de otimização
    # -------------------------------
    st.subheader("🚚 JSON de otimização para ORS")

    if cd_lat is not None and cd_lon is not None:
        cd_coords = [cd_lon, cd_lat]

        # Criar jobs a partir dos clientes
        jobs = []
        for _, row in df.iterrows():
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                jobs.append({
                    "id": str(row["Cliente_ID"]),
                    "service": 300,
                    "delivery": [1],
                    "location": [row["Longitude"], row["Latitude"]],
                    "skills": [1],
                    "time_windows": [[28800, 43200]]
                })

        # Criar veículos (10 rotas)
        vehicles = []
        for i in range(1, 11):
            vehicles.append({
                "id": i,
                "profile": "driving-car",
                "start": cd_coords,
                "end": cd_coords,
                "capacity": [200],
                "skills": [1],
                "time_window": [28800, 43200]
            })

        # Montar JSON final
        optimization_json = {
            "jobs": jobs,
            "vehicles": vehicles
        }

        # Exibir no app
        st.json(optimization_json)

        # Download do JSON
        json_bytes = json.dumps(optimization_json, indent=2).encode("utf-8")
        st.download_button(
            "Baixar JSON de otimização",
            data=json_bytes,
            file_name="ors_optimization.json",
            mime="application/json"
        )

        # -------------------------------
        # Chamada ao ORS Optimization
        # -------------------------------
        st.subheader("📊 Rotas otimizadas pelo ORS")
        url = "https://api.openrouteservice.org/optimization"
        headers = {
            "Authorization": API_KEY_ORS,
            "Content-Type": "application/json"
        }

        with st.spinner("Enviando JSON para ORS e aguardando resposta..."):
            response = requests.post(url, headers=headers, json=optimization_json)
            result = response.json()

        # Extrair rotas
        routes = []
        for vehicle in result.get("routes", []):
            coords = []
            for step in vehicle.get("steps", []):
                if "location" in step:
                    lon, lat = step["location"]
                    coords.append([lon, lat])
            if coords:
                routes.append(coords)

        # Plotar no mapa
        layers = []

        # Clientes + CD
        points = []
        for job in jobs:
            points.append({"lon": job["location"][0], "lat": job["location"][1], "name": job["id"]})
        points.append({"lon": cd_coords[0], "lat": cd_coords[1], "name": "Centro de Distribuição"})

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position='[lon, lat]',
            get_fill_color='[255, 99, 71]',
            get_radius=60,
            pickable=True
        ))

        # Rotas otimizadas
        for i, route in enumerate(routes):
            path_data = [{"path": route, "name": f"Rota {i+1}"}]
            layers.append(pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_width=4,
                get_color=[0, 128, 255],
                width_min_pixels=2
            ))

        view_state = pdk.ViewState(latitude=cd_lat, longitude=cd_lon, zoom=10)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"}))

    else:
        st.warning("Defina um endereço válido para o Centro de Distribuição.")


# In[ ]:





# In[ ]:




