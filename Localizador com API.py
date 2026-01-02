#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from geopy.geocoders import Nominatim
import openrouteservice

st.set_page_config(page_title="Localizador de Endereços", layout="wide")

# -------------------------------
# Centro de Distribuição
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do Centro de Distribuição",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)

# Geocodificação via Nominatim (OSM)
def geocode_osm(endereco):
    geolocator = Nominatim(user_agent="localizador_enderecos", timeout=5)
    try:
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.write(f"Erro na geocodificação: {e}")
        return None, None

cd_lat, cd_lon = None, None
lat, lon = geocode_osm(cd_endereco)
if lat and lon:
    cd_lat, cd_lon = lat, lon
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível geocodificar o endereço do CD.")

# -------------------------------
# Upload de clientes
# -------------------------------
st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("📍 Localizador de Endereços")
st.write("Carregue um Excel com os clientes e endereços para geocodificar e traçar rotas reais a partir do Centro de Distribuição.")

@st.cache_data
def geocode_dataframe_osm(df, endereco_col="Endereco"):
    results = []
    geolocator = Nominatim(user_agent="localizador_enderecos", timeout=5)
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
        time.sleep(1)  # respeitar limite de 1 req/s
    return results

# -------------------------------
# Função para gerar rota real com OpenRouteService
# -------------------------------
def gerar_rota_real(cd_lat, cd_lon, pontos, api_key):
    client = openrouteservice.Client(key=api_key)
    coords = [[cd_lon, cd_lat]] + [[p["lon"], p["lat"]] for p in pontos]
    try:
        rota = client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        caminho = rota['features'][0]['geometry']['coordinates']
        return [{"lon": lon, "lat": lat, "name": "Rota"} for lon, lat in caminho]
    except Exception as e:
        st.error(f"Erro ao gerar rota: {e}")
        return []

# -------------------------------
# Processamento do Excel
# -------------------------------
if arquivo:
    df = pd.read_excel(arquivo)
    df.columns = [c.strip() for c in df.columns]

    if "Cliente_ID" not in df.columns or "Endereco" not in df.columns:
        st.error("Arquivo inválido. É necessário conter as colunas 'Cliente_ID' e 'Endereco'.")
        st.stop()

    st.success(f"{len(df)} clientes carregados.")

    with st.spinner("Geocodificando endereços com OpenStreetMap..."):
        coords = geocode_dataframe_osm(df, endereco_col="Endereco")
    df["Latitude"], df["Longitude"] = zip(*coords)

    total = len(df)
    validos = df["Latitude"].notna().sum()
    st.info(f"Coordenadas obtidas para {validos}/{total} clientes.")

    # -------------------------------
    # ✏️ Inserção manual de coordenadas
    # -------------------------------
    df_faltantes = df[df["Latitude"].isna() | df["Longitude"].isna()]
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente")
        for i, row in df_faltantes.iterrows():
            st.markdown(f"**{row['Cliente_ID']} - {row['Cliente']}**")
            lat = st.number_input(f"Latitude para {row['Cliente_ID']}", key=f"lat_{i}", value=0.0)
            lon = st.number_input(f"Longitude para {row['Cliente_ID']}", key=f"lon_{i}", value=0.0)
            if st.button(f"Salvar coordenadas de {row['Cliente_ID']}", key=f"btn_{i}"):
                df.at[i, "Latitude"] = lat
                df.at[i, "Longitude"] = lon
                st.success(f"Coordenadas salvas para {row['Cliente_ID']}")

    # -------------------------------
    # Download dos resultados
    # -------------------------------
    st.subheader("📥 Baixar resultado geocodificado")
    st.dataframe(df.head(10))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV geocodificado", data=csv_bytes, file_name="clientes_geocodificados.csv", mime="text/csv")

    # -------------------------------
    # Mapa e rota real
    # -------------------------------
    if cd_lat is not None and cd_lon is not None:
        st.subheader("🗺️ Mapa de clientes e rota real")
        pontos = [
            {"lat": r["Latitude"], "lon": r["Longitude"], "name": f"{r['Cliente_ID']} - {r['Cliente']}"}
            for _, r in df.iterrows() if not pd.isna(r["Latitude"]) and not pd.isna(r["Longitude"])
        ]

        rota = gerar_rota_real(cd_lat, cd_lon, pontos, api_key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZlZTI5OWZiMGU4MzQ0OTg4ZWU1YzdmMjc5OGMyNWQyIiwiaCI6Im11cm11cjY0In0=")
        path_data = [{
            "path": [[p["lon"], p["lat"]] for p in rota],
            "name": "Rota CD -> Clientes"
        }]

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=pontos + [{"lat": cd_lat, "lon": cd_lon, "name": "Centro de Distribuição"}],
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




