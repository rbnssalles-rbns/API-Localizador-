#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import openrouteservice
import requests
import json
from sklearn.cluster import KMeans

st.set_page_config(page_title="Logística ORS — calendário, regiões e relatórios", layout="wide")

# -----------------------------
# Configuração ORS
# -----------------------------
API_KEY_ORS = "SUA_CHAVE_ORS_AQUI"  # insira sua chave ORS
CLIENT_ORS = openrouteservice.Client(key=API_KEY_ORS)

# -----------------------------
# Utilitários
# -----------------------------
@st.cache_data(show_spinner=True)
def load_excel(file_bytes):
    return pd.read_excel(file_bytes)

def geocode_addr(addr: str):
    try:
        resp = CLIENT_ORS.pelias_search(text=addr)
        if resp and "features" in resp and len(resp["features"]) > 0:
            lon, lat = resp["features"][0]["geometry"]["coordinates"]
            return lat, lon
    except Exception:
        pass
    return None, None

def build_jobs(df_filtered):
    jobs = []
    for _, row in df_filtered.iterrows():
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
            jobs.append({
                "id": str(row["Cliente_ID"]),
                "service": 300,
                "delivery": [1],
                "location": [row["Longitude"], row["Latitude"]],
                "skills": [1],
                "time_windows": [[28800, 64800]]
            })
    return jobs

def build_vehicles(cd_coords, n=10):
    vehicles = []
    for i in range(1, n+1):
        vehicles.append({
            "id": i,
            "profile": "driving-car",
            "start": cd_coords,
            "end": cd_coords,
            "capacity": [200],
            "skills": [1],
            "time_window": [28800, 64800]
        })
    return vehicles

def decode_routes_for_map(result):
    routes = []
    for vehicle in result.get("routes", []):
        coords = []
        for step in vehicle.get("steps", []):
            if "location" in step:
                lon, lat = step["location"]
                coords.append([lon, lat])
        if coords:
            routes.append({"vehicle": vehicle.get("vehicle"), "coords": coords})
    return routes

def build_report(result):
    relatorio = []
    for rota in result.get("routes", []):
        summary = rota.get("summary", {})
        relatorio.append({
            "Veículo": rota.get("vehicle"),
            "Distância (km)": round(summary.get("distance", 0) / 1000, 2),
            "Tempo (h)": round(summary.get("duration", 0) / 3600, 2),
            "Clientes atendidos": max(0, len(rota.get("steps", [])) - 2)
        })
    return pd.DataFrame(relatorio)

def find_unserved_jobs(sent_jobs, result):
    served_ids = set()
    for rota in result.get("routes", []):
        for step in rota.get("steps", []):
            if step.get("type") == "job":
                served_ids.add(str(step.get("id")))
    sent_ids = {job["id"] for job in sent_jobs}
    return sorted(list(sent_ids - served_ids))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input("Endereço do CD", "Travessa Francisco Marrocos Portela, Maracanaú - CE")

cd_lat, cd_lon = geocode_addr(cd_endereco)
if cd_lat is not None and cd_lon is not None:
    st.sidebar.success(f"CD: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível geocodificar o CD. Verifique o endereço.")

st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

# -----------------------------
# Main
# -----------------------------
st.title("📍 Logística ORS — calendário por dia, regiões e relatórios")
dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
dia_selecionado = st.selectbox("Escolha o dia da semana", dias_semana)

if arquivo:
    df = load_excel(arquivo)
    df.columns = [c.strip() for c in df.columns]

    if "Cliente_ID" not in df.columns or "Endereco" not in df.columns:
        st.error("Arquivo inválido. Faltam colunas obrigatórias: Cliente_ID e Endereco.")
        st.stop()

    # Geocodificação automática
    st.subheader("🧭 Geocodificação de clientes via ORS")
    latitudes, longitudes = [], []
    for endereco in df["Endereco"]:
        lat, lon = geocode_addr(endereco)
        latitudes.append(lat)
        longitudes.append(lon)
    df["Latitude"] = latitudes
    df["Longitude"] = longitudes

    df_valid = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    st.success(f"Coordenadas geradas: {len(df_valid)}/{len(df)} clientes.")

    # Clusters de regiões
    st.subheader("🗺️ Agrupamento automático por regiões")
    n_regioes = st.slider("Quantidade de regiões (clusters)", min_value=3, max_value=15, value=6, step=1)

    coords_matrix = df_valid[["Latitude", "Longitude"]].to_numpy()
    if len(coords_matrix) < n_regioes:
        st.warning("Clientes insuficientes para o número de regiões escolhido. Reduza os clusters.")
        st.stop()

    kmeans = KMeans(n_clusters=n_regioes, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coords_matrix)
    df_valid["RegiaoCluster"] = labels

    # Mapear regiões para dias
    st.markdown("Atribua cada região a um dia da semana:")
    regioes = sorted(df_valid["RegiaoCluster"].unique())
    dia_por_regiao = {}
    cols = st.columns(min(5, len(regioes)))
    for idx, reg in enumerate(regioes):
        with cols[idx % len(cols)]:
            dia_por_regiao[reg] = st.selectbox(
                f"Região {reg}",
                dias_semana,
                index=dias_semana.index(dia_selecionado),
                key=f"reg_{reg}"
            )

    regioes_do_dia = [r for r, d in dia_por_regiao.items() if d == dia_selecionado]
    df_dia = df_valid[df_valid["RegiaoCluster"].isin(regioes_do_dia)].copy()

    st.info(f"{len(df_dia)} clientes selecionados para {dia_selecionado} (Regiões: {regioes_do_dia})")

    # Mapa: pins clientes (vermelho) e CD (verde)
    st.subheader("🗺️ Mapa de clientes e CD com pins diferenciados")
    icon_data = []
    for _, row in df_dia.iterrows():
        icon_data.append({
            "position": [row["Longitude"], row["Latitude"]],
            "name": f"Cliente {row['Cliente_ID']} (Região {int(row['RegiaoCluster'])})",
            "icon_data": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/RedDot.svg",
                "width": 128, "height": 128, "anchorY": 128
            }
        })
    if cd_lat is not None and cd_lon is not None:
        icon_data.append({
            "position": [cd_lon, cd_lat],
            "name": "Centro de Distribuição",
            "icon_data": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/7/7c/GreenDot.svg",
                "width": 128, "height": 128, "anchorY": 128
            }
        })

    layers = [
        pdk.Layer(
            "IconLayer",
            data=icon_data,
            get_position="position",
            get_icon="icon_data",
            get_size=4,
            pickable=True
        )
    ]

    default_lat = cd_lat if cd_lat is not None else (df_dia["Latitude"].mean() if not df_dia.empty else 0)
    default_lon = cd_lon if cd_lon is not None else (df_dia["Longitude"].mean() if not df_dia.empty else 0)
    view_state = pdk.ViewState(latitude=default_lat, longitude=default_lon, zoom=11)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"}))

    # JSON de otimização
    st.subheader("🚚 JSON de otimização (apenas clientes do dia)")
    if cd_lat is None or cd_lon is None:
        st.warning("Defina um CD válido para prosseguir com a otimização.")
        st.stop()

    cd_coords = [cd_lon, cd_lat]
    jobs = build_jobs(df_dia)
    vehicles = build_vehicles(cd_coords, n=10)
    optimization_json = {"jobs": jobs, "vehicles": vehicles}
    st.json(optimization_json)

    # Botão para otimizar e visualizar trajetos
    st.subheader("📊 Rotas otimizadas, trajetos e relatórios")
    if st.button("Enviar para ORS e desenhar trajetos"):
        url = "https://api.openrouteservice.org/optimization"
        headers = {"Authorization": API_KEY_ORS, "Content-Type": "application/json"}

        with st.spinner("Otimizando rotas..."):
            response = requests.post(url, headers=headers, json=optimization_json)

        try:
            result = response.json()
        except Exception:
            st.error(f"Erro ao interpretar a resposta (status {response.status_code}).")
            st.stop()

        # Trajetos (CD -> clientes -> CD)
        routes_decoded = decode_routes_for_map(result)
        route_layers = []
        for i, r in enumerate(routes_decoded):
            path_data = [{"path": r["coords"], "name": f"Rota {i+1} (Veículo {r['vehicle']})"}]
            route_layers.append(pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_width=4,
                get_color=[0, 102, 255],
                width_min_pixels=2
            ))

        st.pydeck_chart(pdk.Deck(
            layers=layers + route_layers,
            initial_view_state=view_state,
            tooltip={"text": "{name}"}
        ))

        # Relatórios
        relatorio_df = build_report(result)
        st.markdown("### 📈 Relatório de eficiência por rota")
        if not relatorio_df.empty:
            st.dataframe(relatorio_df, use_container_width=True)
        else:
            st.info("Nenhuma rota retornada.")

        # Não atendidos
        st.markdown("### 🔄 Clientes não atendidos")
        missing_ids = find_unserved_jobs(jobs, result)
        if missing_ids:
            st.warning(f"{len(missing_ids)} clientes não atendidos.")
            st.code(", ".join(missing_ids))
            if not relatorio_df.empty:
                carga_por_veiculo = {row["Veículo"]: row["Clientes atendidos"] for _, row in relatorio_df.iterrows()}
                menor_carga = sorted(carga_por_veiculo.items(), key=lambda x: x[1])[:3]
                st.write("Veículos candidatos a receber reagendamentos:")
                for v, c in menor_carga:
                    st.write(f"- Veículo {v}: {c} clientes atendidos.")
        else:
            st.success("Todos os clientes foram atribuídos.")
else:
    st.info("Faça upload da planilha para continuar.")


# In[ ]:





# In[ ]:




