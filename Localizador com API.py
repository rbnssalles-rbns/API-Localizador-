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
from datetime import timedelta

st.set_page_config(page_title="Localizador de Endereços — otimizado", layout="wide")

# -------------------------------
# Configurações ORS
# -------------------------------
API_KEY_ORS = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImZlZTI5OWZiMGU4MzQ0OTg4ZWU1YzdmMjc5OGMyNWQyIiwiaCI6Im11cm11cjY0In0="
try:
    CLIENT_ORS = openrouteservice.Client(key=API_KEY_ORS)
except Exception as e:
    CLIENT_ORS = None
    st.warning(f"Falha ao inicializar OpenRouteService: {e}")

# -------------------------------
# Utilitários de paginação
# -------------------------------
def paginate_df(df, page_size=50, key_prefix=""):
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = st.number_input(
        f"Página (1–{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page"
    )
    start = (page - 1) * page_size
    end = start + page_size
    st.caption(f"Mostrando linhas {start+1}–{min(end, total_rows)} de {total_rows}")
    return df.iloc[start:end]

# -------------------------------
# Cache: geocodificação OSM para um endereço
# -------------------------------
@st.cache_data(show_spinner=False)
def geocode_osm_cached(endereco):
    geolocator = Nominatim(user_agent="localizador_enderecos_otimizado", timeout=5)
    try:
        if not endereco or str(endereco).strip() == "":
            return None, None
        location = geolocator.geocode(endereco)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception:
        return None, None

# -------------------------------
# Cache: leitura do Excel
# -------------------------------
@st.cache_data(show_spinner=True)
def load_excel(file_bytes):
    return pd.read_excel(file_bytes)

# -------------------------------
# Cache: geocodificação em lote para DataFrame (apenas faltantes)
# -------------------------------
@st.cache_data(show_spinner=True)
def geocode_dataframe_cached(df, endereco_col="Endereco", sleep_secs=0.5):
    lats = []
    lons = []
    for _, row in df.iterrows():
        addr = str(row.get(endereco_col, "")).strip()
        if not addr:
            lats.append(np.nan)
            lons.append(np.nan)
            continue
        lat, lon = geocode_osm_cached(addr)
        lats.append(lat)
        lons.append(lon)
        # Pequena pausa para respeitar limites; se você tiver cache aquecido, isso é rápido
        time.sleep(sleep_secs)
    return lats, lons

# -------------------------------
# Rota real (CD -> clientes na ordem)
# -------------------------------
def gerar_rota_real(cd_lat, cd_lon, pontos):
    if CLIENT_ORS is None or not pontos:
        return []
    coords = [[cd_lon, cd_lat]] + [[p["lon"], p["lat"]] for p in pontos]
    try:
        rota = CLIENT_ORS.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        caminho = rota['features'][0]['geometry']['coordinates']
        return [{"lon": lon, "lat": lat} for lon, lat in caminho]
    except Exception as e:
        st.warning(f"Erro ao gerar rota: {e}")
        return []

# -------------------------------
# Tempo e distância (CD -> cliente)
# -------------------------------
@st.cache_data(show_spinner=True)
def calcular_tempo_distancia(cd_lat, cd_lon, pontos):
    resultados = []
    if CLIENT_ORS is None or cd_lat is None or cd_lon is None or not pontos:
        for p in pontos:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        return pd.DataFrame(resultados)

    for p in pontos:
        try:
            rota = CLIENT_ORS.directions(
                coordinates=[[cd_lon, cd_lat], [p["lon"], p["lat"]]],
                profile="driving-car",
                format="geojson"
            )
            summary = rota['features'][0]['properties']['summary']
            duracao_min = int(summary['duration'] / 60)          # segundos → minutos
            distancia_km = round(summary['distance'] / 1000, 2)  # metros → km
            resultados.append({"Cliente": p["name"], "Tempo (min)": duracao_min, "Distância (km)": distancia_km})
        except Exception:
            resultados.append({"Cliente": p["name"], "Tempo (min)": None, "Distância (km)": None})
        # Pausa leve; com cache, chamadas repetidas não ocorrem
        time.sleep(0.1)
    return pd.DataFrame(resultados)

# -------------------------------
# Sidebar: Centro de Distribuição
# -------------------------------
st.sidebar.header("📍 Centro de distribuição")
cd_endereco = st.sidebar.text_input(
    "Endereço do CD",
    "Travessa Francisco Marrocos Portela, Alto Alegre I, Maracanaú - CE, Brasil, 61922-120"
)
cd_lat, cd_lon = geocode_osm_cached(cd_endereco)
if cd_lat is not None and cd_lon is not None:
    st.sidebar.success(f"CD localizado: {cd_lat:.6f}, {cd_lon:.6f}")
else:
    st.sidebar.error("Não foi possível geocodificar o endereço do CD.")

# -------------------------------
# Upload de clientes
# -------------------------------
st.sidebar.header("📂 Importar clientes (.xlsx)")
arquivo = st.sidebar.file_uploader("Selecione um arquivo Excel", type=["xlsx"])

st.title("📍 Localizador de Endereços — versão otimizada")
st.write("Cache de leitura, cache de geocodificação, calendário vetorizado e filtros/paginação para alto volume.")

if arquivo:
    # Leitura com cache
    try:
        df = load_excel(arquivo)
    except Exception as e:
        st.error(f"Erro ao ler o Excel: {e}")
        st.stop()

    # Normalizar colunas
    df.columns = [c.strip() for c in df.columns]

    # Checagens mínimas
    col_base = ["Cliente_ID", "Endereco"]
    faltantes_base = [c for c in col_base if c not in df.columns]
    if faltantes_base:
        st.error(f"Arquivo inválido. Faltam colunas: {', '.join(faltantes_base)}.")
        st.stop()

    # Filtros por rota (prefixo Rxx) e dia da semana, antes da geocodificação para agilizar
    st.subheader("🎛️ Filtros")
    # Derivar coluna Rota do Cliente_ID (assume padrão RNN_CMMM)
    df["Rota"] = df["Cliente_ID"].astype(str).str.extract(r'(R\d+)', expand=False)
    rotas_disponiveis = sorted(df["Rota"].dropna().unique().tolist())
    dias_disponiveis = sorted(df["Dia_da_semana"].dropna().unique().tolist()) if "Dia_da_semana" in df.columns else []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        filtro_rotas = st.multiselect("Filtrar por rotas", rotas_disponiveis, default=rotas_disponiveis)
    with col2:
        filtro_dias = st.multiselect("Filtrar por dia da semana", dias_disponiveis, default=dias_disponiveis)
    with col3:
        page_size = st.selectbox("Linhas por página", [25, 50, 100, 200], index=1)

    df_f = df.copy()
    if filtro_rotas:
        df_f = df_f[df_f["Rota"].isin(filtro_rotas)]
    if "Dia_da_semana" in df_f.columns and filtro_dias:
        df_f = df_f[df_f["Dia_da_semana"].isin(filtro_dias)]

    st.caption(f"Total após filtros: {len(df_f)} clientes")

    # Geocodificar somente os filtrados e faltantes
    st.subheader("📍 Geocodificação otimizada")
    # Se já existir Latitude/Longitude, preservar; geocodificar apenas faltantes
    if "Latitude" not in df_f.columns or "Longitude" not in df_f.columns:
        df_f["Latitude"] = np.nan
        df_f["Longitude"] = np.nan

    faltantes_geo_mask = df_f["Latitude"].isna() | df_f["Longitude"].isna()
    df_geo_target = df_f[faltantes_geo_mask].copy()

    if not df_geo_target.empty:
        geocode_sleep = st.slider("Intervalo por geocodificação (segundos)", 0.0, 1.0, 0.2, 0.1)
        with st.spinner(f"Geocodificando {len(df_geo_target)} endereços (cache ativado)..."):
            lats, lons = geocode_dataframe_cached(df_geo_target, endereco_col="Endereco", sleep_secs=geocode_sleep)
        df_f.loc[df_geo_target.index, "Latitude"] = lats
        df_f.loc[df_geo_target.index, "Longitude"] = lons

    validos = df_f["Latitude"].notna() & df_f["Longitude"].notna()
    st.info(f"Clientes com coordenadas válidas: {validos.sum()}/{len(df_f)}")

    # Inserção manual (paginada)
    df_faltantes = df_f[~validos].copy()
    if not df_faltantes.empty:
        st.subheader("✏️ Inserir coordenadas manualmente (paginação)")
        df_falt_page = paginate_df(df_faltantes[["Cliente_ID", "Cliente", "Endereco"]], page_size=page_size, key_prefix="falt")
        for i, row in df_falt_page.iterrows():
            st.markdown(f"**{row.get('Cliente_ID', i)} - {row.get('Cliente', 'Cliente')}**")
            lat_in = st.number_input(f"Latitude ({row.get('Cliente_ID', i)})", key=f"lat_{i}", value=0.0, format="%.6f")
            lon_in = st.number_input(f"Longitude ({row.get('Cliente_ID', i)})", key=f"lon_{i}", value=0.0, format="%.6f")
            if st.button(f"Salvar coordenadas ({row.get('Cliente_ID', i)})", key=f"btn_{i}"):
                if -90 <= lat_in <= 90 and -180 <= lon_in <= 180:
                    df_f.at[i, "Latitude"] = lat_in
                    df_f.at[i, "Longitude"] = lon_in
                    st.success("Coordenadas salvas.")
                else:
                    st.warning("Coordenadas inválidas.")

    # Visualização paginada dos clientes geocodificados
    st.subheader("📥 Clientes geocodificados (paginação)")
    df_geo_view = df_f[["Cliente_ID", "Cliente", "Endereco", "Rota", "Dia_da_semana", "Latitude", "Longitude"]].copy()
    df_geo_page = paginate_df(df_geo_view, page_size=page_size, key_prefix="geo")
    st.dataframe(df_geo_page, use_container_width=True)
    st.download_button(
        "Baixar CSV geocodificado (após filtros)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="clientes_geocodificados_filtrados.csv",
        mime="text/csv"
    )

    # Mapa e rota (opcional, para performance você pode desativar)
    if cd_lat is not None and cd_lon is not None:
        st.subheader("🗺️ Mapa de clientes e rota real")
        pontos = [
            {"lat": r["Latitude"], "lon": r["Longitude"], "name": f"{r['Cliente_ID']} - {r.get('Cliente', '')}"}
            for _, r in df_f.iterrows()
            if not pd.isna(r["Latitude"]) and not pd.isna(r["Longitude"])
        ]

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=pontos + [{"lat": cd_lat, "lon": cd_lon, "name": "Centro de Distribuição"}],
            get_position='[lon, lat]',
            get_fill_color='[255, 99, 71]',
            get_radius=60,
            pickable=True
        )

        rota_caminho = gerar_rota_real(cd_lat, cd_lon, pontos)
        path_data = []
        if rota_caminho:
            path_data = [{"path": [[p["lon"], p["lat"]] for p in rota_caminho], "name": "Rota CD -> Clientes"}]

        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_width=4,
            get_color=[0, 128, 255],
            width_min_pixels=2
        )

        view_state = pdk.ViewState(latitude=cd_lat, longitude=cd_lon, zoom=10)
        st.pydeck_chart(pdk.Deck(layers=[scatter, path_layer], initial_view_state=view_state, tooltip={"text": "{name}"}))

        # Tempo e distância (paginado)
        st.subheader("⏱️ Tempo e distância do CD até cada cliente (paginado)")
        if pontos:
            df_td = calcular_tempo_distancia(cd_lat, cd_lon, pontos)
            df_td_page = paginate_df(df_td, page_size=page_size, key_prefix="td")
            st.dataframe(df_td_page, use_container_width=True)
        else:
            st.info("Nenhum cliente com coordenadas válidas para calcular tempo e distância.")
    else:
        st.warning("Defina um endereço válido para o Centro de Distribuição.")

    # -------------------------------
    # 📅 Calendário vetorizado e paginado
    # -------------------------------
    st.subheader("📅 Calendário de visitas (vetorizado, até 31/03/2026)")

    cal_cols = ["Cliente_ID", "Cliente", "Dia_da_semana", "Frequencia", "Data_ultima_visita"]
    falt_cal = [c for c in cal_cols if c not in df.columns]
    if falt_cal:
        st.warning(f"Para o calendário, faltam colunas: {', '.join(falt_cal)}.")
    else:
        # Converter Data_ultima_visita (serial Excel ou string)
        df_cal = df_f.copy()
        if np.issubdtype(df_cal["Data_ultima_visita"].dtype, np.number):
            df_cal["Data_ultima_visita"] = pd.to_datetime(df_cal["Data_ultima_visita"], origin="1899-12-30", unit="D", errors="coerce")
        else:
            df_cal["Data_ultima_visita"] = pd.to_datetime(df_cal["Data_ultima_visita"], errors="coerce")

        # Frequência válida
        df_cal["Frequencia"] = pd.to_numeric(df_cal["Frequencia"], errors="coerce").astype("Int64")
        df_cal = df_cal.dropna(subset=["Data_ultima_visita", "Frequencia"])
        df_cal = df_cal[df_cal["Frequencia"].isin([7, 14, 30])]

        # Próxima visita (vetorizado)
        df_cal["Proxima_visita"] = df_cal["Data_ultima_visita"] + pd.to_timedelta(df_cal["Frequencia"], unit="D")

        # Geração eficiente de visitas até fim_projecao usando list comprehension
        fim_projecao = pd.to_datetime("2026-03-31")
        registros = []
        # Para grande volume, list comprehension é eficiente o suficiente
        for _, r in df_cal.iterrows():
            start_date = r["Data_ultima_visita"]
            freq = int(r["Frequencia"])
            # Gerar sequência de datas
            d = start_date
            while d <= fim_projecao:
                registros.append((r["Cliente_ID"], r.get("Cliente", ""), r.get("Dia_da_semana", ""), d.date()))
                d += timedelta(days=freq)

        calendario_df = pd.DataFrame(registros, columns=["Cliente_ID", "Cliente", "Dia_da_semana", "Data_visita"])
        if calendario_df.empty:
            st.info("Nenhuma visita gerada. Verifique frequências e datas.")
        else:
            # Ordenar
            calendario_df = calendario_df.sort_values(by=["Data_visita", "Dia_da_semana", "Cliente_ID"]).reset_index(drop=True)

            # Paginação
            df_cal_page = paginate_df(calendario_df, page_size=page_size, key_prefix="cal")
            st.dataframe(df_cal_page, use_container_width=True)

            # Download
            st.download_button(
                "Baixar calendário de visitas (CSV)",
                data=calendario_df.to_csv(index=False).encode("utf-8"),
                file_name="calendario_visitas.csv",
                mime="text/csv"
            )

else:
    st.warning("Importe um arquivo Excel (.xlsx) com pelo menos as colunas 'Cliente_ID' e 'Endereco'.")


# In[ ]:





# In[ ]:




