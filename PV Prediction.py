import streamlit as st

# --- PASSWORT ABFRAGE ---
def check_password():
    """Gibt True zurück, wenn das Passwort korrekt ist."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Passwort aus dem Speicher löschen
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Erstmaliges Anzeigen des Login-Feldes
        st.text_input("Bitte Passwort eingeben", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Passwort war falsch
        st.text_input("Bitte Passwort eingeben", type="password", on_change=password_entered, key="password")
        st.error("😕 Passwort falsch.")
        return False
    else:
        # Passwort korrekt
        return True

if not check_password():
    st.stop()  # Stoppt das Skript hier, wenn nicht eingeloggt

# AB HIER FOLGT DEIN RESTLICHER CODE (st.set_page_config, Berechnungen, etc.)




import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from pvlib import location, irradiance, atmosphere, temperature
import datetime

# App-Konfiguration
st.set_page_config(page_title="PV Heppenheim", layout="centered")

# CSS für Mobile-Optimierung
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #f1c40f; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- EINSTELLUNGEN ---
configs = [
    {"name": "Hausdach", "lat": 49.649865, "lon": 8.631587, "wp": 435, "num": 17, "tilt": 34, "azi": 170, "color": "#f1c40f", "shade": None},
    {"name": "Garage Straße", "lat": 49.64980, "lon": 8.631416, "wp": 435, "num": 9, "tilt": 34, "azi": 210, "color": "#e67e22", "shade": None},
    {"name": "Garage Hof", "lat": 49.649872, "lon": 8.631442, "wp": 435, "num": 9, "tilt": 34, "azi": 30, "color": "#d35400", 
     "shade": {"azi_min": 150, "azi_max": 280, "elev_limit": 35}} 
]
EFFICIENCY = 0.85
TEMP_COEFF = -0.0035

def get_weather(lat, lon, start, end):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=cloudcover,temperature_2m,windspeed_10m&start_date={start}&end_date={end}&timezone=Europe%2FBerlin")
    res = requests.get(url, timeout=10).json()
    return pd.DataFrame({
        'cloud': np.array(res['hourly']['cloudcover']),
        'temp_air': np.array(res['hourly']['temperature_2m']),
        'wind': np.array(res['hourly']['windspeed_10m']) / 3.6
    })

# --- UI ELEMENTE ---
# Wir nutzen den Datumswähler als direkten Trigger
START_DATE = st.date_input("Startdatum wählen", datetime.date.today())
run_btn = st.button("Prognose aktualisieren")

# --- BERECHNUNG ---
# Die Berechnung läuft jetzt immer, wenn ein START_DATE existiert
if START_DATE:
    END_DATE = START_DATE + datetime.timedelta(days=2)
    
    # Zeitachse fixiert
    times = pd.date_range(start=pd.Timestamp(START_DATE).tz_localize('Europe/Berlin'), 
                          end=pd.Timestamp(END_DATE).tz_localize('Europe/Berlin') + pd.Timedelta(hours=23), 
                          freq='h')

    site_loc = location.Location(configs[0]['lat'], configs[0]['lon'], tz='Europe/Berlin')
    solpos = site_loc.get_solarposition(times)
    weather = get_weather(configs[0]['lat'], configs[0]['lon'], START_DATE, END_DATE)
    
    # Wetterdaten auf Indexlänge kürzen
    weather = weather.iloc[:len(times)]

    am_abs = atmosphere.get_absolute_airmass(atmosphere.get_relative_airmass(solpos['zenith'])).values
    ergebnisse = {}
    gesamt_ertrag = np.zeros(len(times))

    for f in configs:
        cs = site_loc.get_clearsky(times)
        f_c = (1 - (weather['cloud'].values / 100 * 0.75))
        dni_adj = cs['dni'].values * (f_c ** 1.8)
        ghi_adj = cs['ghi'].values * f_c
        dhi_adj = cs['dhi'].values * (f_c ** 0.5)
        
        dni_final = dni_adj.copy()
        if f['shade']:
            s = f['shade']
            shade_mask = (solpos['azimuth'].values > s['azi_min']) & (solpos['azimuth'].values < s['azi_max']) & (solpos['elevation'].values < s['elev_limit'])
            dni_final[shade_mask] = 0
        
        poa = irradiance.get_total_irradiance(f['tilt'], f['azi'], solpos['zenith'].values, solpos['azimuth'].values, dni_final, ghi_adj, dhi_adj)
        poa_glob = poa['poa_global'].values if hasattr(poa['poa_global'], 'values') else poa['poa_global']
        t_cell = temperature.faiman(poa_glob, weather['temp_air'].values, weather['wind'].values)
        f_temp = 1 + TEMP_COEFF * (t_cell - 25)
        f_spectral = np.maximum(0.9, 1 - (am_abs / 60))
        
        prod = (poa_glob / 1000) * ((f['wp'] * f['num']) / 1000) * EFFICIENCY * f_temp * f_spectral
        ergebnisse[f['name']] = np.nan_to_num(prod)
        gesamt_ertrag += ergebnisse[f['name']]

    gesamt_series = pd.Series(gesamt_ertrag, index=times)
    tages_summen = gesamt_series.groupby(gesamt_series.index.date).sum()

    # --- DYNAMISCHER HEADER ---
    header_str = " | ".join([f"{d.strftime('%d.%m.')}: {s:.1f} kWh" for d, s in tages_summen.items()])
    st.markdown(f"### ☀️ {header_str}")
    
    # --- GRAFIK ---
    mask = solpos['elevation'].values > 0
    times_plot = times[mask]
    
    # Wichtig: Explizite Figure Erstellung für Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(sum(mask))
    
    for f in configs:
        vals = ergebnisse[f['name']][mask]
        ax.bar(range(len(times_plot)), vals, bottom=bottom, label=f['name'], color=f['color'], width=0.8)
        bottom += vals

    # Tagestrennung
    unique_dates = np.unique(times_plot.date)
    for i, d in enumerate(unique_dates):
        day_indices = np.where(times_plot.date == d)[0]
        if i < len(unique_dates) - 1:
            ax.axvline(day_indices[-1]+0.5, color='black', linestyle='--', alpha=0.3)
        # Datum über die Balken schreiben
        ax.text(day_indices[len(day_indices)//2], max(bottom)*1.1, d.strftime("%d.%m."), ha='center', fontweight='bold')

    ax.set_ylabel("kWh")
    ax.set_ylim(0, max(bottom)*1.2) # Platz für Datumstext oben
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
    
    # X-Achsen Beschriftung (Jede 2. Stunde anzeigen)
    plt.xticks(range(0, len(times_plot), 2), times_plot.strftime("%Hh")[::2], fontsize=8)
    
    # Hier wird die Grafik an Streamlit übergeben
    st.pyplot(fig)
    
    st.success(f"Gesamtertrag: {gesamt_series.sum():.2f} kWh")
else:
    st.write("Bitte wählen Sie ein Datum aus.")