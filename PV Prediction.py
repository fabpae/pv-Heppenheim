import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from pvlib import location, irradiance, atmosphere, temperature
import datetime
import pytz

# 1. APP-KONFIGURATION
st.set_page_config(page_title="PV Heppenheim Pro", layout="centered")

# --- PASSWORT ABFRAGE ---
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #f1c40f;'>☀️ PV Heppenheim Login</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Passwort:", type="password", key="password_input")
        if pwd:
            if pwd == st.secrets["password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Passwort falsch.")
        return False
    return True

if not check_password():
    st.stop()

# --- PARAMETER ---
ALBEDO = 0.2
TURBIDITY_MONTHLY = [2.1, 2.2, 2.5, 2.9, 3.2, 3.4, 3.5, 3.3, 2.9, 2.6, 2.3, 2.1]

configs = [
    {"name": "Hausdach", "lat": 49.649865, "lon": 8.631587, "wp": 435, "num": 17, "tilt": 34, "azi": 170, "color": "#f1c40f", "shade": None},
    {"name": "Garage Straße", "lat": 49.64980, "lon": 8.631416, "wp": 435, "num": 9, "tilt": 34, "azi": 210, "color": "#e67e22", "shade": None},
    {"name": "Garage Hof", "lat": 49.649872, "lon": 8.631442, "wp": 435, "num": 9, "tilt": 34, "azi": 30, "color": "#d35400", 
     "shade": {"azi_min": 150, "azi_max": 280, "elev_limit": 35}} 
]

@st.cache_data(ttl=3600)
def get_weather_dwd(lat, lon, start, end):
    try:
        url = (f"https://api.open-meteo.com/v1/dwd-icon?latitude={lat}&longitude={lon}"
               f"&hourly=cloudcover,temperature_2m,windspeed_10m&start_date={start}&end_date={end}&timezone=Europe%2FBerlin")
        res = requests.get(url, timeout=15).json()
        if 'hourly' not in res: return None
        return pd.DataFrame({
            'cloud': np.array(res['hourly']['cloudcover']),
            'temp_air': np.array(res['hourly']['temperature_2m']),
            'wind': np.array(res['hourly']['windspeed_10m']) / 3.6
        })
    except: return None

# --- UI ---
START_DATE = st.date_input("Startdatum", datetime.date.today())
if START_DATE:
    # Fix: Exakt 3 Tage (72 Stunden) ab Startdatum 00:00 Uhr
    END_DATE = START_DATE + datetime.timedelta(days=2)
    tz = pytz.timezone('Europe/Berlin')
    
    # Zeitbereich exakt auf 72 Stunden festlegen
    times = pd.date_range(start=pd.Timestamp(START_DATE).tz_localize(tz), periods=72, freq='h')

    weather = get_weather_dwd(configs[0]['lat'], configs[0]['lon'], START_DATE, END_DATE)
    
    if weather is not None:
        # Wetterdaten auf die Länge von 'times' kürzen (falls API mehr liefert)
        weather = weather.iloc[:len(times)]
        
        site = location.Location(configs[0]['lat'], configs[0]['lon'], tz='Europe/Berlin', altitude=100)
        solpos = site.get_solarposition(times)
        dni_extra = irradiance.get_extra_radiation(times)
        
        rel_airmass = atmosphere.get_relative_airmass(solpos['zenith'])
        am_abs = atmosphere.get_absolute_airmass(rel_airmass)
        linke_turbidity = TURBIDITY_MONTHLY[START_DATE.month - 1]

        ergebnisse = {}
        for f in configs:
            cs = site.get_clearsky(times, model='ineichen', linke_turbidity=linke_turbidity)
            cloud_factor = weather['cloud'].values / 100
            
            ghi_adj = cs['ghi'].values * (1 - 0.75 * (cloud_factor ** 3.4))
            dni_adj = cs['dni'].values * (1 - cloud_factor**2)
            dhi_adj = np.maximum(ghi_adj - (dni_adj * np.cos(np.radians(solpos['zenith'].values))), 
                                 cs['dhi'].values * (0.3 + 0.7 * cloud_factor))

            if f['shade']:
                s = f['shade']
                mask = (solpos['azimuth'] > s['azi_min']) & (solpos['azimuth'] < s['azi_max']) & (solpos['elevation'] < s['elev_limit'])
                dni_adj[mask] = 0

            poa = irradiance.get_total_irradiance(
                f['tilt'], f['azi'], solpos['zenith'], solpos['azimuth'],
                dni_adj, ghi_adj, dhi_adj, dni_extra=dni_extra, model='perez', albedo=ALBEDO
            )

            t_cell = temperature.faiman(poa['poa_global'], weather['temp_air'].values, weather['wind'].values)
            f_temp = 1 + -0.0035 * (t_cell.values - 25)
            f_spectral = np.maximum(0.8, 1 - (am_abs.values / 150))
            f_lowlight = np.where(poa['poa_global'].values < 50, 0.93, 1.0)

            # Korrektur: .values nutzen um Shape/Index-Fehler zu vermeiden
            prod = (poa['poa_global'].values / 1000) * ((f['wp'] * f['num']) / 1000) * 0.93 * f_temp * f_spectral * f_lowlight
            ergebnisse[f['name']] = np.nan_to_num(prod)

        df_results = pd.DataFrame(ergebnisse, index=times)
        
        # Einheitliche Berechnung der Tagessummen
        tages_summen = df_results.sum(axis=1).groupby(df_results.index.date).sum()

        # --- ANZEIGE ---
        # Oben: Formatierung auf 1 Nachkommastelle
        header_str = " | ".join([f"{d.strftime('%d.%m.')}: {s:.1f} kWh" for d, s in tages_summen.items()])
        st.markdown(f"### ☀️ {header_str}")
        
        # Grafik 1: 3-Tage Balken
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        mask_day = solpos['elevation'] > 0
        df_plot = df_results[mask_day]
        if not df_plot.empty:
            df_plot.plot(kind='bar', stacked=True, ax=ax1, color=[f['color'] for f in configs], width=0.8)
            # X-Achsen Beschriftung fixen (nur alle 3 Stunden eine Zahl)
            labels = [t.strftime("%H") if t.hour % 3 == 0 else "" for t in df_plot.index]
            ax1.set_xticklabels(labels, rotation=0)
            st.pyplot(fig1)

        # Grafik 2: Detail Heute
        st.subheader(f"Detailansicht Heute")
        df_today = df_results[df_results.index.date == START_DATE]
        if not df_today.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            x = np.arange(len(df_today))
            b = np.zeros(len(df_today))
            for f in configs:
                ax2.fill_between(x, b, b + df_today[f['name']], color=f['color'], label=f['name'], alpha=0.8)
                b += df_today[f['name']]
            
            # Live-Linie
            if START_DATE == datetime.date.today():
                now = datetime.datetime.now(tz)
                idx = now.hour + now.minute/60
                if 0 <= idx <= 23:
                    ax2.axvline(idx, color='red', lw=2, linestyle='--')
                    ax2.text(idx+0.2, max(b)*0.8 if max(b)>0 else 1, "JETZT", color='red', fontweight='bold')
            
            ax2.set_xticks(np.arange(0, 25, 2))
            ax2.set_xticklabels([f"{i}h" for i in range(0, 25, 2)])
            ax2.set_xlim(5, 21)
            ax2.legend(loc='upper left', fontsize='small')
            st.pyplot(fig2)

        # Unten: Wert aus der tages_summen Serie ziehen (garantiert identisch mit oben)
        ertrag_heute = tages_summen.get(START_DATE, 0.0)
        st.success(f"Prognostizierter Ertrag für {START_DATE.strftime('%d.%m.')}: {ertrag_heute:.1f} kWh")
    else:
        st.error("Wetterdaten nicht verfügbar.")
