# -*- coding: utf-8 -*-
"""
Hoofdpagina van het hartgezondheid dashboard.
Biedt toegang tot verschillende weergaven en instellingen.

@author: merli and sven
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def detecteer_Rpieken(df, pct_drempel):
    """
    Detecteert R-pieken in het ECG signaal.
    
    Parameters:
    - df: DataFrame met ECG data
    - pct_drempel: Percentage van maximale ECG waarde als drempel
    
    Returns:
    - df_pieken: DataFrame met alleen de gedetecteerde R-pieken
    """
    max_ecg = max(df['ECG'])
    drempelwaarde = pct_drempel * max_ecg
    pieken = find_peaks(df['ECG'], height=drempelwaarde)[0]
    df_pieken = df.iloc[pieken]
    return df_pieken

def initialize_data(uploaded_file=None):
    """
    Initialiseert alle benodigde data voor het dashboard.
    Laadt de ruwe data en berekent afgeleide waarden.
    Slaat alles op in de Streamlit session state.
    """
    if 'data_initialized' not in st.session_state:
        # Drempelwaarde voor R-piek detectie
        st.session_state.rpieken_drempel = 0.6  # Aan te passen naar eigen inzicht

        try:
            # Laad ruwe data en bereken tijdstempels
            if uploaded_file is not None:
                df_ecg = pd.read_csv(uploaded_file, delimiter=',', skipinitialspace=True)
            else:
                df_ecg = pd.read_csv("data Sanne.txt", delimiter=',', skipinitialspace=True)
                
            df_ecg['timestamp'] = (df_ecg['time'] - df_ecg['time'].iloc[0])/1024  # Converteer naar seconden

            # Detecteer R-pieken en bereken hartslag-gerelateerde waarden
            df_ecg_pieken = detecteer_Rpieken(df_ecg, st.session_state.rpieken_drempel)
            df_ecg_pieken['rr'] = df_ecg_pieken['timestamp'].diff()  # RR-intervallen
            df_ecg_pieken['bpm'] = 60 / df_ecg_pieken['rr']  # Hartslag in slagen per minuut
            df_ecg_pieken['gemiddelde_tijd'] = df_ecg_pieken['timestamp'].rolling(window=2).mean()
            df_ecg_pieken = df_ecg_pieken[df_ecg_pieken['rr'].notna()]  # Verwijder NaN waarden

            # Sla bewerkte data op in session state
            st.session_state.df_ecg = df_ecg
            st.session_state.df_ecg_pieken = df_ecg_pieken
            st.session_state.data_initialized = True
            
            return True
        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het laden van het bestand: {str(e)}")
            return False

def homepage():
    """
    Zet de hoofdpagina van het dashboard op.
    Initialiseert de data en toont basis informatie.
    """
    st.set_page_config(page_title="Dashboard hart gezondheid", layout="wide")
    st.title('Dashboard hart gezondheid')
    
    # Sidebar voor gebruikersinformatie en bestandsupload
    with st.sidebar:
        st.header("Dashboard Instellingen")
        
        # Leeftijd input
        if 'user_age' not in st.session_state:
            st.session_state.user_age = 25  # Default waarde
            
        st.session_state.user_age = st.number_input(
            "Wat is uw leeftijd?",
            min_value=1,
            max_value=120,
            value=st.session_state.user_age,
            help="Uw leeftijd wordt gebruikt voor het berekenen van hartslagzones"
        )
        
        st.divider()
        
        # Doel instellingen
        st.header("🎯 Persoonlijke Doelen")
        
        # RMSSD doel
        if 'doelwaarde_rmssd' not in st.session_state:
            st.session_state.doelwaarde_rmssd = 45  # Default waarde
            
        st.session_state.doelwaarde_rmssd = st.number_input(
            "RMSSD doelwaarde (ms)",
            min_value=20,
            max_value=100,
            value=st.session_state.doelwaarde_rmssd,
            help="Streefwaarde voor RMSSD tijdens rust"
        )
        
        # Beweegminuten doel
        if 'doelwaarde_beweging' not in st.session_state:
            st.session_state.doelwaarde_beweging = 150  # Default waarde
            
        st.session_state.doelwaarde_beweging = st.number_input(
            "Wekelijkse beweegminuten doel",
            min_value=30,
            max_value=300,
            value=st.session_state.doelwaarde_beweging,
            help="Aantal minuten beweging per week (intensief telt dubbel)"
        )
        
        # Stressscore doel
        if 'doelwaarde_stress' not in st.session_state:
            st.session_state.doelwaarde_stress = 50  # Default waarde
            
        st.session_state.doelwaarde_stress = st.number_input(
            "Maximale stressscore (%)",
            min_value=20,
            max_value=80,
            value=st.session_state.doelwaarde_stress,
            help="Maximale gewenste stressscore (lager is beter)"
        )
        # Dagelijks stappendoel
        if 'Dagelijks stappendoel' not in st.session_state:
            st.session_state.Dagelijks_stappendoel = 10000
        
        st.session_state.Dagelijks_stappendoel= st.number_input(
            "Dagelijks stappendoel",
            min_value=5000,
            max_value=20000,
            value=st.session_state.Dagelijks_stappendoel,
            step=500,
            help="Streef naar dit aantal stappen per dag. WHO-richtlijn: 10.000 stappen per dag."
        )
        
        st.divider()
        
        # Bestandsupload sectie
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload uw sensordata bestand",
            type=['csv', 'txt'],
            help="Upload een .csv of .txt bestand met uw sensorgegevens"
        )
        
        if uploaded_file is not None:
            try: 
                initialize_data(uploaded_file)
                st.success("✅ Data succesvol geladen!")
            except:
                st.error("❌ Er is iets misgegaan bij het laden van het bestand. Controleer of het bestand het juiste formaat heeft.")
        else:
            if initialize_data():
                st.info("ℹ️ Standaard voorbeelddata geladen. Upload uw eigen data voor persoonlijke analyse.")
    
    # Hoofdinhoud
    st.subheader("Welkom bij uw Hartgezondheid Dashboard")
    st.write("""Dit interactieve dashboard is ontwikkeld om inzicht te geven in uw hartgezondheid 
             op basis van gegevens die zijn verzameld met een draagbare sensor.""") 
    st.write("De sensor registreert analyseerd ECG-signalen, beweging en temperatuur om hartslagvariabiliteit(HRV), stressniveaus en fysieke activiteit te berekenen.")

             
    # Instructies en uitleg
    st.subheader("Wat kunt u hier doen?")   
    st.write("""
    - Bekijk uw hartactiviteit in real-time of per meetmoment.
    - Analyseer stressniveaus aan de hand van HRV-berekeningen zoals RMSSD en SDNN.
    - Volg uw bewegingen zoals stappen, traplopen en herstelmomenten.
    - Visualiseer trends in hartfunctie gekoppeld aan dagelijkse activiteiten.
    - Stel persoonlijke doelen in voor RMSSD, beweging en stress.
    
    Het doel van dit dashboard is om u te ondersteunen bij  meer grip
    op uw gezondheid te krijgen. """)
    st.write("Door patronen in inspanning en herstel te herkennen, kunt u beter begrijpen hoe uw lichaam reageert op stress en beweging.")
    
    
    st.markdown("""
    ###  Hoe gebruikt u dit dashboard?
    1. **Start met uploaden:**
       - Upload uw sensordata via het menu aan de linkerkant
    
    2. **Vul uw gegevens in:**
       - Geef uw leeftijd op voor nauwkeurige hartslagzoneberekeningen
       - Stel uw persoonlijke doelen in voor RMSSD, beweging en stress
       
    
    3. **Kies uw weergave:**
       - **Standaard weergave**: Eenvoudige weergave met basis hartgezondheidsmetrieken
       - **Uitgebreide weergave**: Gedetailleerdere analyses en extra grafieken
    """)
             
    st.write("Kies of u de standaard weergave of de uitgebreide weergave wilt met de knoppen hieronder.")
    st.write("U kunt de keuze altijd nog veranderen in de zijbalk")
    st.write("Als u uitgebreidere informatie wilt over hoe het dashboard achter de schermen werkt, wat het precies meet en hoe de berekeningen gedaan worden? Dan kunt u op informatie klikken.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Standaard weergave"):
            st.switch_page("pages/beginner.py")
    with col2:
        if st.button("Uitgebreide weergave"):
            st.switch_page("pages/Advanced.py")
    with col3:
        if st.button("Informatie"):
            st.switch_page("pages/info.py")
    
    # Disclaimer onderaan de pagina
    st.divider()
    st.caption("""
    **Disclaimer**: Dit dashboard is bedoeld als hulpmiddel ter ondersteuning van uw leefstijl.
    Raadpleeg altijd een arts bij gezondheidsklachten of medische vragen.
    """)

pages = {
    "Home": [
        st.Page(homepage, title="Home pagina")],
    "subpaginas": [
        st.Page("pages/beginner.py", title="Standaard weergave"),
        st.Page("pages/Advanced.py", title="Uitgebreide weergave"),
        st.Page("pages/info.py", title="Informatie"),
    ],
}

# Start de navigatie
pg = st.navigation(pages)
pg.run()
    

