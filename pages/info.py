# -*- coding: utf-8 -*-
"""
Informatiepagina voor het hartgezondheid dashboard.
Biedt gedetailleerde uitleg over alle functionaliteiten en berekeningen.

@author: merli and sven
"""
import streamlit as st
import plotly.graph_objects as go

def info_page():
    st.set_page_config(page_title="Dashboard Informatie", layout="wide")
    st.title("Informatie over het Dashboard")

    # Inhoudsopgave
    st.sidebar.header("Inhoudsopgave")
    page = st.sidebar.radio("Ga naar:", [
        "Algemene Informatie",
        "Data & Metingen",
        "Hartslagvariabiliteit (HRV)",
        "Bewegingsintensiteit",
        "Stappenteller"
    ])

    if page == "Algemene Informatie":
        st.header("Algemene Informatie")
        
        st.subheader("Over het Dashboard")
        st.write("""
        Dit dashboard is ontwikkeld om inzicht te geven in uw hartgezondheid en bewegingspatronen. 
        Het verwerkt gegevens van een draagbare sensor die verschillende fysiologische signalen meet:
        - ECG (elektrocardiogram)
        - Beweging (accelerometer)
        - Temperatuur
        
        Het dashboard biedt twee weergaven:
        1. **Standaard weergave**: Eenvoudige visualisaties van de belangrijkste gezondheidsmetrieken
        2. **Uitgebreide weergave**: Geavanceerdere analyses en meer gedetailleerde grafieken
        """)

        st.subheader("Hoe te Gebruiken")
        st.write("""
        1. **Voorbereiding**:
           - Doe een meting met de hartslagsensor
           - Exporteer de data naar een .csv of .txt bestand
           - Upload het bestand via het menu in de zijbalk
        
        2. **Persoonlijke Instellingen**:
           - Vul uw leeftijd in voor nauwkeurige hartslagzoneberekeningen
           - De leeftijd wordt gebruikt in de Karvonen-formule voor het bepalen van trainingsintensiteit
        
        3. **Navigatie**:
           - Gebruik de knoppen in de zijbalk om grafieken te tonen/verbergen
           - Kies tussen standaard en uitgebreide weergave
           - Gebruik de uitleg bij elke grafiek voor interpretatie
        """)

    elif page == "Data & Metingen":
        st.header("Data & Metingen")
        
        st.subheader("Sensorgegevens")
        st.write("""
        De sensor verzamelt verschillende soorten data:
        
        **ECG-signaal**:
        - Elektrische activiteit van het hart
        - Sampling rate: 1024 Hz
        - Gebruikt voor het detecteren van hartslagen en HRV-analyse
        
        **Bewegingsdata (Accelerometer)**:
        - Driedimensionale bewegingsdetectie (X, Y, Z-as)
        - Gebruikt voor activiteitsherkenning en stappentelling
        - Helpt bij het bepalen van bewegingsintensiteit
        """)

        st.subheader("Dataverwerking")
        st.write("""
        Het dashboard voert verschillende verwerkingsstappen uit:
        
        1. **Voorbewerking**:
           - Tijdsynchronisatie
           - Ruisfiltering
           - Normalisatie van signalen
        
        2. **Feature Extractie**:
           - R-piek detectie in ECG
           - Berekening van RR-intervallen
           - Bewegingspatroonherkenning
        
        3. **Analyse**:
           - HRV-berekeningen (RMSSD, SDNN)
           - Activiteitsclassificatie
           - Stressniveaubepaling
        """)

    elif page == "Hartslagvariabiliteit (HRV)":
        st.header("Hartslagvariabiliteit (HRV)")
        
        st.subheader("Wat is HRV?")
        st.write("""
        Hartslagvariabiliteit (HRV) is de variatie in tijd tussen opeenvolgende hartslagen. 
        Het is een belangrijke indicator voor:
        - Autonome zenuwstelsel functie
        - Stressniveau
        - Herstel en adaptatie
        - Algemene gezondheid
        """)

        st.subheader("HRV Metrieken")
        st.write("""
        Het dashboard gebruikt twee belangrijke HRV-metrieken:
        
        **1. RMSSD (Root Mean Square of Successive Differences)**
        - Maat voor korte-termijn HRV
        - Indicator voor parasympathische activiteit
        - Interpretatie:
          * Hoge waarden: Goed herstel, lage stress
          * Lage waarden: Verhoogde stress, verminderd herstel
        - Beschikbaar in:
          * Standaard weergave: Gemiddelde tijdens rust
          * Uitgebreide weergave: Live monitoring en verloop over tijd
        
        **2. SDNN (Standard Deviation of NN intervals)**
        - Maat voor totale HRV
        - Weerspiegelt alle cyclische componenten
        - Interpretatie:
          * Hoge waarden: Goede aanpassingscapaciteit
          * Lage waarden: Verminderde hartflexibiliteit
        - Beschikbaar in:
          * Standaard weergave: Gemiddelde waarde
          * Uitgebreide weergave: Live monitoring en verloop over tijd
        """)

        st.subheader("Live Monitoring")
        st.write("""
        In de uitgebreide weergave is live monitoring van hartgegevens beschikbaar:
        
        **Real-time HRV Meters**
        - Toont actuele RMSSD en SDNN waarden
        - Vergelijkt met persoonlijke gemiddelden en leeftijdsreferenties
        - Geeft directe feedback over HRV-status
        - Inclusief interpretatie onder elke meter
        
        **Voordelen van Live Monitoring**
        - Direct inzicht in hartritme variabiliteit
        - Mogelijkheid om acute veranderingen te volgen
        - Directe feedback over stress en herstel
        """)

        st.subheader("Stressscore Berekening")
        st.write("""
        De stressscore wordt berekend op basis van HRV-waarden:
        1. Normalisatie van RMSSD-waarden
        2. Omzetting naar 0-100 schaal
        3. Inverse relatie met stress:
           - Lage HRV = Hoge stressscore
           - Hoge HRV = Lage stressscore
        """)

    elif page == "Bewegingsintensiteit":
        st.header("Bewegingsintensiteit")
        
        st.subheader("Intensiteitsberekening")
        st.write("""
        De bewegingsintensiteit wordt bepaald door een combinatie van:
        
        **1. Hartslag Zones (Karvonen formule)**
        - Berekend met de Karvonen formule:
            - Doelhartslag = Rusthartslag + (HRR × percentage)
            - HRR = max_HR - rust_HR
            - met:
                - HRR = Heart Rate Reserve (het verschil tussen de maximale hartslag en de rusthartslag)
                - max_HR = de maximale hartslag (220 - leeftijd)
                - rust_HR = de hartslag in rust (5e percentiel van alle hartslagwaarden)
        - Gebruikt persoonlijke leeftijd uit het profiel
        - Zones:
          * Rust: < 45% van HRR
          * Matig: 45-70% van HRR
          * Intensief: > 70% van HRR
        
        **2. Bewegingsdata**
        - Gebaseerd op versnellingsmetingen
        - Drempelwaarden:
          * Matig: > 60e percentiel van alle bewegingsdata
          * Intensief: > 80e percentiel van alle bewegingsdata
        
        **Eindclassificatie**
        De uiteindelijke intensiteit wordt bepaald door de combinatie van hartslag en beweging:
        - Intensief (telt 2x): Beide metingen intensief
        - Matig (telt 1x): 
          * Één meting intensief en andere matig/rust, OF
          * Beide metingen matig
        - Rust: Alle andere combinaties
        """)

        st.subheader("WHO Richtlijnen")
        st.write("""
        Het dashboard volgt de WHO-richtlijnen voor beweging:
        
        **Dagelijks doel: 30 intensiteitsminuten**
        - 30 minuten matige intensiteit OF
        - 15 minuten hoge intensiteit
        - Hoge intensiteit telt dubbel
        
        **Voorwaarden**:
        - Minimaal 10 minuten aaneengesloten activiteit
        - Combinatie van hartslag en beweging
        """)

    elif page == "Stappenteller":
        st.header("Stappenteller")
        
        st.subheader("Stapdetectie")
        st.write("""
        De stappenteller werkt via een geavanceerd algoritme:
        
        **1. Signaalverwerking**
        - Combinatie van X, Y, Z versnellingsdata
        - Berekening van totale versnellingsmagnitude
        - Ruisfiltering en normalisatie
        
        **2. Stapdetectie**
        - Piekdetectie in versnellingssignaal
        - Adaptieve drempelwaarde
        - Minimale tijd tussen stappen
        
        **3. Validatie**
        - Activiteitscontext
        - Bewegingspatroonherkenning
        - Filtering van valse detecties
        """)

        st.subheader("Activiteitsniveaus")
        st.write("""
        **Dagelijks stappendoel: 10.000 stappen**
        
        Classificatie van activiteitsniveaus:
        - Nauwelijks actief: < 5.000 stappen
        - Licht actief: 5.000-7.499 stappen
        - Redelijk actief: 7.500-9.999 stappen
        - Actief: 10.000-12.499 stappen
        - Zeer actief: ≥ 12.500 stappen
        """)

    # Disclaimer onderaan elke pagina
    st.divider()
    st.caption("""
    **Disclaimer**: Dit dashboard is bedoeld als hulpmiddel ter ondersteuning van uw leefstijl.
    Raadpleeg altijd een arts bij gezondheidsklachten of medische vragen.
    """)

# Voeg de infopagina toe aan de navigatie
if __name__ == "__main__":
    info_page()

