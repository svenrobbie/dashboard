# -*- coding: utf-8 -*-
"""
gevorderde versie van het hartgezondheid dashboard.
Toont meerdere en soms complexe visualisaties en analyses.

@author: merli and sven
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Check if data is initialized
if 'data_initialized' not in st.session_state:
    st.error("Begin alstublieft op de homepagina om de data te initialiseren.")
    st.stop()

# Get data from session state
df_ecg = st.session_state.df_ecg
df_ecg_pieken = st.session_state.df_ecg_pieken



#%% Functies
def detecteer_Rpieken(df, pct_drempel):
    max_ecg = max(df['ECG'])
    drempelwaarde = pct_drempel * max_ecg
    pieken = find_peaks(df['ECG'], height=drempelwaarde)[0]
    df_pieken = df.iloc[pieken]
    return df_pieken

def bereken_rmssd(df):
    rmssd = np.sqrt(np.mean(np.square(np.diff(df['rr'])))) * 100
    return rmssd

def detecteer_rustsegmenten(df_acc, df_bpm, drempel=0.05, min_duur=120, resolutie=60):
    """
    Detecteer rustsegmenten als inverse van activiteit - alles wat niet als activiteit wordt geclassificeerd is rust.
    
    Parameters:
    - df_acc: DataFrame met accelerometerdata en timestamp
    - df_bpm: DataFrame met hartslag data
    - drempel: drempelwaarde voor gemiddelde beweging (niet gebruikt in nieuwe implementatie)
    - min_duur: minimale rustduur in seconden
    - resolutie: tijdsresolutie in seconden voor beoordeling

    Returns:
    - rust_intervallen: lijst met (start, eind) tuples in seconden
    """
    # Gebruik berekenen_intensiteit om activiteit te classificeren
    if 'samengesteld' not in df_acc.columns:
        df_acc['samengesteld'] = np.sqrt(df_acc['accX']**2 + df_acc['accY']**2 + df_acc['accZ']**2)
    
    # Bereken intensiteit maar sla beide versies op (voor en na WHO filter)
    df_intensiteit, _ = berekenen_intensiteit(df_acc, df_bpm, resolutie)
    
    # Gebruik de originele intensiteit (voor WHO filter) voor rustdetectie
    df_intensiteit['is_rust'] = df_intensiteit['original_intensity'] == 0
    
    # Detecteer opeenvolgende rustblokken
    df_intensiteit['groep'] = (df_intensiteit['is_rust'] != df_intensiteit['is_rust'].shift()).cumsum()
    groepen = df_intensiteit.groupby('groep')
    
    rust_intervallen = []
    for _, grp in groepen:
        if grp['is_rust'].all():
            duur = len(grp) * resolutie
            if duur >= min_duur:  # Alleen periodes langer dan min_duur
                start = grp['blok'].min() * resolutie
                eind = grp['blok'].max() * resolutie
                rust_intervallen.append((start, eind))
    
    return rust_intervallen

def bereken_rmssd_rust(df_pieken, rust_intervallen):
    """
    Berekent RMSSD per minuut binnen opgegeven rustintervallen.
    
    Returns:
    - DataFrame met timestamp en RMSSD waarden tijdens rust
    """
    rmssd_data = []
    
    for start, eind in rust_intervallen:
        # Filter data voor dit rustinterval
        blok = df_pieken[(df_pieken['timestamp'] >= start) & (df_pieken['timestamp'] <= eind)]
        
        # Bereken RMSSD per minuut binnen dit interval
        blok['minuut'] = (blok['timestamp'] // 60) * 60
        for _, minuut_data in blok.groupby('minuut'):
            if len(minuut_data) > 2:  # Minimaal 3 hartslagen nodig voor RMSSD
                rmssd = bereken_rmssd(minuut_data)
                rmssd_data.append({
                    'timestamp': minuut_data['timestamp'].mean(),
                    'rmssd': rmssd
                })
    
    if not rmssd_data:
        return pd.DataFrame(columns=['timestamp', 'rmssd'])
    
    return pd.DataFrame(rmssd_data)

def get_age_reference_rmssd(age):
    """
    Returns the reference RMSSD value based on age group.
    """
    if age < 30:
        return 53
    elif age < 40:
        return 42
    elif age < 50:
        return 34
    elif age < 60:
        return 31
    else:
        return 39

def plot_rmssd_rust(rmssd_df, toon_tijdverloop=False):
    if rmssd_df.empty:
        return go.Figure().update_layout(title="Geen geldige rustsegmenten gevonden.")
    
    # Calculate common statistics
    gemiddelde = rmssd_df['rmssd'].mean()
    age_reference = get_age_reference_rmssd(st.session_state.user_age)
    
    # Prepare interpretation text (moved outside of if/else)
    interpretatie = ""
    if gemiddelde < 25:
        interpretatie = "Zeer lage HRV - Mogelijk teken van stress of vermoeidheid."
    elif gemiddelde < 50:
        interpretatie = "Lage HRV - Ruimte voor verbetering in herstel."
    elif gemiddelde < 75:
        interpretatie = "Optimale HRV - Goede balans in het zenuwstelsel."
    else:
        interpretatie = "Hoge HRV - Uitstekende hartflexibiliteit."
        
    # Add comparison with age reference
    verschil = gemiddelde - age_reference
    if abs(verschil) > 5:  # Only add comparison if difference is significant
        if verschil > 0:
            interpretatie += f"<br>Uw HRV is {verschil:.1f} ms hoger dan gemiddeld voor uw leeftijd"
        else:
            interpretatie += f"<br>Uw HRV is {-verschil:.1f} ms lager dan gemiddeld voor uw leeftijd"
    
    if toon_tijdverloop:
        # Create figure with time series
        fig = go.Figure()
        
        # Add RMSSD time series
        fig.add_trace(
            go.Scatter(
                x=rmssd_df['timestamp'] / 60,  # Convert to minutes
                y=rmssd_df['rmssd'],
                mode='lines+markers',
                name='RMSSD',
                line=dict(color='green', width=2),
                marker=dict(size=6),
            )
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[rmssd_df['timestamp'].min() / 60, rmssd_df['timestamp'].max() / 60],
                y=[age_reference, age_reference],
                mode='lines',
                name='Leeftijdsreferentie',
                line=dict(color='gray', dash='dash'),
            )
        )
        
        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=[rmssd_df['timestamp'].min() / 60, rmssd_df['timestamp'].max() / 60],
                y=[gemiddelde, gemiddelde],
                mode='lines',
                name='Uw gemiddelde',
                line=dict(color='blue', dash='dot'),
            )
        )
        
        # Update layout with adjusted margins and positions
        fig.update_layout(
            title={
                'text': 'RMSSD tijdens rust',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Tijd (minuten)',
            yaxis_title='RMSSD (ms)',
            template='plotly_white',
            showlegend=True,
            height=500,  # Slightly reduced height
            margin=dict(t=100, b=110),  # Reduced bottom margin
            annotations=[
                dict(
                    text=interpretatie,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.25,  # Moved up
                    showarrow=False,
                    font=dict(size=12),
                    align="center"
                )
            ]
        )
        
    else:
        # Create gauge chart for average RMSSD
        fig = go.Figure()
        
        # Add the main gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=gemiddelde,
            number={'font': {'size': 40}, 'valueformat': '.1f'},
            title={'text': "Gemiddelde RMSSD tijdens rust", 
                   'font': {'size': 16}},
            delta={'reference': age_reference, 
                   'increasing': {'color': "green"}, 
                   'decreasing': {'color': "red"},
                   'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100], 
                        'tickwidth': 1,
                        'tickcolor': "darkgray",
                        'ticktext': ["Zeer Laag", "Laag", "Optimaal", "Hoog"],
                        'tickvals': [12.5, 37.5, 62.5, 87.5]},
                'bar': {'color': "darkgreen", 'thickness': 0.15},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#ff9999'},      # Rood voor zeer lage HRV
                    {'range': [25, 50], 'color': '#ffcc99'},     # Oranje voor lage HRV
                    {'range': [50, 75], 'color': '#99ff99'},     # Groen voor optimale HRV
                    {'range': [75, 100], 'color': '#ffff99'}     # Geel voor hoge HRV
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': age_reference
                }
            }
        ))
        
        # Update layout for better presentation
        fig.update_layout(
            height=400,
            margin=dict(t=120, b=100),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Add interpretation text
        fig.add_annotation(
            text=interpretatie.replace("<br>", "\n"),  # Convert <br> to \n for gauge view
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
            xref='paper',
            yref='paper',
            align='center'
        )
    
    return fig

def visualize_rmssd_per_block(df):
    df['time_block'] = (df['timestamp'] // 60) * 60
    rmssd_per_block = df.groupby('time_block').apply(bereken_rmssd).reset_index(name='rmssd')
    rmssd_per_block['time_block_min'] = rmssd_per_block['time_block'] / 60
    fig = px.line(rmssd_per_block, x='time_block_min', y='rmssd', markers=True)
    fig.update_layout(xaxis_title='Tijd (minuten)', yaxis_title='RMSSD (ms)')
    return fig

def visualize_stressscore(df):
    """
    Berekent en visualiseert een stressscore op basis van RMSSD.
    Lage RMSSD = hoge stress, hoge RMSSD = lage stress.
    """
    min_rmssd = df['rmssd'].min()
    max_rmssd = df['rmssd'].max()
    stressscore = 100 * (1 - (df['rmssd'] - min_rmssd) / (max_rmssd - min_rmssd))
    gemiddelde_stress = stressscore.mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add colored background rectangles for zones with more vibrant colors
    zones = [
        (0, 30, 'rgba(100, 255, 100, 0.6)', 'Ontspannen'),    # Brighter green
        (30, 60, 'rgba(255, 255, 100, 0.6)', 'Licht'),        # Brighter yellow
        (60, 80, 'rgba(255, 180, 100, 0.6)', 'Matig'),        # Brighter orange
        (80, 100, 'rgba(255, 100, 100, 0.6)', 'Hoog')         # Brighter red
    ]
    
    for start, end, color, label in zones:
        # Add background rectangle
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )
        
        # Add zone label
        fig.add_annotation(
            x=(start + end) / 2,
            y=1.2,
            text=label,
            showarrow=False,
            font=dict(size=12)
        )
    
    # Add progress bar with deep blue color
    progress_color = 'rgba(0, 102, 204, 0.9)'  # Deep blue
    border_color = 'rgba(0, 102, 204, 1.0)'    # Solid deep blue for border
    
    fig.add_trace(go.Bar(
        x=[gemiddelde_stress],
        y=[0.5],
        orientation='h',
        width=0.6,
        marker=dict(
            color=progress_color,
            line=dict(color=border_color, width=2)
        ),
        showlegend=False
    ))
    
    # Add vertical line at current value with matching color
    fig.add_shape(
        type="line",
        x0=gemiddelde_stress,
        x1=gemiddelde_stress,
        y0=0,
        y1=1,
        line=dict(color=border_color, width=3)
    )
    
    # Add stress level text
    stress_text = ""
    if gemiddelde_stress < 30:
        stress_text = "Laag stressniveau - U bent ontspannen"
    elif gemiddelde_stress < 60:
        stress_text = "Gemiddeld stressniveau - Normale dagelijkse spanning"
    elif gemiddelde_stress < 80:
        stress_text = "Verhoogd stressniveau - Let op uw herstel"
    else:
        stress_text = "Hoog stressniveau - Neem tijd voor ontspanning"
    
    # Add percentage text
    fig.add_annotation(
        x=gemiddelde_stress,
        y=-0.2,
        text=f"{gemiddelde_stress:.1f}%",
        showarrow=False,
        font=dict(size=24, color=border_color)  # Match the progress bar color
    )
    
    # Add interpretation text
    fig.add_annotation(
        x=50,
        y=-0.6,
        text=stress_text,
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(t=80, b=100, l=40, r=40),
        xaxis=dict(
            range=[-5, 105],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-1, 1.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        title=dict(
            text="Stressniveau<br><sup>Gebaseerd op HRV analyse</sup>",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    return fig

def visualize_gauge(rmssd, gemiddelde_rmssd):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rmssd,
        delta={'reference': gemiddelde_rmssd, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75], 'ticktext': ["0", "25", "50", "75"]},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "yellow"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gemiddelde_rmssd
            }
        }
    ))
    return fig

def bereken_sdnn(df):
    """
    Berekent SDNN (standaarddeviatie van RR-intervallen in ms).
    Input: DataFrame met kolom 'rr' in seconden.
    Output: SDNN in milliseconden.
    """
    sdnn = np.std(df['rr'], ddof=1) * 1000  # Seconden naar ms
    return sdnn

def get_age_reference_sdnn(age):
    """
    Returns the reference SDNN value based on age group.
    """
    if age < 30:
        return 164
    elif age < 40:
        return 151
    elif age < 50:
        return 141
    elif age < 60:
        return 143
    else:
        return 128

def visualize_sdnn_per_block(df, toon_tijdverloop=False):
    """
    Visualizes SDNN either as a gauge (default) or as a time series plot.
    """
    # Calculate SDNN per minute first
    df['time_block'] = (df['timestamp'] // 60) * 60
    sdnn_per_block = df.groupby('time_block').apply(bereken_sdnn).reset_index(name='sdnn')
    
    # Calculate statistics from the per-minute values
    gemiddelde_sdnn = sdnn_per_block['sdnn'].mean()  # Average of per-minute SDNNs
    age_reference = get_age_reference_sdnn(st.session_state.user_age)
    
    # Prepare interpretation text
    interpretatie = ""
    if gemiddelde_sdnn < 30:
        interpretatie = "Zeer lage HRV - Mogelijk teken van chronische stress of verminderde hartgezondheid."
    elif gemiddelde_sdnn < 50:
        interpretatie = "Lage HRV - Ruimte voor verbetering in algemene hartflexibiliteit."
    elif gemiddelde_sdnn < 100:
        interpretatie = "Normale HRV - Gezonde hartflexibiliteit."
    else:
        interpretatie = "Hoge HRV - Uitstekende hartflexibiliteit en aanpassingsvermogen."
        
    # Add comparison with age reference
    verschil = gemiddelde_sdnn - age_reference
    if abs(verschil) > 5:  # Only add comparison if difference is significant
        if verschil > 0:
            interpretatie += f"<br>Uw HRV is {verschil:.1f} ms hoger dan gemiddeld voor uw leeftijd"
        else:
            interpretatie += f"<br>Uw HRV is {-verschil:.1f} ms lager dan gemiddeld voor uw leeftijd"

    if toon_tijdverloop:
        # Create time series plot
        fig = go.Figure()
        
        # Add SDNN time series
        fig.add_trace(
            go.Scatter(
                x=sdnn_per_block['time_block'] / 60,  # Convert to minutes
                y=sdnn_per_block['sdnn'],
                mode='lines+markers',
                name='SDNN',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
            )
        )
        
        # Add reference lines
        fig.add_trace(
            go.Scatter(
                x=[sdnn_per_block['time_block'].min() / 60, sdnn_per_block['time_block'].max() / 60],
                y=[gemiddelde_sdnn, gemiddelde_sdnn],
                mode='lines',
                name='Uw gemiddelde',
                line=dict(color='red', dash='dot'),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[sdnn_per_block['time_block'].min() / 60, sdnn_per_block['time_block'].max() / 60],
                y=[age_reference, age_reference],
                mode='lines',
                name='Leeftijdsreferentie',
                line=dict(color='gold', dash='dash'),
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'SDNN over tijd',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Tijd (minuten)',
            yaxis_title='SDNN (ms)',
            template='plotly_white',
            showlegend=True,
            height=500,
            margin=dict(t=100, b=110),
            annotations=[
                dict(
                    text=interpretatie,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.25,
                    showarrow=False,
                    font=dict(size=12),
                    align="center"
                )
            ]
        )
        
    else:
        # Create gauge chart
        fig = go.Figure()
        
        # Add the main gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=gemiddelde_sdnn,
            number={'font': {'size': 40}, 'valueformat': '.1f'},
            title={'text': "Gemiddelde SDNN", 
                   'font': {'size': 16}},
            delta={'reference': age_reference, 
                   'increasing': {'color': "green"}, 
                   'decreasing': {'color': "red"},
                   'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 200], 
                        'tickwidth': 1,
                        'tickcolor': "darkgray",
                        'ticktext': ["Zeer Laag", "Laag", "Normaal", "Hoog"],
                        'tickvals': [15, 45, 75, 105]},
                'bar': {'color': "darkblue", 'thickness': 0.15},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ff9999'},      # Rood voor zeer lage HRV
                    {'range': [30, 50], 'color': '#ffcc99'},     # Oranje voor lage HRV
                    {'range': [50, 100], 'color': '#99ff99'},    # Groen voor normale HRV
                    {'range': [100, 200], 'color': '#99ccff'}    # Blauw voor hoge HRV
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': age_reference  # Changed from gemiddelde_sdnn to age_reference
                }
            }
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(t=120, b=100),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Add interpretation text
        fig.add_annotation(
            text=interpretatie,
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
            xref='paper',
            yref='paper',
            align='center'
        )
    
    return fig

def bereken_stappenteller(df, drempel=0.5):
    """
    Detecteert stappen op basis van versnellingsdata.
    
    Parameters:
    - df: DataFrame met versnellingsdata
    - drempel: Gevoeligheid voor stapdetectie (0-1)
    
    Returns:
    - df: DataFrame met stappenteller
    - stappen_index: Indices waar stappen gedetecteerd zijn
    """
    df = df.copy()
    df['samengesteld_signaal'] = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)
    df['genormaliseerd'] = df['samengesteld_signaal'] - df['samengesteld_signaal'].mean()

    # Detecteer pieken in het versnellingssignaal
    pieken = find_peaks(df['genormaliseerd'],
                      height=df['genormaliseerd'].max() * drempel,
                      distance=10)[0]
    
    # Markeer stappen en tel ze
    df['is_een_stap'] = df.index.isin(pieken)
    df['stappenteller'] = df['is_een_stap'].cumsum()

    return df, pieken

def plot_cumulatieve_stappen(df):
    """
    Plot het cumulatieve aantal stappen over tijd.
    
    Parameters:
    - df: DataFrame met stappenteller kolom
    - titel: Titel voor de grafiek
    
    Returns:
    - fig: Plotly figuur
    """
    fig = px.line(df, x=df['timestamp'] / 60, y='stappenteller',
                  labels={'x': 'Tijd (minuten)', 'stappenteller': 'Aantal stappen'},
                  template='plotly_white',
                  title="Cumulatief aantal stappen over tijd")
    return fig

def berekenen_intensiteit(df_acc, df_bpm, resolutie=60):
    """
    Berekent de bewegingsintensiteit per tijdsblok.
    """
    # Zorg ervoor dat we de volledige tijdsrange hebben
    start_tijd = min(df_acc['timestamp'].min(), df_bpm['timestamp'].min())
    eind_tijd = max(df_acc['timestamp'].max(), df_bpm['timestamp'].max())
    
    # Maak een DataFrame met alle mogelijke tijdsblokken
    alle_blokken = pd.DataFrame({
        'blok': range(int(start_tijd // resolutie), int(eind_tijd // resolutie) + 1)
    })
    
    # Verdeel data in tijdsblokken
    df_acc['blok'] = (df_acc['timestamp'] // resolutie).astype(int)
    df_bpm['blok'] = (df_bpm['timestamp'] // resolutie).astype(int)
    
    # Bereken gemiddelde versnelling en hartslag per blok
    acc_blok = df_acc.groupby('blok')['samengesteld'].mean().reset_index(name='gem_acc')
    bpm_blok = df_bpm.groupby('blok')['bpm'].mean().reset_index(name='gem_bpm')
    
    # Voeg alle tijdsblokken samen met de berekende waarden
    df_blok = alle_blokken.merge(acc_blok, on='blok', how='left')
    df_blok = df_blok.merge(bpm_blok, on='blok', how='left')
    
    # Vul ontbrekende waarden in met vorige/volgende waarden
    df_blok = df_blok.fillna(method='ffill').fillna(method='bfill')
    
    # Bepaal rusthartslag (5e percentiel van alle hartslagwaarden)
    resting_hr = df_bpm['bpm'].quantile(0.05)
    
    # Bereken hartslagzones met Karvonen formule
    max_hr = 220 - st.session_state.user_age  # Maximale hartslag op basis van leeftijd
    hrr = max_hr - resting_hr  # Heart Rate Reserve
    moderate_hr = resting_hr + (hrr * 0.45)  # 45% van HRR voor matige intensiteit
    vigorous_hr = resting_hr + (hrr * 0.70)  # 70% van HRR voor hoge intensiteit
    
    # Bepaal versnellingsdrempels op basis van percentielwaarden
    acc_moderate = np.percentile(df_acc['samengesteld'], 60)  # 60e percentiel voor matig
    acc_vigorous = np.percentile(df_acc['samengesteld'], 80)  # 80e percentiel voor intensief
    
    # Begin met alle minuten als rust
    df_blok['intensiteit'] = 0
    
    # Classificeer eerst op basis van hartslag
    df_blok['hr_intensity'] = np.where(
        df_blok['gem_bpm'] > vigorous_hr, 2,  # Boven vigorous_hr = intensief
        np.where(df_blok['gem_bpm'] > moderate_hr, 1, 0)  # Boven moderate_hr = matig
    )
    
    # Classificeer daarna op basis van versnelling
    df_blok['acc_intensity'] = np.where(
        df_blok['gem_acc'] > acc_vigorous, 2,  # Boven acc_vigorous = intensief
        np.where(df_blok['gem_acc'] > acc_moderate, 1, 0)  # Boven acc_moderate = matig
    )
    
    # Combineer hartslag en versnelling voor eindclassificatie:
    # Intensief: Beide metingen hoog
    # Matig: Of één hoog, of beide matig
    df_blok['original_intensity'] = np.where(
        ((df_blok['hr_intensity'] == 2) & (df_blok['acc_intensity'] == 2)),  # Beide hoog
        2,  # Intensief
        np.where(
            ((df_blok['hr_intensity'] == 2) & (df_blok['acc_intensity'] >= 1)) |  # Hoge hartslag met matige beweging
            ((df_blok['hr_intensity'] >= 1) & (df_blok['acc_intensity'] == 2)) |  # Hoge beweging met matige hartslag
            ((df_blok['hr_intensity'] == 1) & (df_blok['acc_intensity'] == 1)),   # Beide matig
            1,  # Matig
            0   # Rust
        )
    )
    
    # Kopieer originele intensiteit
    df_blok['intensiteit'] = df_blok['original_intensity'].copy()
    
    # Zoek aaneengesloten blokken activiteit
    df_blok['consecutive_group'] = (df_blok['intensiteit'] != df_blok['intensiteit'].shift()).cumsum()
    df_blok['group_size'] = df_blok.groupby('consecutive_group')['intensiteit'].transform('size')
    
    # Tel alleen blokken van 10+ minuten mee (WHO richtlijn)
    min_consecutive_blocks = 10
    df_blok.loc[df_blok['group_size'] < min_consecutive_blocks, 'intensiteit'] = 0
    
    # Bereken totaal aantal intensiteitsminuten
    moderate_minutes = (df_blok['intensiteit'] == 1).sum()
    vigorous_minutes = (df_blok['intensiteit'] == 2).sum() * 2  # Intensief telt dubbel
    total_intensity_minutes = moderate_minutes + vigorous_minutes
    
    return df_blok, total_intensity_minutes

def plot_bewegingsintensiteit(df_blok):
    """
    Visualiseert de bewegingsintensiteit over tijd in een staafdiagram.
    
    Parameters:
    - df_blok: DataFrame met intensiteitsclassificatie per minuut
    
    Returns:
    - fig: Plotly figuur met het staafdiagram
    """
    # Zet bloknummer om naar minuten voor x-as
    df_blok['tijd_min'] = df_blok['blok']
    
    # Definieer kleurenschema voor verschillende intensiteitsniveaus
    kleurmap = {
        0: '#f0f0f0',  # Lichtgrijs voor rust
        1: '#4a9eff',  # Lichtblauw voor matige intensiteit
        2: '#0066cc'   # Donkerblauw voor hoge intensiteit
    }
    
    # Maak staafdiagram
    fig = px.bar(
        df_blok,
        x='tijd_min',
        y='intensiteit',
        color='intensiteit',
        color_discrete_map=kleurmap,
        labels={
            'tijd_min': 'Tijd (minuten)', 
            'intensiteit': 'Intensiteitsniveau'
        },
    )
    
    # Pas layout aan voor betere leesbaarheid
    fig.update_layout(
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["Rust", "Matig (1x)", "Intensief (2x)"],
            range=[-0.5, 2.5],  # Voeg ruimte toe boven en onder
        ),
        title={
            'text': 'Bewegingsintensiteit per minuut<br><sup>Intensieve minuten tellen dubbel voor je dagelijkse doel</sup>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            title='Tijd (minuten)',
            range=[df_blok['tijd_min'].min() - 1, df_blok['tijd_min'].max() + 1]
        ),
        yaxis_showgrid=False,
        bargap=0
    )
    
    # Voeg tooltip toe voor interactiviteit
    fig.update_traces(
        showlegend=True,
        name='Intensiteitsniveau',
        hovertemplate='Tijd: %{x} min<br>Intensiteit: %{y}<extra></extra>'
    )
    
    return fig

#%% Streamlit opmaak
st.set_page_config(page_title="Dashboard hart gezondheid", layout="wide", initial_sidebar_state="collapsed")
st.title('Dashboard hart gezondheid')

# Sidebar
st.sidebar.header("Navigatie: selectie van grafieken")
with st.sidebar:
    st.sidebar.header("Standaard weergave")
    show_hrv = st.checkbox("HRV", value=True)
    show_sdnn = st.checkbox("SDNN", value=True)
    show_stappenteller = st.checkbox("Stappenteller", value=True)
    show_bewegingsminuten = st.checkbox("Bewegings minuten", value=True)
    st.divider()
    st.sidebar.header("Uitgebreide weergave")
    
    # Group HRV visualization options
    st.sidebar.subheader("HRV Visualisatie Opties")
    show_realtime_hrv = st.checkbox("Live monitoring van hartgegevens", value=False)
    col_hrv1, col_hrv2 = st.sidebar.columns(2)
    with col_hrv1:
        toon_rmssd_verloop = st.checkbox("RMSSD verloop over tijd", value=False, key="rmssd_verloop")
    with col_hrv2:
        toon_sdnn_verloop = st.checkbox("SDNN verloop over tijd", value=False, key="sdnn_verloop")
    
    st.sidebar.subheader("Activiteit")
    
    show_bewegingsintensiteit = st.checkbox("Bewegingsintensiteit", value=True)
    show_stappen_cumulatief = st.checkbox("Cumulatieve stappen", value=True)
    show_rmssd_stressscore = st.checkbox("RMSSD-Stressscore", value=True)
    
doel1, doel2, doel3, doel4 = st.columns(4)
# DOEL 1: RMSSD VERHOGEN TIJDENS RUST
with doel1:
    st.header("🎯 Doel 1: Verhoog RMSSD tijdens rust")
    rust_intervallen = detecteer_rustsegmenten(df_ecg, df_ecg_pieken)
    rmssd_df = bereken_rmssd_rust(df_ecg_pieken, rust_intervallen)
    huidige_rmssd = rmssd_df['rmssd'].mean() if not rmssd_df.empty else 0
    doelwaarde_rmssd = st.session_state.doelwaarde_rmssd

    st.metric("Gemiddelde RMSSD", f"{huidige_rmssd:.1f} ms", delta=f"{huidige_rmssd - doelwaarde_rmssd:.1f} ms")
    progressie_rmssd = min(huidige_rmssd / doelwaarde_rmssd, 1.0)
    st.progress(progressie_rmssd)
    if progressie_rmssd >= 1.0:
        st.success("✅ Doel behaald!")
        st.badge("HRV-topper", color="green")
    elif progressie_rmssd >= 0.75:
        st.info("Bijna daar! Je bent goed op weg naar je doel.")
    elif progressie_rmssd < 0.75:
        st.warning("Nog niet op het gewenste niveau? Rust en slaap dragen bij aan verbetering.")


# DOEL 2: BEWEEGMINUTEN PER WEEK
with doel2:
    st.header("🎯 Doel 2: Beweeg " + str(st.session_state.doelwaarde_beweging) + " minuten/week")
    df_blok, total_intensity_minutes = berekenen_intensiteit(df_ecg, df_ecg_pieken)

    st.metric("Beweegminuten (intensief x2)", f"{total_intensity_minutes} min", delta=f"{total_intensity_minutes - st.session_state.doelwaarde_beweging} min")
    progressie_beweging = min(total_intensity_minutes / st.session_state.doelwaarde_beweging, 1.0)
    st.progress(progressie_beweging)
    if progressie_beweging >= 1.0:
        st.success("✅ Doel behaald!")
        st.badge("Actief deze week", color="green")
    elif progressie_beweging >= 0.75:
        st.info("Bijna daar! Blijf zo doorgaan.")
    elif progressie_beweging < 0.75:
        st.warning("Je bent er nog niet, maar het is nooit te laat om te starten")


# DOEL 3: GEMIDDELDE STRESSSCORE VERLAGEN
with doel3:
    st.header("🎯 Doel 3: Stress onder " + str(st.session_state.doelwaarde_stress) + "%")
    rmssd_df = df_ecg_pieken.copy()
    rmssd_df['time_block'] = (rmssd_df['timestamp'] // 10) * 10
    rmssd_blocks = rmssd_df.groupby('time_block').apply(bereken_rmssd).reset_index(name='rmssd')
    if not rmssd_blocks.empty:
        min_rmssd = rmssd_blocks['rmssd'].min()
        max_rmssd = rmssd_blocks['rmssd'].max()
        stressscore = 100 * (1 - (rmssd_blocks['rmssd'] - min_rmssd) / (max_rmssd - min_rmssd))
        gemiddelde_stress = stressscore.mean()
    else:
        gemiddelde_stress = 100
    
    verschil_stress = st.session_state.doelwaarde_stress - gemiddelde_stress
    st.metric("Gemiddelde stressscore", f"{gemiddelde_stress:.1f}%", delta=f"{verschil_stress:.1f}%")
    if gemiddelde_stress <= st.session_state.doelwaarde_stress:
        progressie_stress = 1.0
    else:
        overschrijding = gemiddelde_stress - st.session_state.doelwaarde_stress
        progressie_stress = max(0.0, 1.0 - (overschrijding / (100 - st.session_state.doelwaarde_stress)))
    st.progress(progressie_stress)
    if progressie_stress >= 1.0:
        st.success("✅ Doel behaald!")
        st.badge("Kalm en in balans", color="blue")
    elif progressie_stress >= 0.75:
        st.info("Top bezig! Je zit dichtbij je doel, hou dit vol!")
    elif progressie_stress < 0.75:
        st.warning("Je zit boven je gewenste stressniveau. Tijd voor jezelf kan goed doen.")


# DOEL 4: DAGELIJKS STAPPENDOEL
with doel4:
    st.header("🎯 Doel 4: " + str(st.session_state.Dagelijks_stappendoel) + " stappen/dag")
    df_bewerkt, stappen_index = bereken_stappenteller(df_ecg_pieken, drempel=0.1)
    totaal_stappen = df_bewerkt['stappenteller'].iloc[-1]

    st.metric("Aantal stappen", f"{int(totaal_stappen):,}", delta=f"{int(totaal_stappen - st.session_state.Dagelijks_stappendoel):,}")
    progressie_stappen = min(1.0, totaal_stappen / st.session_state.Dagelijks_stappendoel)
    st.progress(progressie_stappen)
    if progressie_stappen >= 1.0:
        st.success("✅ Doel behaald!")
        st.badge("Stappenkampioen", color="green")
    elif progressie_stappen >= 0.75:
        st.info("Je bent er bijna — zet die laatste stap")
    elif progressie_stappen < 0.75:
        st.warning("Je zit nog onder je stappendoel — een korte wandeling kan al verschil maken")

st.divider()      
# plots
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# Add real-time HRV meters section if enabled
if show_realtime_hrv:
    st.subheader("Live monitoring van hartgegevens")
    st.write("Deze meters tonen de meest recente HRV-waarden van verschillende metrieken.")
    
    # Create two columns for the gauges
    gauge_col1, gauge_col2 = st.columns(2)
    
    with gauge_col1:
        # Real-time RMSSD
        df_recent = df_ecg_pieken.copy()
        df_recent['time_block'] = (df_recent['timestamp'] // 10) * 10
        rmssd_blocks = df_recent.groupby('time_block').apply(bereken_rmssd).reset_index(name='rmssd')
        if not rmssd_blocks.empty:
            laatste_rmssd = rmssd_blocks['rmssd'].iloc[-1]
            gemiddelde_rmssd = rmssd_blocks['rmssd'].mean()
            
            # Build interpretation for this specific gauge
            interpretatie_gauge_realtime = ""
            if laatste_rmssd < 25:
                interpretatie_gauge_realtime = "Zeer lage HRV - Mogelijk teken van stress of vermoeidheid."
            elif laatste_rmssd < 50:
                interpretatie_gauge_realtime = "Lage HRV - Ruimte voor verbetering in herstel."
            elif laatste_rmssd < 75:
                interpretatie_gauge_realtime = "Optimale HRV - Goede balans in het zenuwstelsel."
            else:
                interpretatie_gauge_realtime = "Hoge HRV - Uitstekende hartflexibiliteit."

            fig_realtime = go.Figure()
            fig_realtime.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=laatste_rmssd,
                number={'font': {'size': 40}, 'valueformat': '.1f'},
                title={'text': "Live monitoring van hartgegevens (RMSSD)", 
                       'font': {'size': 16}},
                delta={'reference': gemiddelde_rmssd, 
                       'increasing': {'color': "green"}, 
                       'decreasing': {'color': "red"},
                       'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100], 
                            'tickwidth': 1,
                            'tickcolor': "darkgray",
                            'ticktext': ["Zeer Laag", "Laag", "Optimaal", "Hoog"],
                            'tickvals': [12.5, 37.5, 62.5, 87.5]},
                    'bar': {'color': "darkblue", 'thickness': 0.15},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': '#ff9999'},
                        {'range': [25, 50], 'color': '#ffcc99'},
                        {'range': [50, 75], 'color': '#99ff99'},
                        {'range': [75, 100], 'color': '#ffff99'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': gemiddelde_rmssd
                    }
                }
            ))
            fig_realtime.update_layout(height=300, margin=dict(t=60, b=80))
            
            # Add interpretation text below the gauge
            fig_realtime.add_annotation(
                text=interpretatie_gauge_realtime,
                x=0.5,
                y=-0.3,
                showarrow=False,
                font=dict(size=12),
                xref="paper",
                yref="paper",
                align="center"
            )
            
            st.plotly_chart(fig_realtime, use_container_width=True)
    
    with gauge_col2:
        # Real-time SDNN
        df_sdnn = df_ecg_pieken.copy()
        df_sdnn['time_block'] = (df_sdnn['timestamp'] // 60) * 60
        sdnn_blocks = df_sdnn.groupby('time_block').apply(bereken_sdnn).reset_index(name='sdnn')
        if not sdnn_blocks.empty:
            laatste_sdnn = sdnn_blocks['sdnn'].iloc[-1]
            age_reference_sdnn = get_age_reference_sdnn(st.session_state.user_age)
            
            # Build interpretation for this specific gauge
            interpretatie_gauge_sdnn = ""
            if laatste_sdnn < 30:
                interpretatie_gauge_sdnn = "Zeer lage HRV - Mogelijk teken van chronische stress."
            elif laatste_sdnn < 50:
                interpretatie_gauge_sdnn = "Lage HRV - Ruimte voor verbetering in hartflexibiliteit."
            elif laatste_sdnn < 100:
                interpretatie_gauge_sdnn = "Normale HRV - Gezonde hartflexibiliteit."
            else:
                interpretatie_gauge_sdnn = "Hoge HRV - Uitstekende hartflexibiliteit."

            fig_sdnn = go.Figure()
            fig_sdnn.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=laatste_sdnn,
                number={'font': {'size': 40}, 'valueformat': '.1f'},
                title={'text': "Live monitoring van hartgegevens (SDNN)", 
                       'font': {'size': 16}},
                delta={'reference': age_reference_sdnn, 
                       'increasing': {'color': "green"}, 
                       'decreasing': {'color': "red"},
                       'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 200], 
                            'tickwidth': 1,
                            'tickcolor': "darkgray",
                            'ticktext': ["Zeer Laag", "Laag", "Normaal", "Hoog"],
                            'tickvals': [15, 45, 75, 105]},
                    'bar': {'color': "purple", 'thickness': 0.15},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#ff9999'},
                        {'range': [30, 50], 'color': '#ffcc99'},
                        {'range': [50, 100], 'color': '#99ff99'},
                        {'range': [100, 200], 'color': '#99ccff'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': age_reference_sdnn
                    }
                }
            ))
            fig_sdnn.update_layout(height=300, margin=dict(t=60, b=80))
            
            # Add interpretation text below the gauge
            fig_sdnn.add_annotation(
                text=interpretatie_gauge_sdnn,
                x=0.5,
                y=-0.3,
                showarrow=False,
                font=dict(size=12),
                xref="paper",
                yref="paper",
                align="center"
            )
            
            st.plotly_chart(fig_sdnn, use_container_width=True)
    
    with st.expander("ℹ️ Uitleg over de real-time meters"):
        st.write("""
        Deze sectie toont real-time HRV meters:
        
        1. **Real-time RMSSD** (links):
           - Toont de meest recente RMSSD-waarde, ongeacht activiteit
           - Vergelijkt met je persoonlijke gemiddelde
           - Nuttig om directe veranderingen in HRV te zien
        
        2. **Real-time SDNN** (rechts):
           - Toont de laatste SDNN-waarde
           - Vergelijkt met de leeftijdsreferentie
           - Geeft een breder beeld van je hartritme variabiliteit
        
        De zwarte lijn op elke meter geeft de referentiewaarde aan waarmee wordt vergeleken.
        """)

with col1:
    if show_hrv:
        st.subheader("HRV (RMSSD)")

        # Detecteer rust op basis van acceleratie en hartslag
        rust_intervallen = detecteer_rustsegmenten(df_ecg, df_ecg_pieken)
        rmssd_df = bereken_rmssd_rust(df_ecg_pieken, rust_intervallen)

        if show_realtime_hrv:
            # Real-time RMSSD
            df_recent = df_ecg_pieken.copy()
            df_recent['time_block'] = (df_recent['timestamp'] // 10) * 10
            rmssd_blocks = df_recent.groupby('time_block').apply(bereken_rmssd).reset_index(name='rmssd')
            laatste_rmssd = rmssd_blocks['rmssd'].iloc[-1]
            gemiddelde_rmssd = rmssd_blocks['rmssd'].mean()
            
            fig_realtime = go.Figure()
            fig_realtime.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=laatste_rmssd,
                number={'font': {'size': 40}, 'valueformat': '.1f'},
                title={'text': f"Live monitoring van RMSSD<br><span style='font-size:0.8em;color:gray'>Gemiddelde: {gemiddelde_rmssd:.1f} ms</span>", 
                       'font': {'size': 16}},
                delta={'reference': gemiddelde_rmssd, 
                       'increasing': {'color': "green"}, 
                       'decreasing': {'color': "red"},
                       'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100], 
                            'tickwidth': 1,
                            'tickcolor': "darkgray",
                            'ticktext': ["Zeer Laag", "Laag", "Optimaal", "Hoog"],
                            'tickvals': [12.5, 37.5, 62.5, 87.5]},
                    'bar': {'color': "darkblue", 'thickness': 0.15},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': '#ff9999'},
                        {'range': [25, 50], 'color': '#ffcc99'},
                        {'range': [50, 75], 'color': '#99ff99'},
                        {'range': [75, 100], 'color': '#ffff99'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': gemiddelde_rmssd
                    }
                }
            ))
            fig_realtime.update_layout(height=400, margin=dict(t=100, b=100))
            st.plotly_chart(fig_realtime, use_container_width=True)
        else:
            fig = plot_rmssd_rust(rmssd_df, toon_tijdverloop=toon_rmssd_verloop)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Uitleg over de grafiek"):
            st.subheader("HRV-analyse: RMSSD tijdens rust")
            st.write("""             
Wat laat het zien:
De RMSSD (Root Mean Square of Successive Differences) is een maat voor de variatie in tijd tussen opeenvolgende hartslagen. 
Hier wordt het alleen getoond tijdens periodes van rust, wanneer zowel de hartslag als beweging laag zijn.

Wat betekent dit:
RMSSD is een indicator voor parasympathische activiteit (ontspanningszenuwstelsel). 
Omdat dit een indicator is voor het parasympathische zenuwstelsel geeft deze alleen accurate resultaten als er in rust gemeten is.

**Hoge RMSSD** = betere stressregulatie en rustiger autonoom zenuwstelsel.

**Lage RMSSD** = verhoogde stress en vermoeidheid of verminderde cardiovasculaire flexibiliteit.

Belangrijk bij het beoordelen van stressniveau, herstel na inspanning, of algemene hartgezondheid.

*bron: Caring Medical Florida. (2024, 12 juli). Heart rate variability. https://caringmedical.com/hauser-neck-center/heart-rate-variability/*
                     """)

with col2:
    if show_sdnn:
        st.subheader("HRV (SDNN)")
        
        if show_realtime_hrv:
            # SDNN
            df_sdnn = df_ecg_pieken.copy()
            df_sdnn['time_block'] = (df_sdnn['timestamp'] // 60) * 60
            sdnn_blocks = df_sdnn.groupby('time_block').apply(bereken_sdnn).reset_index(name='sdnn')
            laatste_sdnn = sdnn_blocks['sdnn'].iloc[-1]
            gemiddelde_sdnn = sdnn_blocks['sdnn'].mean()
            age_reference_sdnn = get_age_reference_sdnn(st.session_state.user_age)
            
            fig_sdnn = go.Figure()
            fig_sdnn.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=laatste_sdnn,
                number={'font': {'size': 40}, 'valueformat': '.1f'},
                title={'text': f"Live monitoring van SDNN<br><span style='font-size:0.8em;color:gray'>Referentie: {age_reference_sdnn} ms</span>", 
                       'font': {'size': 16}},
                delta={'reference': age_reference_sdnn, 
                       'increasing': {'color': "green"}, 
                       'decreasing': {'color': "red"},
                       'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 200], 
                            'tickwidth': 1,
                            'tickcolor': "darkgray",
                            'ticktext': ["Zeer Laag", "Laag", "Normaal", "Hoog"],
                            'tickvals': [15, 45, 75, 105]},
                    'bar': {'color': "purple", 'thickness': 0.15},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#ff9999'},
                        {'range': [30, 50], 'color': '#ffcc99'},
                        {'range': [50, 100], 'color': '#99ff99'},
                        {'range': [100, 200], 'color': '#99ccff'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': age_reference_sdnn
                    }
                }
            ))
            fig_sdnn.update_layout(height=400, margin=dict(t=100, b=100))
            st.plotly_chart(fig_sdnn, use_container_width=True)
        else:
            fig3 = visualize_sdnn_per_block(df_ecg_pieken.copy(), toon_sdnn_verloop)
            st.plotly_chart(fig3, use_container_width=True)
        
        with st.expander("ℹ️ Uitleg over de grafiek"):
            st.subheader("HRV-analyse: SDNN")
            st.write("""                 
Wat laat het zien:
De SDNN (Standard Deviation of NN intervals) is de standaarddeviatie van alle RR-intervallen. Dit is een globale maat voor hartslagvariatie die zowel korte als lange termijn variaties meeneemt.

Wat betekent dit:
SDNN is een algemene HRV-indicator die geldig is tijdens zowel rust als activiteit:

- **< 30 ms**: Zeer lage HRV - Mogelijk teken van chronische stress of verminderde hartgezondheid
- **30-50 ms**: Lage HRV - Ruimte voor verbetering in algemene hartflexibiliteit
- **50-100 ms**: Normale HRV - Gezonde hartflexibiliteit
- **> 100 ms**: Hoge HRV - Uitstekende hartflexibiliteit en aanpassingsvermogen

Leeftijdsreferentiewaarden voor SDNN:
- < 30 jaar: 164 ms
- 30-39 jaar: 151 ms
- 40-49 jaar: 141 ms
- 50-59 jaar: 143 ms
- > 60 jaar: 128 ms

Anders dan RMSSD, die vooral de korte-termijn variaties meet en het beste werkt tijdens rust, geeft SDNN een completer beeld van je hartritme variabiliteit over de hele meting.

*bron: Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate Variability Metrics and Norms. Frontiers in Public Health, 5, 258.*
                     """)

with col3:
    if show_rmssd_stressscore:
        st.subheader("Stressscore")
        rmssd_df = df_ecg_pieken.copy()
        rmssd_df['time_block'] = (rmssd_df['timestamp'] // 10) * 10
        rmssd_blocks = rmssd_df.groupby('time_block').apply(bereken_rmssd).reset_index(name='rmssd')
        fig4 = visualize_stressscore(rmssd_blocks)
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("ℹ️ Uitleg over de grafiek"):
            st.subheader("de RMSSD-stressscore")
            st.write("""
                     Wat laat het zien:
Deze grafiek geeft een samenvattende stressscore op basis van uw hartslagvariabiliteit (HRV), gemeten via de RMSSD-waarde. Hoe lager de variatie tussen uw hartslagen, hoe hoger de stressscore.

Hoe werkt het?

Uw HRV wordt omgerekend naar een stressscore tussen 0 en 100.

0 betekent: weinig stress – uw hart reageert flexibel en u bent ontspannen.

100 betekent: verhoogde stress – uw hart reageert minder flexibel.

Wat betekent dit voor u?

**Lage stressscore** (0–30): U bent ontspannen. Uw lichaam is goed in balans.

**Gemiddelde stressscore** (30–70): U bevindt zich in een normaal spanningsniveau. Mogelijk enige mentale of fysieke activiteit.

**Hoge stressscore** (70–100): Uw lichaam ervaart spanning. Dit kan wijzen op stress, vermoeidheid of onvoldoende herstel.

*bron: Caring Medical Florida. (2024, 12 juli). Heart rate variability. https://caringmedical.com/hauser-neck-center/heart-rate-variability/*
                    """)

with col4:
    if show_stappenteller:
        st.subheader("Stappenteller")
        df_bewerkt, stappen_index = bereken_stappenteller(df_ecg_pieken, drempel=0.1)
        totaal_stappen = df_bewerkt['stappenteller'].iloc[-1]
        
        # Calculate progress towards daily goal (typical goal is 10,000 steps)
        daily_goal = 10000
        progress = min(100, (totaal_stappen / daily_goal) * 100)
        
        # Display current steps with big number
        st.markdown(f"<h1 style='text-align: center; color: #0066cc;'>{int(totaal_stappen):,}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>stappen</p>", unsafe_allow_html=True)
        
        # Show progress bar
        st.progress(progress / 100)
        
        # Show goal information
        st.markdown(f"<p style='text-align: center;'>{int(totaal_stappen):,} / {daily_goal:,} stappen</p>", unsafe_allow_html=True)
        
        # Add context about step counts
        with st.expander("ℹ️ Over stappenteller"):
            st.write("""
            **Dagelijks stappendoel: 10.000 stappen**
            
            Verschillende activiteitsniveaus:
            - Nauwelijks actief: < 5.000 stappen
            - Licht actief: 5.000-7.499 stappen
            - Redelijk actief: 7.500-9.999 stappen
            - Actief: 10.000-12.499 stappen
            - Zeer actief: ≥ 12.500 stappen
            
            *Bron: Tudor-Locke et al. (2008) - International Journal of Behavioral Nutrition and Physical Activity*
            """)

with col5:
    # Calculate acceleration magnitude if not already done
    if 'samengesteld' not in df_ecg.columns:
        df_ecg['samengesteld'] = np.sqrt(df_ecg['accX']**2 + df_ecg['accY']**2 + df_ecg['accZ']**2)
    
    # Calculate intensity minutes
    df_intensiteit, total_intensity_minutes = berekenen_intensiteit(df_ecg, df_ecg_pieken)
    
    if show_bewegingsminuten:
        st.subheader("Bewegingsintensiteit")
        
        # Calculate moderate and vigorous minutes separately
        moderate_minutes = (df_intensiteit['intensiteit'] == 1).sum()
        vigorous_minutes = (df_intensiteit['intensiteit'] == 2).sum()
        
        # Create two columns for the metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Matige intensiteit",
                value=f"{moderate_minutes} min",
                help="Minuten waarin je stevig doorstapt of licht actief bent"
            )
        
        with metric_col2:
            st.metric(
                label="Hoge intensiteit",
                value=f"{vigorous_minutes} min (telt 2x)",
                help="Minuten waarin je intensief beweegt (tellen dubbel voor je dagelijkse doel)"
            )
        
        # Show total with progress towards daily goal
        daily_goal = 30  # Standard daily goal is often 30 minutes
        progress = min(100, (total_intensity_minutes / daily_goal) * 100)
        
        st.progress(progress / 100)
        st.metric(
            label="Totaal intensiteitsminuten",
            value=f"{total_intensity_minutes} / {daily_goal} min",
            help="Dagelijks doel: 30 minuten matige OF 15 minuten intensieve beweging"
        )
        
        with st.expander("ℹ️ Over bewegingsminuten"):
            st.write("""
            **Bewegingsminuten Doel: 30 min per dag**
            
            De Wereldgezondheidsorganisatie (WHO) beveelt aan:
            - 30 min matige intensiteit OF
            - 15 min hoge intensiteit
            
            Hoge intensiteit telt dubbel, dus bijvoorbeeld:
            - 20 min matig + 5 min intensief = 30 min totaal
            - 15 min intensief = 30 min totaal
            
            Alleen activiteiten van 10+ minuten aaneengesloten tellen mee.
            
            *Bron: WHO Guidelines on Physical Activity and Sedentary Behaviour (2020)*
            """)
    
    if show_bewegingsintensiteit:
        fig_intensiteit = plot_bewegingsintensiteit(df_intensiteit)
        st.plotly_chart(fig_intensiteit, use_container_width=True)
        with st.expander("ℹ️ Uitleg over de grafiek"):
            st.subheader("Wat laat deze grafiek zien?")
            st.write("""
In deze grafiek wordt per minuut weergegeven hoe intensief je bewoog, op basis van een combinatie van versnellingsgegevens en hartslag.

- **Grijs (Rust)**: Geen of minimale activiteit

- **Oranje (Matig)**: Je kunt een gesprek voeren, maar niet zingen tijdens het bewegen
  - Bijvoorbeeld: stevig doorwandelen, rustig fietsen
  - Telt 1x mee voor je dagelijkse doel
  
- **Rood (Intensief)**: Je kunt slechts enkele woorden spreken tussen ademhalingen
  - Bijvoorbeeld: joggen, sporten, traplopen
  - Telt 2x mee voor je dagelijkse doel
  
**Let op**: Alleen activiteiten van 10 minuten of langer tellen mee voor je dagelijkse doel.
            """)

with col6:
    if show_stappen_cumulatief:
        df_bewerkt, stappen_index = bereken_stappenteller(df_ecg_pieken, drempel=0.1)
        fig6 = plot_cumulatieve_stappen(df_bewerkt)
        st.plotly_chart(fig6, use_container_width=True)
        with st.expander("ℹ️ Uitleg over de grafiek"):
            st.write("""
            **Cumulatieve stappenteller**
            
            Deze grafiek toont hoe je stappen zich gedurende de tijd opbouwen:
            - De x-as toont de tijd in minuten
            - De y-as toont het totaal aantal stappen tot dat moment
            - De steilheid van de lijn geeft aan hoe snel je stapt:
              * Steile lijn = snel wandelen/rennen
              * Vlakke lijn = stilstaan/zitten
            """)


   
