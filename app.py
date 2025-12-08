"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    🏀 COURTSENSE - TACTICAL DASHBOARD                      ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st                         # Framework web interattivo
import pandas as pd                            # Manipolazione dati tabellari
import cv2                                     # Computer vision (lettura/disegno video)
import numpy as np                             # Operazioni numeriche
import os                                      # Gestione filesystem
import re                                      # Regex per parsing nomi file
import time                                    # Timing e sleep
import matplotlib.pyplot as plt                # Visualizzazione statici
import seaborn as sns                          # Grafici avanzati (heatmap, etc)
from scipy.spatial import Voronoi, ConvexHull # Geometria computazionale
from matplotlib.patches import Polygon, Rectangle, Circle  # Disegno forme
from itertools import combinations            # Combinatoria (pairwise distances)
import imageio.v2 as imageio                  # Generazione GIF
import shutil                                  # Operazioni file system avanzate

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE GLOBALE - PARAMETRI CRITICI
# ═══════════════════════════════════════════════════════════════════════════
CSV_FILE = 'tracking_data.csv'        # File CSV con dati tracking (posizioni giocatori/palla per frame)
IMAGES_FOLDER = 'train'               # Cartella base dove trovare frame video estratti
RADAR_HEIGHT = 300                    # Altezza radar tattico (pixel display)
COURT_WIDTH = 3840                    # Larghezza frame video (4K)
COURT_HEIGHT = 2160                   # Altezza frame video (4K)
POSSESSION_THRESHOLD = 60             # Tolleranza in pixel per rilevare possesso (Box-in-Box)

# ─────────────────────────────────────────────────────────────────────────
# PARAMETRI FISICI - Conversione Pixel ↔ Metri
# ─────────────────────────────────────────────────────────────────────────
FPS = 30.0                            # Frame rate video (frame al secondo)
REAL_WIDTH_M = 28.0                   # Larghezza reale campo (metri) - 
REAL_HEIGHT_M = 15.0                  # Altezza reale campo (metri)

# 🔧 FATTORE DI CONVERSIONE CRITICO
# Converte pixel in metri
# Logica: Se il frame 4K copre esattamente 28m di larghezza:
#   1 pixel = 28m / 3840px = 0.0073 m/px
PX_TO_M = REAL_WIDTH_M / COURT_WIDTH

# 🛡️ FILTRI PER ROBUSTEZZA - Eliminano dati sporchi/teletrasporti
MAX_PIXEL_STEP = 100                  # Se un giocatore salta >100px in 1 frame (teletrasporto) → ignora movimento
SMOOTHING_WINDOW = 5                  # Media mobile su 5 frame per smoothare posizioni rumorose

# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONI UTILITY - Parser e Disegno
# ═══════════════════════════════════════════════════════════════════════════
def extract_frame_number(filename):
  #  """
  #  ⬇️ Estrae numero frame da stringa filename
  #  Input:  "azione_01_frame_0045.jpg" → Output: 45
  #  Regex: Cerca pattern "frame_(\d+)" nel nome file
   # """
    match = re.search(r'frame_(\d+)', str(filename))
    if match: return int(match.group(1))
    return 0

def draw_radar_court(img, width, height, color=(200, 200, 200)):
   # """
  #  🎨 Disegna campo di basket su immagine OpenCV (in PIXEL)
   # ├─ Linea centrale verticale
   # ├─ Cerchio centro (R=1.8m)
  #  ├─ Rettangoli vernice (pitture) per ciascun lato
  #  ├─ Cerchi goaltender
   # └─ Archi 3 punti (se basket)
    
   # Args:
  #      img: Array OpenCV (BGR)
  #      width, height: Dimensioni in pixel (per radar tipicamente 600x300)
  #      color: Colore linee (B, G, R tuple)
    
   # Note: Usa PIXEL per disegno, ma scala basata su METRI reali
   # """
    thick = 2
    # Scala radar: quanti pixel del radar per 1 metro reale?
    ppm = width / REAL_WIDTH_M  # pixel per metro
    
    # Linea mediana (verticale centrale)
    cv2.line(img, (int(width/2), 0), (int(width/2), height), color, thick)
    # Cerchio centro campo (raggio 1.8m in una liga tipo EuroLega/NBA)
    cv2.circle(img, (int(width/2), int(height/2)), int(1.8 * ppm), color, thick)
    
    # VERNICE (Paint) - Rettangolo di tiro
    # Dimensioni standard: 5.8m x 4.9m
    paint_w = int(5.8 * ppm); paint_h = int(4.9 * ppm)
    y_top = int((height - paint_h) / 2); y_bot = int((height + paint_h) / 2)
    
    # Lato sinistro
    cv2.rectangle(img, (0, y_top), (paint_w, y_bot), color, thick)
    cv2.circle(img, (paint_w, int(height/2)), int(1.8 * ppm), color, thick)  # Cerchio goaltender SX
    
    # Lato destro
    cv2.rectangle(img, (width - paint_w, y_top), (width, y_bot), color, thick)
    cv2.circle(img, (width - paint_w, int(height/2)), int(1.8 * ppm), color, thick)  # Cerchio goaltender DX
    
    # ARCHI 3 PUNTI (se basket) - distanza 6.75m dalla linea di fondo
    three_pt = int(6.75 * ppm)
    cv2.ellipse(img, (0, int(height/2)), (three_pt, int(height*0.9)), 0, -90, 90, color, thick)      # Arco SX
    cv2.ellipse(img, (width, int(height/2)), (three_pt, int(height*0.9)), 0, 90, 270, color, thick)  # Arco DX
    return img

def draw_mpl_court(ax, color='black', lw=2):
  #  """
  #  Disegna campo in METRI per matplotlib (grafici statici/Voronoi/HeatMap)
  #  
   # Simile a draw_radar_court ma:
   # - Usa coordinate METRICHE (non pixel)
   # - Disegna su asse matplotlib
   # - Perfetto per overlay con dati in metri
    
  #  Args:
  #      ax: Asse matplotlib
    #    color: Colore linee
   #     lw: Line width
   # """
    # Rettangolo campo intero
    court = Rectangle((0, 0), REAL_WIDTH_M, REAL_HEIGHT_M, linewidth=lw, color=color, fill=False)
    ax.add_patch(court)
    
    # Linea mediana (Y=14m, che è metà di 28m)
    ax.plot([14, 14], [0, 15], color=color, linewidth=lw)
    
    # Cerchio centro (raggio 1.8m)
    ax.add_patch(Circle((14, 7.5), 1.8, color=color, fill=False, linewidth=lw))
    
    # Vernici
    ax.add_patch(Rectangle((0, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))      # SX
    ax.add_patch(Rectangle((22.2, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))   # DX

@st.cache_data
def load_data():
   # """
    #Carica e processa CSV di tracking
    
   # Pipeline:
   # 1. Leggi CSV
   # 2. Estrai action_id dal nome file (es: "azione_01")
   # 3. Estrai frame_id numerici
   # 4. Crea player_unique_id = team + numero (es: "Red_5", "White_12")
   # 5. CONVERTI coordinate da pixel a METRI al volo
   #    ├─ x_feet * PX_TO_M → x_meters
   #    └─ y_feet * (REAL_HEIGHT_M/COURT_HEIGHT) → y_meters
    
   # Returns:
   #     DataFrame processato con colonne:
   #     - frame_filename, frame_id, action_id
   #     - team, number, player_unique_id
   #     - bbox_x/y/w/h (bounding box in pixel)
   #     - x_feet, y_feet (posizioni raw in pixel)
   #     - x_meters, y_meters (posizioni convertite in METRI) ✓ CRITICO
   # """
    if not os.path.exists(CSV_FILE): return None
    df = pd.read_csv(CSV_FILE)
    
    # Estrai action_id da nome file
    if 'action_id' not in df.columns:
        df['action_id'] = df['frame_filename'].apply(lambda x: x.split('_frame')[0] if '_frame' in x else 'unknown')
    
    # Estrai numero frame
    df['frame_id'] = df['frame_filename'].apply(extract_frame_number)
    
    # Crea ID unico giocatore
    df['player_unique_id'] = df['team'] + "_" + df['number'].astype(str)
    
    # 🔧 CONVERSIONE PIXEL → METRI (il fulcro dell'analisi ibrida)
    if 'x_meters' not in df.columns:
        df['x_meters'] = df['x_feet'] * PX_TO_M                                    # X: conversione semplice
        df['y_meters'] = df['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)             # Y: ratio diverso
        
    return df

# ═══════════════════════════════════════════════════════════════════════════
# LOGICA POSSESSO - Rileva chi possiede la palla
# ═══════════════════════════════════════════════════════════════════════════
def get_possession_table(df_subset):
  #  """
  #  🏀 ALGORITMO BOX-IN-BOX - Determina chi possiede la palla
    
  #  Logica:
  #  1. Estrai bounding box della palla (Ball)
  #  2. Per ogni frame, controlla quali giocatori hanno palla nel loro box
  #  3. Se multipli giocatori "contengono" la palla → vince il più vicino
    
  #  Dettagli tecnici:
  #  ├─ Shrink 5% sulla larghezza e 10% sull'altezza per robustezza
  #  ├─ Usa centro bounding box palla (bx_c, by_c)
  #  ├─ Confronta con box giocatore ridotto
  #  └─ Distanza euclidea in PIXEL per risolvere conflitti
    
  #  Args:
  #      df_subset: DataFrame con frame di una azione
    
   # Returns:
   #     DataFrame con colonne [frame_id, player_unique_id] per frame con possesso
   #     (Indice = frame_id, valore = chi possiede la palla)
   # """
    # Estrai dati palla e giocatori
    ball_df = df_subset[df_subset['team'] == 'Ball'][['frame_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
    players_df = df_subset[df_subset['team'].isin(['Red', 'White'])].copy()
    
    if ball_df.empty or players_df.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    # Join su frame_id: per ogni frame, accoppia ogni giocatore con la palla
    m = pd.merge(players_df, ball_df, on='frame_id', suffixes=('', '_b'), how='inner')
    
    # BOX-IN-BOX: Shrink sui Pixel per robustezza
    # Riduce il box giocatore del 5% su X e 10% su Y
    mw, mh = m['bbox_w']*0.05, m['bbox_h']*0.10
    bx_c = m['bbox_x_b'] + m['bbox_w_b']/2          # Centro X palla
    by_c = m['bbox_y_b'] + m['bbox_h_b']/2          # Centro Y palla
    px1, px2 = m['bbox_x']+mw, m['bbox_x']+m['bbox_w']-mw       # Bordi X box giocatore (ridotto)
    py1, py2 = m['bbox_y'], m['bbox_y']+m['bbox_h']-mh          # Bordi Y box giocatore (ridotto)
    
    # Test overlap: è il centro della palla dentro il box del giocatore?
    m['is_overlap'] = (bx_c > px1) & (bx_c < px2) & (by_c > py1) & (by_c < py2)
    
    candidates = m[m['is_overlap']].copy()
    if candidates.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    # RISOLUZIONE CONFLITTO: Vince il più vicino in PIXEL
    p_cx = candidates['bbox_x'] + candidates['bbox_w']/2
    p_cy = candidates['bbox_y'] + candidates['bbox_h']/2
    candidates['dist_px'] = np.sqrt((p_cx - bx_c)**2 + (p_cy - by_c)**2)
    
    # Per ogni frame, tieni solo il giocatore più vicino
    best_idx = candidates.groupby('frame_id')['dist_px'].idxmin()
    owners = candidates.loc[best_idx, ['frame_id', 'player_unique_id']].set_index('frame_id')
    return owners

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICHE IBRIDE - Calcoli in PIXEL, output in METRI
# ═══════════════════════════════════════════════════════════════════════════
def calculate_advanced_stats_hybrid(df_action, player_id, current_frame, ownership_table):
  #  """
  #  📈 Calcola metriche avanzate PER UN GIOCATORE fino a un frame specifico
    
  #  Metriche calcolate:
  #  1️⃣  total_dist_m        → Distanza totale percorsa (m)
   # 2️⃣  off_ball_m          → Distanza percorsa SENZA palla (m)
  #  3️⃣  speed_kmh           → Velocità media ultimi 15 frame (km/h)
  #  4️⃣  poss_frames         → Frame in cui possiede palla
    
    #Pipeline:
   # ├─ Filtra dati giocatore fino a current_frame
  #  ├─ SMOOTHING in pixel (media mobile 5 frame)
  #  ├─ Calcola step in pixel, filtra picchi >100px
  #  ├─ Converte in METRI tramite PX_TO_M
  #  ├─ Determina frame con possesso da ownership_table
   # └─ Separa distanza On-Ball vs Off-Ball
    
  #  Args:
  #      df_action: DataFrame dell'intera azione
   #     player_id: ID giocatore univoco (es: "Red_5")
  #      current_frame: Frame fino al quale calcolare
  #      ownership_table: DataFrame con chi possiede palla per frame
    
  #  Returns:
  #      (total_dist_m, off_ball_m, speed_kmh, poss_frames)
   # """
    # Filtra dati del giocatore fino al frame corrente
    p_data = df_action[(df_action['player_unique_id'] == player_id) & (df_action['frame_id'] <= current_frame)].sort_values('frame_id')
    if len(p_data) < 5: return 0, 0, 0, 0  # Skip se dati insufficienti
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1️⃣  DISTANZA TOTALE (in PIXEL, con smoothing e filtering)
    # ═══════════════════════════════════════════════════════════════════════
    # Applica media mobile per smoothare posizioni rumorose
    p_data['xp'] = p_data['x_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    p_data['yp'] = p_data['y_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    
    # Calcola step tra frame consecutivi
    dx = p_data['xp'].diff().fillna(0)
    dy = p_data['yp'].diff().fillna(0)
    step_px = np.sqrt(dx**2 + dy**2)
    
    # Filtra picchi pixel (teletrasporti) > 100px in 1 frame
    step_px[step_px > MAX_PIXEL_STEP] = 0
    
    # CONVERSIONE: pixel → METRI
    total_dist_m = step_px.sum() * PX_TO_M
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2️⃣  VELOCITÀ (ultimi 15 frame)
    # ═══════════════════════════════════════════════════════════════════════
    if len(p_data) > 15:
        dist_last_15_px = step_px.tail(15).sum()
        dist_last_15_m = dist_last_15_px * PX_TO_M
        time_s = 15 / FPS                           # 15 frame a 30 FPS = 0.5 secondi
        speed_kmh = (dist_last_15_m / time_s) * 3.6  # m/s → km/h
    else: 
        speed_kmh = 0
    
    # Rumore gate: velocità < 1 km/h → poni a 0 (giocatore fermo)
    if speed_kmh < 1.0: speed_kmh = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣  POSSESSO (da ownership_table) - Separa On-Ball vs Off-Ball
    # ═══════════════════════════════════════════════════════════════════════
    # Join con ownership_table per sapere chi possiede la palla ogni frame
    p_data = p_data.join(ownership_table, on='frame_id', rsuffix='_owner')
    p_data['is_mine'] = (p_data['player_unique_id_owner'] == player_id)
    
    # Buffer 3-frame: se almeno 2 su 3 frame dicono "possesso", allora è possesso
    is_mine_buf = p_data['is_mine'].rolling(window=3, center=True, min_periods=1).sum() >= 2
    
    # Distanza Off-Ball: movimento SENZA palla
    off_ball_px = step_px[~is_mine_buf].sum()
    off_ball_m = off_ball_px * PX_TO_M
    
    # Frame in cui possiede la palla
    poss_frames = is_mine_buf.sum()
    
    return int(total_dist_m), int(off_ball_m), round(speed_kmh, 1), poss_frames

# ═══════════════════════════════════════════════════════════════════════════
# GRAFICI STATICI - Voronoi, Convex Hull, Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def generate_static_voronoi(frame_data, title=None):
 #   """
 #   🔷 Genera diagramma di Voronoi per un frame
    
  #  Voronoi = divisione campo in aree (una per giocatore) dove ogni pixel
  #  appartiene al giocatore più vicino. Utile per analizzare "spazio controllato"
    
  #  Processo:
  #  1. Estrai posizioni giocatori in METRI
  #  2. Aggiungi punti dummy attorno al campo (evita artefatti bordo)
  #  3. Calcola diagramma Voronoi
   # 4. Colora regioni: Red=rosso, White=blu
  #  5. Sovrapponi posizioni giocatori e palla
    
  #  Args:
   #     frame_data: DataFrame con dati di UN frame
   #     title: Titolo del grafico
    
   # Returns:
   #     Figure matplotlib
   # """
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    
    # Estrai giocatori in METRI
    players = frame_data[frame_data['team'].isin(['Red', 'White'])]
    
    # Calcola Voronoi se almeno 4 giocatori
    if len(players) >= 4:
        # Usa coordinate in METRI
        points = players[['x_meters', 'y_meters']].values
        teams = players['team'].values
        
        # Dummy points: 4 angoli attorno al campo (per robustezza)
        dummy = np.array([[-5, -5], [35, -5], [35, 20], [-5, 20]])
        try:
            vor = Voronoi(np.vstack([points, dummy]))
            
            # Disegna regioni Voronoi
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    c = 'red' if teams[i] == 'Red' else 'blue'
                    ax.add_patch(Polygon(vor.vertices[region], facecolor=c, alpha=0.4, edgecolor='white'))
        except: 
            pass  # Se Voronoi fallisce, continua senza
    
    # Sovrapponi posizioni giocatori
    for t, c in [('Red', 'red'), ('White', 'blue')]:
        tp = players[players['team'] == t]
        ax.scatter(tp['x_meters'], tp['y_meters'], c=c, s=80, edgecolors='white', zorder=5)
    
    # Sovrappo palla
    ball = frame_data[frame_data['team']=='Ball']
    if not ball.empty: 
        ax.scatter(ball['x_meters'], ball['y_meters'], c='orange', s=150, edgecolors='black', zorder=10)
    
    ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off')
    if title: ax.set_title(title, fontsize=15)
    return fig

def generate_static_hull(frame_data):
  #  """
  #  Convex Hull - Disegna forma "involucro" di ogni squadra
   # 
   # Convex Hull = poligono minimo che racchiude tutti i giocatori di una squadra.
   # Utile per visualizzare compattezza/dislocazione tattica
    
  #  Returns:
  #      Figure matplotlib
  #  """
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    
    colors = {'Red': 'red', 'White': 'blue'}
    fill = {'Red': 'salmon', 'White': 'lightblue'}
    
    for team in ['Red', 'White']:
        points = frame_data[frame_data['team'] == team][['x_meters', 'y_meters']].values
        
        # Scatter dei giocatori
        ax.scatter(points[:,0], points[:,1], c=colors[team], s=80, zorder=5)
        
        # Calcola e disegna convex hull
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                poly = Polygon(points[hull.vertices], facecolor=fill[team], edgecolor=colors[team], 
                             alpha=0.3, lw=2, linestyle='--')
                ax.add_patch(poly)
            except: 
                pass  # Se ConvexHull fallisce, continua
    
    ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off')
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# RENDERING VIDEO - Dual View (Frame + Radar)
# ═══════════════════════════════════════════════════════════════════════════
def render_dual_view(f_id, df, quality_mode, highlight_id=None, is_possessor=False):
 #   """
 #   🎬 Rendering frame video + radar tattico
    
  #  Questo è il motore visuale del sistema. Per ogni frame:
  #  1. Carica immagine video originale
   # 2. Crea radar tattico in METRI
  #  3. Disegna giocatori/palla su entrambi
  #  4. Se highlight_id specificato, marca quel giocatore
   # 5. Controlla possesso palla (visual overlap)
    
  #  Quality modes:
  #  - "Ottimizzata (HD)": ridimensiona a 1280px (più veloce)
  #  - "Massima (4K)": mantiene risoluzione originale (preciso ma lento)
    
  #  Args:
   #     f_id: Frame ID numerico
   #     df: DataFrame dell'azione
   #     quality_mode: "Ottimizzata (HD)" o "Massima (4K)"
   #     highlight_id: player_unique_id da evidenziare (es: "Red_5")
  #      is_possessor: True se giocatore in highlight possiede palla
    
   # Returns:
   #     (video_RGB, radar_RGB, red_count, white_count, ref_count, is_holding_ball)
   # """
    # Trova riga con questo frame_id
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None, None, 0, 0, 0, False
    fname = fname_row['frame_filename'].iloc[0]
    
    # Trova path immagine (prova molteplici location)
    img_path = None
    if 'image_path' in fname_row.columns:
        p = fname_row['image_path'].iloc[0]
        if os.path.exists(p): img_path = p
    if img_path is None: img_path = os.path.join(IMAGES_FOLDER, fname)
    if not os.path.exists(img_path): img_path = os.path.join('train', fname)
        
    # Carica immagine
    frame_img_orig = cv2.imread(img_path)
    if frame_img_orig is None: return None, None, 0, 0, 0, False

    # Applica quality mode (ridimensionamento)
    scale_factor = 1280 / frame_img_orig.shape[1] if quality_mode == "Ottimizzata (HD)" else 1.0
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale_factor, fy=scale_factor) if scale_factor != 1.0 else frame_img_orig
    
    # Crea radar vuoto
    radar_img = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_img, 600, RADAR_HEIGHT)
    
    # Dati frame corrente
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M  # Scala: pixel radar per metro
    red_c, white_c, ref_c = 0, 0, 0
    ball_row = frame_data[frame_data['team'] == 'Ball']
    is_holding_ball = False

    # Disegna tutti i soggetti (giocatori, palla, arbitri)
    for _, row in frame_data.iterrows():
        # ┌─ Converti coordinate METRICHE → PIXEL RADAR
        rx = int(row['x_meters'] * sx)
        ry = int(row['y_meters'] * sy)
        rx = max(0, min(rx, 600-1))
        ry = max(0, min(ry, RADAR_HEIGHT-1))
        
        t_str = str(row['team'])
        raw_c = str(row.get('raw_class', '')).lower()
        
        # ├─ PALLA (arancione + cerchio bianco di contorno)
        if t_str == 'Ball':
            cv2.circle(radar_img, (rx, ry), 8, (0, 165, 255), -1)
            cv2.circle(radar_img, (rx, ry), 10, (255, 255, 255), 1)
        
        # ├─ ARBITRI (verde)
        elif 'Ref' in t_str or 'ref' in raw_c:
            cv2.circle(radar_img, (rx, ry), 5, (0, 255, 0), -1)
            ref_c += 1
        
        # └─ GIOCATORI (rosso o bianco)
        elif t_str in ['Red', 'White']:
            c = (0, 0, 255) if t_str == 'Red' else (255, 255, 255)  # BGR!
            if t_str == 'Red': red_c += 1
            else: white_c += 1
            cv2.circle(radar_img, (rx, ry), 6, c, -1)
            
            # 🎯 HIGHLIGHT PLAYER
            if highlight_id and row['player_unique_id'] == highlight_id:
                # ┌─ Disegna box su VIDEO (pixel)
                bx, by = int(row['bbox_x']*scale_factor), int(row['bbox_y']*scale_factor)
                bw, bh = int(row['bbox_w']*scale_factor), int(row['bbox_h']*scale_factor)
                col_box = (0, 165, 255) if is_possessor else (0, 255, 255)  # Ciano vs Blu
                cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 3)
                cv2.putText(frame_img, row['player_unique_id'], (bx, by-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_box, 2)
                
                # ├─ Disegna cerchio su RADAR (metri)
                cv2.circle(radar_img, (rx, ry), 12, col_box, 2)
                
                # └─ Verifica visivamente se possiede palla (Box-in-Box pixel)
                if not ball_row.empty:
                    b = ball_row.iloc[0]
                    bcx, bcy = b['bbox_x']+b['bbox_w']/2, b['bbox_y']+b['bbox_h']/2
                    mw, mh = row['bbox_w']*0.05, row['bbox_h']*0.10
                    if (row['bbox_x']+mw < bcx < row['bbox_x']+row['bbox_w']-mw) and \
                       (row['bbox_y'] < bcy < row['bbox_y']+row['bbox_h']-mh):
                        is_holding_ball = True

    # Converti BGR → RGB per Streamlit
    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB), \
           red_c, white_c, ref_c, is_holding_ball

# ═══════════════════════════════════════════════════════════════════════════
# INTERFACCIA STREAMLIT - Main UI
# ═══════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════════════
# MAIN - Initiali Streamlit
# ═════════════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="CourtSense Dashboard", layout="wide")
st.title("🏀 CourtSense: Tactical Dashboard")

# Carica dati CSV
df_full = load_data()
if df_full is None: 
    st.error("CSV non trovato.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────────────
# SIDEBAR - Pannello di controllo
# ─────────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Pannello di Controllo")

available_actions = sorted(df_full['action_id'].unique())
if not available_actions: 
    st.error("Nessuna azione trovata.")
    st.stop()

# Default: seleziona 'out13' se esiste, altrimenti primo disponibile
idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("📂 Seleziona Azione:", available_actions, index=idx)

# Filtra DataFrame per azione selezionata
df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
if len(df) == 0: 
    st.warning("Dati vuoti.")
    st.stop()

# Calcola possesso globale (tabella di chi possiede la palla ogni frame)
ownership_table = get_possession_table(df)

# Setup variabili globali
unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

# Selezioni sidebar
analysis_mode = st.sidebar.radio("Modalità:", ("🕹️ Navigazione (Manuale)", "▶️ Riproduzione (Auto)"))
quality_mode = st.sidebar.radio("Qualità:", ("Ottimizzata (HD)", "Massima (4K)"))
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Player Focus")
selected_player = st.sidebar.selectbox("Traccia Giocatore:", player_list)

# ─────────────────────────────────────────────────────────────────────────────────────
# LAYOUT - Main content area
# ─────────────────────────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([3, 1])
video_ph = col_main.empty()          # Placeholder video frame
radar_ph = col_side.empty()          # Placeholder radar tattico
stats_ph = col_side.empty()          # Placeholder statistiche giocatore

def update_ui_elements(fid, dist_tot, dist_off, poss_frames, speed):
  #  """
   # 🔄 Helper per aggiornare placeholder UI
    
   # Effettua:
   # 1. Rendering dual view per il frame
   # 2. Aggiorna video, radar, e statistiche
   # 3. Mostra icon 🏀 se giocatore possiede palla
   # """
    is_owner = False
    try:
        if fid in ownership_table.index and ownership_table.loc[fid]['player_unique_id'] == selected_player:
            is_owner = True
    except: 
        pass
    
    vid, rad, rc, wc, ref, hold = render_dual_view(fid, df, quality_mode, selected_player, is_owner)
    
    if vid is not None:
        video_ph.image(vid, channels="RGB", width='stretch')
        radar_ph.image(rad, channels="RGB", caption="Tactical Board (Meters)", width='stretch')
        icon = "🏀" if is_owner else ""
        stats_ph.markdown(f"""
        ### Frame: {fid}
        **Focus:** `{selected_player}` {icon}
        📏 **Dist:** {dist_tot} m
        🏃 **Off-Ball:** {dist_off} m
        ⚡ **Speed:** {speed} km/h
        ⏱️ **Poss:** {(poss_frames/FPS):.1f} s
        ---
        🔴 R:{rc} | ⚪ W:{wc}
        """)
    return hold

# ═════════════════════════════════════════════════════════════════════════════════════
# MODALITÀ 1: NAVIGAZIONE MANUALE
# ═════════════════════════════════════════════════════════════════════════════════════

if analysis_mode == "🕹️ Navigazione (Manuale)":
    sel_frame = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, selected_player, sel_frame, ownership_table)
    update_ui_elements(sel_frame, dt, do, pf, spd)
    
    st.markdown("---"); st.subheader("📊 Analisi Puntuale"); c1, c2 = st.columns(2)
    fdata = df[df['frame_id'] == sel_frame]
    if c1.button("📸 Voronoi"): c1.pyplot(generate_static_voronoi(fdata))
    if c2.button("🛡️ Convex Hull"): c2.pyplot(generate_static_hull(fdata))

else: # AUTO
    st.sidebar.markdown("---")
    start, end = st.sidebar.select_slider("Clip:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    fps = st.sidebar.slider("FPS:", 1, 60, 25)
    
    if st.sidebar.button("▶️ PLAY", type="primary"):
        
        #🎬 Loop di riproduzione video
        
        #Per ogni frame nel range [start, end]:
        #- Accumula distanza totale (cum_m)
        #- Separa distanza Off-Ball (cum_off_m)
        #- Calcola velocità da ultimo buffer di 5 frame
        #- Aggiorna UI in tempo reale
        #- Sincronizza con FPS per smoothness
        
        frames = [f for f in unique_frames if start <= f <= end]
        cum_m = 0                    # Distanza cumulativa totale (m)
        cum_off_m = 0                # Distanza cumulativa Off-Ball (m)
        prev_x_px = None; prev_y_px = None  # Coordinate precedenti per calcolo step
        pos_buffer = []              # Buffer 10 frame per smoothing posizione
        curr_poss_frames = 0         # Frame conteggiati con possesso
        

        for f_id in frames:
            t0 = time.time()
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            step_m = 0; speed_kmh = 0.0
            
            if not curr.empty:
                # 📍 Estrai coordinate pixel giocatore
                cx, cy = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buffer.append((cx, cy))              # Aggiungi buffer
                if len(pos_buffer)>10: pos_buffer.pop(0) # Mantieni max 10 frame
                
                # Smoothing su ultimi 3 frame
                if len(pos_buffer)>=1:
                    cx_s = np.mean([p[0] for p in pos_buffer[-3:]])
                    cy_s = np.mean([p[1] for p in pos_buffer[-3:]])
                    
                    # Calcola step da frame precedente (se esiste)
                    if prev_x_px is not None:
                        raw_step_px = np.sqrt((cx_s-prev_x_px)**2 + (cy_s-prev_y_px)**2)
                        # Filter: ignora picchi >100px
                        if raw_step_px < MAX_PIXEL_STEP:
                            step_m = raw_step_px * PX_TO_M  # Converti PIXEL → METRI
                            cum_m += step_m
                    prev_x_px, prev_y_px = cx_s, cy_s

                # Velocità: da buffer 5 frame
                if len(pos_buffer)>=5:
                    d_px = np.sqrt((pos_buffer[-1][0]-pos_buffer[0][0])**2 + 
                                   (pos_buffer[-1][1]-pos_buffer[0][1])**2)
                    tm = (len(pos_buffer)-1)/FPS  # Tempo in secondi
                    raw_spd = (d_px * PX_TO_M / tm) * 3.6  # PIXEL → METRI → km/h
                    if raw_spd > 1.0: speed_kmh = min(raw_spd, 36.0)  # Cap a 36 km/h

            # Verifica possesso
            is_owner = False
            try:
                if f_id in ownership_table.index and ownership_table.loc[f_id]['player_unique_id'] == selected_player:
                    is_owner = True
            except: pass
            
            # Accumula metriche
            if is_owner: 
                curr_poss_frames += 1  # Frame con palla
            else: 
                cum_off_m += step_m    # Aggiungi distanza Off-Ball
            
            # 🎨 Aggiorna UI
            update_ui_elements(f_id, int(cum_m), int(cum_off_m), curr_poss_frames, round(speed_kmh, 1))
            
            # ⏱️ Sincronizza FPS: attendi per ottenere velocità corretta
            time.sleep(max(0.02, (1.0/fps) - (time.time()-t0)))

    st.markdown("---")
    st.subheader("📈 Report Azione")
    c1, c2, c3 = st.columns(3)
    if c1.button("Genera Metriche"):
        
        #REPORT AGGREGATO - Calcola statistiche per TUTTA la clip
        
        #Analisi:
        #1. SPACING: Distanza media tra giocatori (per team)
        #2. MOVIMENTO: Distanza per giocatore (On/Off-Ball)
        #3. POSSESSO: Tempo di possesso per giocatore
        #4. VELOCITÀ: Media km/h per giocatore
        
        #Output:
        #- Grafici linea (spacing nel tempo)
        #- Grafici barre (workload per giocatore)
        #- Confronti Red vs White
        
        with st.spinner("Calcolo..."):
            # Filtra dati nel range start-end
            sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            duration_s = (end - start + 1) / FPS
            
            # ─────────────────────────────────────────────────────────────────
            # 1️⃣  SPACING: Distanza media intra-team per frame
            # ─────────────────────────────────────────────────────────────────
            #
            #Per ogni frame e team, calcola tutte le distanze pairwise tra giocatori
            #Formula: media di ||P_i - P_j|| per i<j in team
            #Converti in METRI
            #*
            spac = []
            for f, g in players.groupby('frame_id'):
                for t in ['Red', 'White']:
                    tg = g[g['team']==t]
                    if len(tg)>=2:
                        # Tutte le coppie pairwise
                        dists = [np.linalg.norm(a-b) for a,b in combinations(tg[['x_feet','y_feet']].values, 2)]
                        spac.append({'f':f, 't':t, 'v':np.mean(dists) * PX_TO_M})  # PIXEL → METRI
            
            if spac:
                sdf = pd.DataFrame(spac)
                fig, ax = plt.subplots(figsize=(6, 4))
                # Linea: Spacing nel tempo, differenziato per team
                sns.lineplot(data=sdf, x='f', y='v', hue='t', palette={'Red':'red','White':'blue'}, ax=ax)
                # Linee orizzontali: Media per team
                mr = sdf[sdf['t']=='Red']['v'].mean()
                mw = sdf[sdf['t']=='White']['v'].mean()
                ax.axhline(mr, c='darkred', ls='--', label=f"R:{mr:.1f}m")
                ax.axhline(mw, c='darkblue', ls='--', label=f"W:{mw:.1f}m")
                ax.set_title("Avg Spacing (Meters)")
                ax.legend(fontsize='small')
                c1.pyplot(fig)
            
            # ─────────────────────────────────────────────────────────────────
            # 2️⃣  MOVIMENTO: Distanze per giocatore (On/Off Ball)
            # ─────────────────────────────────────────────────────────────────
            #
            #Per ogni giocatore:
            #- Calcola distanza TOTALE
            #- Separa con/senza palla
            #- Stima velocità media
            #
            moves = []
            speed_poss = []
            own_sub = ownership_table[ownership_table.index.isin(sub['frame_id'].unique())]
            
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                
                # Smoothing: media mobile 3 frame su pixel
                g['xm'] = g['x_feet'].rolling(3).mean()
                g['ym'] = g['y_feet'].rolling(3).mean()
                
                # Calcola step in PIXEL
                steps_px = np.sqrt(np.diff(g['xm'], prepend=g['xm'].iloc[0])**2 + 
                                  np.diff(g['ym'], prepend=g['ym'].iloc[0])**2)
                
                # Filtra picchi
                steps_px = np.where(steps_px > MAX_PIXEL_STEP, 0, steps_px)
                
                # Converte PIXEL → METRI
                steps_m = steps_px * PX_TO_M
                
                # Determina quali frame aveva possesso
                is_poss = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                
                # Metriche aggregate
                tot = np.nansum(steps_m)                        # Distanza totale (m)
                off = np.nansum(steps_m[~is_poss])              # Distanza Off-Ball (m)
                poss_s = is_poss.sum()/FPS                      # Tempo possesso (s)
                avg_spd = (tot/duration_s)*3.6 if duration_s>0 else 0  # km/h
                
                # Aggiungi record
                moves.append({'Player':pid, 'Dist':tot, 'Type':'Total', 'Team':g['team'].iloc[0]})
                moves.append({'Player':pid, 'Dist':off, 'Type':'Off-Ball', 'Team':g['team'].iloc[0]})
                speed_poss.append({'Player':pid, 'Team':g['team'].iloc[0], 'Speed':avg_spd, 'Poss':poss_s})
            
            if moves:
                #
                #─────────────────────────────────────────────────────────────────
                #3️⃣  VISUALIZZAZIONE REPORT
                #─────────────────────────────────────────────────────────────────
                
                #Crea DataFrames con risultati aggregati:
                #- mdf: Movimento per giocatore (On/Off)
                #- spdf: Possesso e Velocità
                
                #Calcola medie per team e crea 4 grafici:
                #1. Spacing nel tempo
                #2. Workload (distanza totale vs off-ball)
                #3. Possesso (tempo in secondi)
                #4. Velocità media (km/h)
                #
                mdf = pd.DataFrame(moves)
                spdf = pd.DataFrame(speed_poss)
                
                # Calcola medie per team
                ar_t = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Total')]['Dist'].mean()
                ar_o = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
                asp_r = spdf[spdf['Team']=='Red']['Speed'].mean()
                ap_r = spdf[spdf['Team']=='Red']['Poss'].mean()
                
                aw_t = mdf[(mdf['Team']=='White')&(mdf['Type']=='Total')]['Dist'].mean()
                aw_o = mdf[(mdf['Team']=='White')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
                asp_w = spdf[spdf['Team']=='White']['Speed'].mean()
                ap_w = spdf[spdf['Team']=='White']['Poss'].mean()
                
                # Info box: Riepilogo team
                k1, k2 = st.columns(2)
                k1.info(f"🔴 Red: Tot {ar_t:.1f}m | Off {ar_o:.1f}m | Spd {asp_r:.1f}km/h | Poss {ap_r:.1f}s")
                k2.info(f"⚪ White: Tot {aw_t:.1f}m | Off {aw_o:.1f}m | Spd {asp_w:.1f}km/h | Poss {ap_w:.1f}s")
                
                # Grafico 1: WORKLOAD - Distanza totale vs Off-Ball per giocatore
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=mdf, x='Player', y='Dist', hue='Type', 
                           palette={'Total':'gray', 'Off-Ball':'limegreen'}, ax=ax2)
                # Linee di riferimento per media team
                ax2.axhline(ar_t, c='darkred', ls='--', label=f"Tot R")
                ax2.axhline(ar_o, c='red', ls=':', label=f"Off R")
                ax2.axhline(aw_t, c='darkblue', ls='--', label=f"Tot W")
                ax2.axhline(aw_o, c='blue', ls=':', label=f"Off W")
                ax2.tick_params(axis='x', rotation=90)
                ax2.set_title("Workload (Meters)")
                ax2.legend(fontsize='x-small')
                c1.pyplot(fig2)
                
                # Grafico 2: POSSESSO - Tempo palla in secondi
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=spdf, x='Player', y='Poss', hue='Team', 
                           palette={'Red':'red', 'White':'blue'}, ax=ax3)
                ax3.tick_params(axis='x', rotation=90)
                ax3.set_title("Possession (s)")
                c2.pyplot(fig3)
                
                # Grafico 3: VELOCITÀ - Media km/h
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=spdf, x='Player', y='Speed', hue='Team', 
                           palette={'Red':'red', 'White':'blue'}, ax=ax4)
                ax4.axhline(asp_r, c='darkred', ls='--')
                ax4.axhline(asp_w, c='darkblue', ls='--')
                ax4.tick_params(axis='x', rotation=90)
                ax4.set_title("Avg Speed (km/h)")
                c3.pyplot(fig4)

    if c2.button("GIF Voronoi"):
        #"""
        #GENERATORE GIF VORONOI
        
        #Pipeline:
        #1. Estrai lista frame nel range [start, end]
        #2. Per ogni frame, genera un grafico Voronoi
        #3. Salva ogni frame come PNG temporaneo
        #4. Assembla tutti i PNG in una GIF animata
        #5. Pulisci file temporanei
        
        #Utile per visualizzare come cambia il "possesso spazio" nel tempo
        #"""
        # 1️⃣  Recupera la lista dei frame da processare
        frames_list = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)]['frame_filename'].unique()
        
        if len(frames_list) == 0:
            st.error("Nessun frame selezionato.")
        else:
            # UI Progress Bar
            prog_bar = st.progress(0, text="Inizializzazione rendering...")
            
            # 2️⃣  Cartella temporanea per i frame
            temp_dir = "temp_voronoi_gif"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            generated_files = []
            
            try:
                # 3️⃣  Generazione Frame-by-Frame
                for i, fn in enumerate(frames_list):
                    # Aggiorna progress bar
                    percent = int((i / len(frames_list)) * 90)
                    prog_bar.progress(percent, text=f"Rendering Frame {i+1}/{len(frames_list)}")
                    
                    # Estrai dati del singolo frame
                    frame_data = df[df['frame_filename'] == fn]
                    
                    # Genera figura Voronoi per questo frame (in METRI)
                    frame_num = extract_frame_number(fn)
                    fig = generate_static_voronoi(frame_data, title=f"Space Ownership | Frame {frame_num}")
                    
                    # Salva immagine temporanea (PNG)
                    # dpi=60 è compromesso tra qualità e velocità/RAM
                    out_path = os.path.join(temp_dir, f"v_{i:04d}.png")
                    fig.savefig(out_path, dpi=60, bbox_inches='tight')
                    plt.close(fig)  # CRITICO: chiude figura per liberare RAM

                    generated_files.append(out_path)
                
                # 4️⃣  Assemblaggio GIF da PNG
                prog_bar.progress(95, text="Compilazione GIF...")
                gif_output = "action_voronoi.gif"
                
                # duration=0.15s = 6-7 FPS (buono per visualizzare tattica senza essere troppo veloce)
                with imageio.get_writer(gif_output, mode='I', duration=0.15) as writer:
                    for filename in generated_files:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                
                # 5️⃣  Mostra risultato
                prog_bar.empty()
                st.success(f"GIF Generata ({len(generated_files)} frames)")
                st.image(gif_output, width='stretch')
                
            except Exception as e:
                st.error(f"Errore durante la generazione della GIF: {e}")
                
            finally:
                # 6️⃣  CLEANUP: Rimuove cartella temporanea
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    if c3.button("Heatmap Azione"):
        #"""
        #HEATMAP - Densità di movimento per team
        
        #Visualizza quale area del campo viene più frequentemente occupata
        #da ogni squadra nel range di frame selezionato.
        
        #Usa KDE (Kernel Density Estimation) sui dati di posizione in METRI.
        #Output:
       # - Red: Gradiente rosso (intenso = area molto frequentata)
        #- White: Gradiente blu (intenso = area molto frequentata)
        #"""
        sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
        colors_map = {'Red':'Reds', 'White':'Blues'}  # Colormaps Seaborn
        
        for t in ['Red', 'White']:
            st_t = sub[sub['team']==t]
            if not st_t.empty:
                fig, ax = plt.subplots(figsize=(5,3))
                draw_mpl_court(ax)
                
                # KDE plot: densità 2D su coordinate in METRI
                sns.kdeplot(x=st_t['x_meters'], y=st_t['y_meters'], 
                           fill=True, cmap=colors_map.get(t, 'Greys'), 
                           alpha=0.6, ax=ax)
                
                ax.set_xlim(0, REAL_WIDTH_M)
                ax.set_ylim(REAL_HEIGHT_M, 0)
                ax.axis('off')
                ax.set_title(f"Heatmap {t} (Meters)")
                c3.pyplot(fig)
