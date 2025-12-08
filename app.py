import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, ConvexHull
from matplotlib.patches import Polygon, Rectangle, Circle
from itertools import combinations
import imageio.v2 as imageio
import shutil

# --- CONFIGURAZIONE ---
CSV_FILE = 'tracking_data.csv'
# Questa cartella serve solo se il path nel CSV fallisce
IMAGES_ROOT = 'datasets' 
RADAR_HEIGHT = 300
COURT_WIDTH = 3840 
COURT_HEIGHT = 2160
POSSESSION_THRESHOLD = 60 # Pixel

# Parametri Fisici (Solo per conversione finale)
FPS = 30.0
PX_TO_M = 28.0 / 3840.0 # ~0.0073 m/px

# --- FUNZIONI UTILI ---
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', str(filename))
    if match: return int(match.group(1))
    return 0

def draw_radar_court(img, width, height, color=(200, 200, 200)):
    thick = 2
    # Disegna in scala radar (600px = 28m)
    # Rapporto Pixel Radar su Pixel Video
    scale_x = width / COURT_WIDTH
    scale_y = height / COURT_HEIGHT
    
    # Campo base
    cv2.line(img, (int(width/2), 0), (int(width/2), height), color, thick)
    cv2.circle(img, (int(width/2), int(height/2)), int(180 * scale_x), color, thick)
    
    # Aree (proporzionate dai pixel video)
    pw = int(550 * scale_x); ph = int(500 * scale_y)
    yt = int(height/2 - ph/2); yb = int(height/2 + ph/2)
    
    cv2.rectangle(img, (0, yt), (pw, yb), color, thick)
    cv2.rectangle(img, (width-pw, yt), (width, yb), color, thick)
    
    return img

def draw_mpl_court(ax, color='black', lw=2):
    # Disegna in PIXEL (per coerenza totale)
    court = Rectangle((0, 0), COURT_WIDTH, COURT_HEIGHT, linewidth=lw, color=color, fill=False)
    ax.add_patch(court)
    ax.plot([COURT_WIDTH/2, COURT_WIDTH/2], [0, COURT_HEIGHT], color=color, linewidth=lw)
    ax.add_patch(Rectangle((0, COURT_HEIGHT/2 - 250), 550, 500, linewidth=lw, color=color, fill=False))
    ax.add_patch(Rectangle((COURT_WIDTH-550, COURT_HEIGHT/2 - 250), 550, 500, linewidth=lw, color=color, fill=False))

@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE): return None
    df = pd.read_csv(CSV_FILE)
    
    # Crea action_id se manca
    if 'action_id' not in df.columns:
        df['action_id'] = df['frame_filename'].apply(lambda x: x.split('_frame')[0] if '_frame' in x else 'unknown')
        
    df['frame_id'] = df['frame_filename'].apply(extract_frame_number)
    df['player_unique_id'] = df['team'] + "_" + df['number'].astype(str)
    return df

# --- FUNZIONE CRITICA: TROVA IMMAGINE (FIX CLOUD/WINDOWS) ---
def get_image_path(row):
    fname = row['frame_filename']
    
    # 1. Prova il path esatto del CSV (se esiste)
    if 'image_path' in row and isinstance(row['image_path'], str):
        path_csv = row['image_path']
        # Corregge slash per OS corrente (Windows \ <-> Linux /)
        path_norm = path_csv.replace('\\', os.sep).replace('/', os.sep)
        if os.path.exists(path_norm): return path_norm
    
    # 2. Ricostruisci percorso standard: datasets/azione_XX/train/file.jpg
    action = row['action_id'] if 'action_id' in row else 'azione_01'
    path_built = os.path.join('datasets', action, 'train', fname)
    if os.path.exists(path_built): return path_built
    
    # 3. Cerca in root o cartella train semplice
    path_simple = os.path.join('train', fname)
    if os.path.exists(path_simple): return path_simple
    
    return None

# --- CALCOLO STATISTICHE (Pixel Logic) ---
def calculate_stats_pixel(df_action, player_id, current_frame):
    p_data = df_action[(df_action['player_unique_id'] == player_id) & (df_action['frame_id'] <= current_frame)].sort_values('frame_id')
    if len(p_data) < 5: return 0, 0, 0
    
    # Distanza Pixel
    dx = p_data['x_feet'].diff().fillna(0)
    dy = p_data['y_feet'].diff().fillna(0)
    step_px = np.sqrt(dx**2 + dy**2)
    # Filtro anti-teletrasporto (100px)
    step_px[step_px > 100] = 0
    
    tot_px = step_px.sum()
    
    # Velocità (Media ultimi 15 frame)
    spd_kmh = 0.0
    if len(p_data) > 15:
        dist_last = step_px.tail(15).sum() # pixel
        dist_m = dist_last * PX_TO_M
        time_s = 15 / FPS
        spd_kmh = (dist_m / time_s) * 3.6
        if spd_kmh < 1.0: spd_kmh = 0.0 # Noise gate

    # Off-Ball (Semplificato per manuale: = tot)
    off_px = tot_px 
    
    return int(tot_px*PX_TO_M), int(off_px*PX_TO_M), round(spd_kmh, 1)

# --- RENDERING ---
def render_frame(f_id, df, quality, selected_p):
    row_df = df[df['frame_id'] == f_id]
    if row_df.empty: return None, None, 0, 0, 0
    
    # Recupera immagine in modo sicuro
    first_row = row_df.iloc[0]
    img_path = get_image_path(first_row)
    
    if img_path is None:
        # Immagine non trovata: Restituisci nero
        return np.zeros((720, 1280, 3), dtype=np.uint8), np.zeros((300, 600, 3), dtype=np.uint8), 0, 0, 0
        
    frame_img = cv2.imread(img_path)
    if frame_img is None: return None, None, 0, 0, 0

    # Resize
    scale = 1280 / frame_img.shape[1] if quality == "Ottimizzata (HD)" else 1.0
    frame_img = cv2.resize(frame_img, (0, 0), fx=scale, fy=scale)
    
    # Radar
    radar = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar, 600, RADAR_HEIGHT)
    
    # Scala Radar (Pixel -> Radar)
    sx = 600 / COURT_WIDTH
    sy = RADAR_HEIGHT / COURT_HEIGHT
    
    red_c=0; white_c=0; ref_c=0
    is_owner = False
    
    # Palla
    ball = row_df[row_df['team'] == 'Ball']
    bx_c, by_c = -1, -1
    if not ball.empty:
        b = ball.iloc[0]
        bx_c = b['bbox_x'] + b['bbox_w']/2
        by_c = b['bbox_y'] + b['bbox_h']/2
        # Radar Palla
        rx = int(b['x_feet']*sx); ry = int(b['y_feet']*sy)
        cv2.circle(radar, (rx, ry), 6, (0, 165, 255), -1)

    for _, r in row_df.iterrows():
        # Radar
        rx = int(r['x_feet']*sx); ry = int(r['y_feet']*sy)
        team = str(r['team'])
        
        col = (200, 200, 200)
        if team == 'Red': col = (0, 0, 255); red_c+=1
        elif team == 'White': col = (255, 255, 255); white_c+=1
        elif 'Ref' in team: col = (0, 255, 0); ref_c+=1
        
        if team != 'Ball':
            cv2.circle(radar, (rx, ry), 5, col, -1)
        
        # Highlight e Possesso (Box-in-Box)
        if r['player_unique_id'] == selected_p:
            bx = int(r['bbox_x']*scale); by = int(r['bbox_y']*scale)
            bw = int(r['bbox_w']*scale); bh = int(r['bbox_h']*scale)
            
            # Possesso Logic
            has_ball = False
            if bx_c != -1: # Se c'è la palla
                px1 = r['bbox_x']; px2 = px1 + r['bbox_w']
                py1 = r['bbox_y']; py2 = py1 + r['bbox_h']
                # Margine 5%
                mx = r['bbox_w']*0.05
                if (px1+mx < bx_c < px2-mx) and (py1 < by_c < py2):
                    has_ball = True
                    is_owner = True
            
            c_box = (0, 165, 255) if has_ball else (0, 255, 255)
            cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), c_box, 3)
            cv2.circle(radar, (rx, ry), 10, c_box, 2)
            
    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(radar, cv2.COLOR_BGR2RGB), red_c, white_c, ref_c, is_owner

# --- MAIN APP ---
st.set_page_config(page_title="CourtSense", layout="wide")
st.title("🏀 CourtSense: Tactical Dashboard")

df_full = load_data()
if df_full is None: st.error("CSV Tracking non trovato."); st.stop()

# 1. Selettore Azione (Fondamentale)
available_actions = sorted(df_full['action_id'].unique())
st.sidebar.header("⚙️ Configurazione")
selected_action = st.sidebar.selectbox("📂 Azione:", available_actions)

# Filtro i dati
df = df_full[df_full['action_id'] == selected_action].sort_values('frame_id')
unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))

# 2. Selettore Giocatore
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])
selected_player = st.sidebar.selectbox("👤 Giocatore:", player_list)

mode = st.sidebar.radio("Modalità:", ("Navigazione", "Riproduzione"))
quality = st.sidebar.radio("Qualità:", ("Ottimizzata (HD)", "Massima (4K)"))

col1, col2 = st.columns([3, 1])
ph_vid = col1.empty()
ph_rad = col2.empty()
ph_stat = col2.empty()

def update_ui(fid, d_tot, d_off, poss_s, spd):
    vid, rad, rc, wc, ref, hold = render_frame(fid, df, quality, selected_player)
    if vid is not None:
        ph_vid.image(vid, use_container_width=True)
        ph_rad.image(rad, caption="Radar Tattico", use_container_width=True)
        icon = "🏀" if hold else ""
        ph_stat.markdown(f"""
        ### Frame: {fid}
        **Player:** `{selected_player}` {icon}
        📏 **Dist:** {d_tot} m
        🏃 **Off-Ball:** {d_off} m
        ⚡ **Speed:** {spd} km/h
        ⏱️ **Poss:** {poss_s}
        """)
    return hold

if mode == "Navigazione":
    sel_f = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    d, off, s = calculate_stats_pixel(df, selected_player, sel_f)
    update_ui(sel_f, d, off, "-", s)
    
    st.markdown("---"); c1, c2 = st.columns(2)
    fdata = df[df['frame_id'] == sel_f]
    # Pulsanti statici (Voronoi/Hull) qui se vuoi rimetterli
    
else: # Riproduzione
    start, end = st.sidebar.select_slider("Clip:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    if st.sidebar.button("▶️ PLAY"):
        frames = [f for f in unique_frames if start <= f <= end]
        cum_px = 0; cum_off_px = 0; poss_count = 0; prev_x = None; prev_y = None
        pos_buffer = []
        
        # Pre-Load Palla
        ball_df = df[df['team'] == 'Ball']
        
        for f_id in frames:
            t0 = time.time()
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            ball_curr = ball_df[ball_df['frame_id'] == f_id]
            step = 0; spd = 0.0
            
            if not curr.empty:
                cx, cy = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buffer.append((cx, cy)); 
                if len(pos_buffer)>15: pos_buffer.pop(0)
                
                if prev_x:
                    raw_s = np.sqrt((cx-prev_x)**2 + (cy-prev_y)**2)
                    if raw_s < 100: step = raw_s; cum_px += step
                prev_x, prev_y = cx, cy
                
                # Speed (15 frame)
                if len(pos_buffer)>=15:
                    dist_buf = np.sqrt((pos_buffer[-1][0]-pos_buffer[0][0])**2 + (pos_buffer[-1][1]-pos_buffer[0][1])**2)
                    m_buf = dist_buf * PX_TO_M
                    spd = (m_buf / (14/FPS)) * 3.6

            # Possesso Live (Box-in-Box + Buffer 3 frame manuale)
            is_hold = False
            # Per semplicità nel live usiamo il flag ritornato dal render (istantaneo) o un buffer locale
            # Qui usiamo il flag del render per coerenza visiva immediata
            is_hold = update_ui(f_id, int(cum_px*PX_TO_M), int(cum_off_px*PX_TO_M), f"{(poss_count/FPS):.1f} s", round(spd, 1))
            
            if is_hold: poss_count += 1
            else: cum_off_px += step
            
            time.sleep(max(0.02, (1.0/FPS) - (time.time()-t0)))

# --- REPORT (MANTENUTO UGUALE) ---
st.markdown("---"); st.subheader("📈 Report")
if st.button("Genera Metriche"):
    with st.spinner("Calcolo..."):
        sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
        players = sub[sub['team'].isin(['Red', 'White'])]
        ball_sub = sub[sub['team']=='Ball']
        
        moves = []
        m = pd.merge(players, ball_sub[['frame_id','bbox_x','bbox_y','bbox_w','bbox_h']], on='frame_id', suffixes=('','_b'), how='left')
        
        for pid, g in m.groupby('player_unique_id'):
            g = g.sort_values('frame_id')
            steps = np.sqrt(np.diff(g['x_feet'], prepend=g['x_feet'].iloc[0])**2 + np.diff(g['y_feet'], prepend=g['y_feet'].iloc[0])**2)
            steps = np.where(steps > 100, 0, steps) # Filtro px
            
            # Possesso Box
            bx = g['bbox_x_b']+g['bbox_w_b']/2; by = g['bbox_y_b']+g['bbox_h_b']/2
            mw = g['bbox_w']*0.05; mh = g['bbox_h']*0.10
            is_in = (bx > g['bbox_x']+mw) & (bx < g['bbox_x']+g['bbox_w']-mw) & (by > g['bbox_y']) & (by < g['bbox_y']+g['bbox_h']-mh)
            is_poss = is_in.fillna(False).astype(int).rolling(3, center=True).sum() >= 2
            
            tot_m = steps.sum() * PX_TO_M
            off_m = steps[~is_poss].sum() * PX_TO_M
            poss_s = is_poss.sum() / FPS
            spd_avg = (tot_m / ((end-start)/FPS)) * 3.6
            
            moves.append({'Player':pid, 'Team':g['team'].iloc[0], 'Tot':tot_m, 'Off':off_m, 'Poss':poss_s, 'Spd':spd_avg})
            
        res = pd.DataFrame(moves)
        
        # KPI e Grafici
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(); sns.barplot(data=res, x='Player', y='Tot', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax)
        ax.set_title("Workload (m)"); ax.tick_params(axis='x', rotation=90)
        c1.pyplot(fig)
        
        fig2, ax2 = plt.subplots(); sns.barplot(data=res, x='Player', y='Poss', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax2)
        ax2.set_title("Possession (s)"); ax2.tick_params(axis='x', rotation=90)
        c2.pyplot(fig2)