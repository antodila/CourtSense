import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, ConvexHull
from matplotlib.patches import Polygon, Rectangle, Circle, Arc
from itertools import combinations
import imageio.v2 as imageio
import shutil
from scipy.signal import savgol_filter
import plotly.express as px

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE BASE
# ═══════════════════════════════════════════════════════════════════════════
CSV_FILE = 'tracking_data.csv'
IMAGES_FOLDER = 'datasets' 
RADAR_HEIGHT = 300
COURT_WIDTH = 3840 
COURT_HEIGHT = 2160
POSSESSION_THRESHOLD = 60 

# --- FISICA ---
FPS = 12.0
REAL_WIDTH_M = 28.0
REAL_HEIGHT_M = 15.0
PX_TO_M = REAL_WIDTH_M / COURT_WIDTH 

# PHYSICS_FPS: Frame scattati a 12 FPS
PHYSICS_FPS = 12.0 

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE OMOGRAFIA
# ═══════════════════════════════════════════════════════════════════════════

PTS_REAL_METERS = np.float32([
    [0, 0],              # Top-Left
    [28.0, 0],           # Top-Right
    [28.0, 15.0],        # Bottom-Right
    [0, 15.0]            # Bottom-Left
])

DATASET_CONFIG = {
    'azione_01': np.float32([[857, 980], [2849, 1011], [3552, 1505], [149, 1458]]),
    'azione_02': np.float32([[857, 980], [2849, 1011], [3552, 1505], [149, 1458]]),
    'azione_03': np.float32([[143, 290], [474, 299], [591, 446], [24, 432]]),
    'azione_04': np.float32([[143, 290], [474, 299], [591, 446], [24, 432]]),
    'azione_05': np.float32([[143, 290], [474, 299], [591, 446], [24, 432]]),
}

def apply_perspective_transform(points_px, src_pts, dst_pts):
    if len(points_px) == 0: return points_px
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    points_reshaped = points_px.reshape(-1, 1, 2).astype(np.float32)
    points_transformed = cv2.perspectiveTransform(points_reshaped, H)
    return points_transformed.reshape(-1, 2)

# ═══════════════════════════════════════════════════════════════════════════
# PARSING & LOADING
# ═══════════════════════════════════════════════════════════════════════════
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', str(filename))
    if match: return int(match.group(1))
    return 0

@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE): return None
    df = pd.read_csv(CSV_FILE)
    
    if 'action_id' not in df.columns:
        df['action_id'] = df['frame_filename'].apply(lambda x: x.split('_frame')[0] if '_frame' in x else 'unknown')
    df['frame_id'] = df['frame_filename'].apply(extract_frame_number)
    df['player_unique_id'] = df['team'] + "_" + df['number'].astype(str)
    
    # --- CALCOLO METRI (RETTIFICA) ---
    df['x_meters'] = 0.0; df['y_meters'] = 0.0
    
    for action in df['action_id'].unique():
        mask = df['action_id'] == action
        if action in DATASET_CONFIG:
            points_px = df.loc[mask, ['x_feet', 'y_feet']].values
            points_m = apply_perspective_transform(points_px, DATASET_CONFIG[action], PTS_REAL_METERS)
            df.loc[mask, 'x_meters'] = points_m[:, 0]
            df.loc[mask, 'y_meters'] = points_m[:, 1]
        else:
            df.loc[mask, 'x_meters'] = df.loc[mask, 'x_feet'] * PX_TO_M
            df.loc[mask, 'y_meters'] = df.loc[mask, 'y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# LOGICA POSSESSO
# ═══════════════════════════════════════════════════════════════════════════
def get_possession_table(df_subset):
    ball_df = df_subset[df_subset['team'] == 'Ball'][['frame_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
    players_df = df_subset[df_subset['team'].isin(['Red', 'White'])].copy()
    if ball_df.empty or players_df.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    m = pd.merge(players_df, ball_df, on='frame_id', suffixes=('', '_b'), how='inner')
    mw, mh = m['bbox_w']*0.05, m['bbox_h']*0.10
    bx_c = m['bbox_x_b'] + m['bbox_w_b']/2; by_c = m['bbox_y_b'] + m['bbox_h_b']/2
    
    is_ov = (bx_c > m['bbox_x']+mw) & (bx_c < m['bbox_x']+m['bbox_w']-mw) & \
            (by_c > m['bbox_y']) & (by_c < m['bbox_y']+m['bbox_h']-mh)
    
    candidates = m[is_ov].copy()
    if candidates.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    p_cx = candidates['bbox_x'] + candidates['bbox_w']/2
    p_cy = candidates['bbox_y'] + candidates['bbox_h']/2
    candidates['dist_px'] = np.sqrt((p_cx - bx_c)**2 + (p_cy - by_c)**2)
    best_idx = candidates.groupby('frame_id')['dist_px'].idxmin()
    return candidates.loc[best_idx, ['frame_id', 'player_unique_id']].set_index('frame_id')

# ═══════════════════════════════════════════════════════════════════════════
# GRAFICA & RENDERING
# ═══════════════════════════════════════════════════════════════════════════
def draw_radar_court(img, width, height, color=(200, 200, 200)):
    """
    Disegna un campo da basket FIBA realistico sul radar.
    """
    thick = 2
    # Scala: pixel per metro
    ppm_x = width / 28.0
    ppm_y = height / 15.0
    
    # Colore linee
    c = color
    
    # 1. Campo Esterno
    # (Già disegnato dal bordo immagine o non necessario se nero)
    
    # 2. Linea di Metà Campo e Cerchio Centrale
    mid_x = int(width / 2)
    mid_y = int(height / 2)
    cv2.line(img, (mid_x, 0), (mid_x, height), c, thick)
    cv2.circle(img, (mid_x, mid_y), int(1.8 * ppm_x), c, thick)
    
    # 3. Aree (The Paint) - Rettangoli
    # FIBA: 5.8m x 4.9m
    paint_w = int(5.8 * ppm_x)
    paint_h = int(4.9 * ppm_y)
    paint_top = int((height - paint_h) / 2)
    paint_bot = int((height + paint_h) / 2)
    
    # Sinistra
    cv2.rectangle(img, (0, paint_top), (paint_w, paint_bot), c, thick)
    # Lunetta Sinistra (Semicerchio)
    cv2.ellipse(img, (paint_w, mid_y), (int(1.8*ppm_x), int(1.8*ppm_y)), 0, -90, 90, c, thick)
    
    # Destra
    cv2.rectangle(img, (width - paint_w, paint_top), (width, paint_bot), c, thick)
    # Lunetta Destra (Semicerchio)
    cv2.ellipse(img, (width - paint_w, mid_y), (int(1.8*ppm_x), int(1.8*ppm_y)), 0, 90, 270, c, thick)
    
    # 4. Linea da 3 Punti (Arco)
    # FIBA: Raggio 6.75m
    three_r_x = int(6.75 * ppm_x)
    three_r_y = int(6.75 * ppm_y)
    
    # Centro del canestro (offset di 1.575m dal fondo)
    hoop_offset = int(1.575 * ppm_x)
    
    # Arco Sinistro
    cv2.ellipse(img, (hoop_offset, mid_y), (three_r_x, three_r_y), 0, -90, 90, c, thick)
    # Arco Destro
    cv2.ellipse(img, (width - hoop_offset, mid_y), (three_r_x, three_r_y), 0, 90, 270, c, thick)
    
    # 5. Canestri (Cerchietti colorati per orientamento)
    hoop_col = (0, 165, 255) # Arancione Basket
    cv2.circle(img, (hoop_offset, mid_y), 4, hoop_col, -1)
    cv2.circle(img, (width - hoop_offset, mid_y), 4, hoop_col, -1)
    
    return img

def draw_mpl_court(ax, color='black', lw=2):
    """Disegna un campo da basket FIBA realistico su Matplotlib."""
    # 1. Perimetro e linea centrale
    court = Rectangle((0, 0), REAL_WIDTH_M, REAL_HEIGHT_M, linewidth=lw, color=color, fill=False)
    ax.add_patch(court)
    ax.plot([14, 14], [0, 15], color=color, linewidth=lw) # Metà campo
    ax.add_patch(Circle((14, 7.5), 1.8, color=color, fill=False, linewidth=lw)) # Cerchio centrale
    
    # 2. Aree (The Paint) - 5.8m x 4.9m
    # Sinistra
    ax.add_patch(Rectangle((0, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))
    # Destra
    ax.add_patch(Rectangle((22.2, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))
    
    # 3. Lunette (Free Throw Circles) - Raggio 1.8m
    # Sinistra
    ax.add_patch(Arc((5.8, 7.5), 3.6, 3.6, theta1=-90, theta2=90, color=color, linewidth=lw))
    # Destra
    ax.add_patch(Arc((22.2, 7.5), 3.6, 3.6, theta1=90, theta2=270, color=color, linewidth=lw))
    
    # 4. Linea da 3 Punti (Arco) - Raggio 6.75m
    # Sinistra
    ax.add_patch(Arc((1.575, 7.5), 13.5, 13.5, theta1=-90, theta2=90, color=color, linewidth=lw))
    # Destra
    ax.add_patch(Arc((26.425, 7.5), 13.5, 13.5, theta1=90, theta2=270, color=color, linewidth=lw))
    
    # 5. Canestri (Cerchietti arancioni)
    ax.add_patch(Circle((1.575, 7.5), 0.25, color='orange', fill=True))
    ax.add_patch(Circle((26.425, 7.5), 0.25, color='orange', fill=True))

def generate_static_hull(frame_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax) # Disegna il campo da basket
    
    colors = {'Red': 'red', 'White': 'blue'}
    fill_colors = {'Red': 'salmon', 'White': 'lightblue'}
    
    for team in ['Red', 'White']:
        points = frame_data[frame_data['team'] == team][['x_meters', 'y_meters']].values
        ax.scatter(points[:,0], points[:,1], c=colors[team], s=80, edgecolors='white', zorder=5)
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                # Disegna poligono colorato
                poly = Polygon(points[hull.vertices], facecolor=fill_colors[team], edgecolor=colors[team], alpha=0.3, lw=2)
                ax.add_patch(poly)
            except: pass
            
    ax.set_xlim(0, REAL_WIDTH_M)
    ax.set_ylim(REAL_HEIGHT_M, 0) # Inverti Y per coerenza col video
    ax.axis('off')
    return fig

def generate_static_voronoi(frame_data, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Usa la nuova funzione realistica
    draw_mpl_court(ax) 
    
    players = frame_data[frame_data['team'].isin(['Red', 'White'])]
    
    if len(players) >= 4:
        points = players[['x_meters', 'y_meters']].values
        teams = players['team'].values
        # Punti fittizi per chiudere le regioni Voronoi
        dummy = np.array([[-10, -10], [40, -10], [40, 25], [-10, 25]])
        
        try:
            vor = Voronoi(np.vstack([points, dummy]))
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region:
                    polygon = [vor.vertices[i] for i in region]
                    c = 'red' if teams[i] == 'Red' else 'blue'
                    ax.add_patch(Polygon(polygon, facecolor=c, alpha=0.3, edgecolor='white'))
        except: pass
        
    # Disegna giocatori
    for _, r in players.iterrows():
        c = 'red' if r['team']=='Red' else 'blue'
        ax.scatter(r['x_meters'], r['y_meters'], c=c, s=80, edgecolors='white', zorder=10)
        
    # Palla
    ball = frame_data[frame_data['team']=='Ball']
    if not ball.empty:
        ax.scatter(ball['x_meters'], ball['y_meters'], c='orange', s=150, edgecolors='black', marker='o', zorder=15)
        
    ax.set_xlim(0, REAL_WIDTH_M)
    ax.set_ylim(REAL_HEIGHT_M, 0)
    ax.axis('off')
    if title: ax.set_title(title, fontweight='bold')
    return fig

def render_nba_style(f_id, df, target_width, highlight_id=None, is_possessor=False, stats=None):
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None
    fname = fname_row['frame_filename'].iloc[0]
    
    # Path Logic
    img_path = None
    if 'image_path' in fname_row.columns:
        p = fname_row['image_path'].iloc[0]
        if os.path.exists(p): img_path = p
        elif os.path.exists(p.replace('\\', '/')): img_path = p.replace('\\', '/')
    if img_path is None: 
        if 'action_id' in fname_row.columns:
            act = fname_row['action_id'].iloc[0]
            chk = os.path.join('datasets', act, 'train', fname)
            if os.path.exists(chk): img_path = chk
    if img_path is None: img_path = os.path.join(IMAGES_FOLDER, fname)
    if not os.path.exists(img_path): img_path = os.path.join('train', fname)
        
    frame_img_orig = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None
    if frame_img_orig is None: 
        frame_img_orig = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame_img_orig, "IMG NOT FOUND", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    scale = target_width / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    H, W = frame_img.shape[:2]
    
    radar_base = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_base, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    
    # --- LOGICA PALLA "SNAP-TO-PLAYER" ---
    ball_row = frame_data[frame_data['team'] == 'Ball']
    player_positions = {} # Salva posizioni per controllo snap
    
    # 1. Disegna Giocatori e Bounding Box
    for _, row in frame_data.iterrows():
        t = str(row['team'])
        if t == 'Ball': continue # Salta palla nel primo giro

        rx = int(row['x_meters'] * sx); ry = int(row['y_meters'] * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        
        player_positions[row['player_unique_id']] = (rx, ry, row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h'])
        
        col = (200,200,200)
        if t=='Red': col=(0,0,255)
        elif t=='White': col=(255,255,255)
        
        cv2.circle(radar_base, (rx, ry), 5, col, -1)
        
        if row['player_unique_id'] == highlight_id:
            bx = int(row['bbox_x']*scale); by = int(row['bbox_y']*scale)
            bw = int(row['bbox_w']*scale); bh = int(row['bbox_h']*scale)
            col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
            cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 3)
            cv2.circle(radar_base, (rx, ry), 9, col_box, 2)

    # 2. Disegna Palla (Snap o Proiezione)
    if not ball_row.empty:
        b = ball_row.iloc[0]
        # Check possesso visivo
        owner_id = None
        bx_c, by_c = b['bbox_x']+b['bbox_w']/2, b['bbox_y']+b['bbox_h']/2
        
        for pid, vals in player_positions.items():
            prx, pry, pbx, pby, pbw, pbh = vals
            # Logica Box-in-Box semplificata
            mw, mh = pbw*0.05, pbh*0.10
            if (pbx+mw < bx_c < pbx+pbw-mw) and (pby < by_c < pby+pbh-mh):
                owner_id = pid
                break
        
        if owner_id:
            # SNAP: Usa posizione giocatore sul radar (+ piccolo offset)
            prx, pry = player_positions[owner_id][:2]
            final_bx, final_by = prx + 3, pry + 3
        else:
            # VOLO: Usa proiezione omografica
            final_bx = int(b['x_meters'] * sx)
            final_by = int(b['y_meters'] * sy)
            
        final_bx = max(0, min(final_bx, 600-1))
        final_by = max(0, min(final_by, RADAR_HEIGHT-1))
        cv2.circle(radar_base, (final_bx, final_by), 6, (0, 165, 255), -1)

    # Overlay Radar su Frame
    mini_w = int(W * 0.25); mini_h = int(mini_w * (RADAR_HEIGHT/600))
    radar_mini = cv2.resize(radar_base, (mini_w, mini_h))
    y1 = H - mini_h - 20; x1 = W - mini_w - 20
    frame_img[y1:y1+mini_h, x1:x1+mini_w] = cv2.addWeighted(frame_img[y1:y1+mini_h, x1:x1+mini_w], 0.3, radar_mini, 0.7, 0)
    cv2.rectangle(frame_img, (x1, y1), (x1+mini_w, y1+mini_h), (255,255,255), 1)

    # Box Statistiche
    if stats:
        dist, off, spd, poss = stats
        cv2.rectangle(frame_img, (20, 20), (350, 130), (0,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_img, f"{highlight_id}{' (BALL)' if is_possessor else ''}", (30, 50), font, 0.6, (0,255,255), 2)
        
        d_str = f"{dist} m" if isinstance(dist, (int, float)) else str(dist)
        cv2.putText(frame_img, f"DIST: {d_str}", (30, 75), font, 0.5, (255,255,255), 1)
        
        # Mostra velocità solo se ha un valore numerico valido (no "-" e no 0 fisso inutile)
        if spd != "-" and spd != 0:
             s_str = f"{spd} m/s"
             cv2.putText(frame_img, f"SPEED: {s_str}", (30, 95), font, 0.5, (255,255,255), 1)
             
        cv2.putText(frame_img, f"POSS: {poss}", (30, 115), font, 0.5, (255,255,255), 1)

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

def calculate_stats_dummy(df_action, player_id, current_frame, ownership_table):
    # Dummy per UI
    return 0, 0, "-", 0

@st.cache_data(show_spinner=False)
def get_cached_voronoi_gif(df_subset, file_prefix="voronoi"):
    """
    Genera la GIF Voronoi solo se i dati cambiano.
    """
    frames_list = df_subset['frame_filename'].unique()
    if len(frames_list) == 0:
        return None

    tmp_dir = f"tmp_{file_prefix}"
    gif_path = f"{file_prefix}_output.gif"
    
    # Creazione cartella temp
    os.makedirs(tmp_dir, exist_ok=True)
    files = []
    
    try:
        # Generazione frame (Matplotlib)
        for i, fn in enumerate(frames_list):
            # Nota: qui chiamiamo la tua funzione esistente
            frame_num = extract_frame_number(fn)
            fig = generate_static_voronoi(
                df_subset[df_subset['frame_filename'] == fn], 
                title=f"Tactical Space - Frame {frame_num}"
            )
            p = os.path.join(tmp_dir, f"{i:03d}.png")
            fig.savefig(p, dpi=80, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig) # Chiude la figura per liberare memoria
            files.append(p)
        
        # Creazione GIF
        with imageio.get_writer(gif_path, mode='I', duration=0.15, loop=0) as w:
            for f in files:
                w.append_data(imageio.imread(f))
                
        return gif_path

    except Exception as e:
        return None
    finally:
        # Pulizia file temporanei
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CourtSense Cloud", layout="wide")
st.title("🏀 CourtSense: Broadcast Analytics")

df_full = load_data()
if df_full is None: st.error("CSV 'tracking_data.csv' non trovato."); st.stop()

st.sidebar.header("Configurazione")
available_actions = sorted(df_full['action_id'].unique())
idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("Azione:", available_actions, index=idx)

df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
own_table = get_possession_table(df)

frames = df['frame_id'].unique()
min_f, max_f = int(min(frames)), int(max(frames))
players = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

mode = st.sidebar.radio("Modalità:", ("🕹️ Navigazione (Manuale)", "▶️ Genera Video (Auto)"))
sel_player = st.sidebar.selectbox("Giocatore:", players)
preview_ph = st.empty()

# --- MODALITÀ MANUALE ---
if mode == "🕹️ Navigazione (Manuale)":
    f = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    
    is_own = False
    try:
        if f in own_table.index and own_table.loc[f]['player_unique_id'] == sel_player: is_own = True
    except: pass
    
    # Render statico per preview veloce
    img = render_nba_style(f, df, 1920, sel_player, is_own, ("-", "-", "-", "-"))
    if img is not None: preview_ph.image(img, channels="RGB", width="stretch")
    
    c1, c2 = st.columns(2)
    frm_data = df[df['frame_id']==f]
    if c1.button("📸 Voronoi"): c1.pyplot(generate_static_voronoi(frm_data))
    if c2.button("🛡️ Convex Hull"): c2.pyplot(generate_static_hull(frm_data))

# --- MODALITÀ VIDEO (BATCH) ---
else: 
    st.info("Genera un video MP4 fluido (1080p).")
    start, end = st.sidebar.select_slider("Clip Range:", options=frames, value=(min_f, min(min_f+40, max_f)))
    
    if st.sidebar.button("🎥 GENERA VIDEO"):
        clip = [x for x in frames if start <= x <= end]
        output_file = "analysis_output.mp4"
        prog_bar = st.progress(0, "Inizializzazione...")
        
        cum_m = 0; cum_off = 0; poss_c = 0
        last_macro_pos = None; frame_counter = 0; MACRO_INTERVAL = 6 
        
        video_frames = []
        
        for i, f in enumerate(clip):
            prog_bar.progress(int(i / len(clip) * 100))
            
            curr = df[(df['frame_id']==f) & (df['player_unique_id']==sel_player)]
            step_val = 0
            
            if not curr.empty:
                # Coordinate METRICHE
                cx, cy = curr.iloc[0]['x_meters'], curr.iloc[0]['y_meters']
                curr_pos_arr = np.array([cx, cy])
                
                if last_macro_pos is None: last_macro_pos = curr_pos_arr
                
                # Calcolo Distanza (Campionata)
                frame_counter += 1
                if frame_counter % MACRO_INTERVAL == 0:
                    dist_segment = np.linalg.norm(curr_pos_arr - last_macro_pos)
                    # Filtro fermo (10cm in 0.2s)
                    if dist_segment > 0.10:
                        cum_m += dist_segment
                        step_val = dist_segment 
                    last_macro_pos = curr_pos_arr

            # Possesso
            is_own = False
            try:
                if f in own_table.index and own_table.loc[f]['player_unique_id'] == sel_player: is_own = True
            except: pass
            
            if is_own: poss_c += 1
            else: 
                if step_val > 0: cum_off += step_val
            
            # Render con "-" sulla velocità
            img = render_nba_style(f, df, 1280, sel_player, is_own, (int(cum_m), int(cum_off), "-", f"{(poss_c/FPS):.1f}s"))
            if img is not None: video_frames.append(img)
            
        prog_bar.progress(95, "Compilazione MP4...")
        imageio.mimwrite(output_file, video_frames, fps=FPS, macro_block_size=1)
        
        prog_bar.empty()
        st.success("Video Generato!")
        st.video(output_file)

    # --- REPORT FINALE ---
    st.markdown("---"); st.subheader("📈 Report")

    if 'metrics_active' not in st.session_state:
        st.session_state.metrics_active = False

    if st.button("Genera Metriche") or st.session_state.metrics_active:
        st.session_state.metrics_active = True
        
        with st.spinner("Calcolo metriche coerenti..."):
            sub = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            own_sub = own_table[own_table.index.isin(sub['frame_id'].unique())]
            duration_s = (end - start + 1) / PHYSICS_FPS 
            
            # 1. Spacing (Invariato)
            spac = []
            for f, g in players.groupby('frame_id'):
                for t in ['Red', 'White']:
                    tg = g[g['team'] == t]
                    if len(tg) >= 2:
                        dists_m = [np.linalg.norm(a-b) for a,b in combinations(tg[['x_meters','y_meters']].values, 2)]
                        spac.append({'f': f, 't': t, 'v': np.mean(dists_m)})
            
            if spac:
                sdf = pd.DataFrame(spac)
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                sns.lineplot(data=sdf, x='f', y='v', hue='t', palette={'Red':'red','White':'blue'}, ax=ax1)
                mr = sdf[sdf['t']=='Red']['v'].mean(); mw = sdf[sdf['t']=='White']['v'].mean()
                ax1.axhline(mr, c='darkred', ls='--', label=f"R:{mr:.1f}m"); ax1.axhline(mw, c='darkblue', ls='--', label=f"W:{mw:.1f}m")
                ax1.set_title("Avg Team Spacing (Meters)", fontweight='bold'); ax1.legend(); st.pyplot(fig1)

            # 2. Stats & Workload (LOGICA UNICA: MACRO STEP)
            moves = []; speed_poss_data = []
            
            # Calcoliamo PRIMA tutti i dati singoli con la STESSA logica
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                
                # Logica Macro-Step / Savgol coerente per tutti
                try:
                    xm = savgol_filter(g['x_meters'], 15, 2)
                    ym = savgol_filter(g['y_meters'], 15, 2)
                except:
                    xm = g['x_meters'].values; ym = g['y_meters'].values
                
                dx = np.diff(xm, prepend=xm[0])
                dy = np.diff(ym, prepend=ym[0])
                dists = np.sqrt(dx**2 + dy**2)
                dists[dists < 0.02] = 0 # Noise gate
                
                tot_m = np.sum(dists)
                # QUESTA è la velocità ufficiale per tutti: Distanza Totale / Tempo Totale
                avg_spd_ms = (tot_m / duration_s) if duration_s > 0 else 0
                
                is_poss = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                poss_s = is_poss.sum() / PHYSICS_FPS
                poss_ratio = poss_s / duration_s if duration_s > 0 else 0
                off_m = tot_m * (1.0 - poss_ratio)
                
                moves.append({'Player': pid, 'Dist': tot_m, 'Type': 'Total', 'Team': g['team'].iloc[0]})
                moves.append({'Player': pid, 'Dist': off_m, 'Type': 'Off-Ball', 'Team': g['team'].iloc[0]})
                speed_poss_data.append({'Player': pid, 'Team': g['team'].iloc[0], 'Speed': avg_spd_ms, 'Poss': poss_s})
            
            if moves:
                mdf = pd.DataFrame(moves); spdf = pd.DataFrame(speed_poss_data)
                
                # FILTRO COERENZA: Prendiamo SOLO i giocatori con velocità > 1.0 m/s
                real_players_red = spdf[(spdf['Team']=='Red') & (spdf['Speed'] > 1.0)]
                real_players_white = spdf[(spdf['Team']=='White') & (spdf['Speed'] > 1.0)]
                
                # --- CORREZIONE ERRORE KEYERROR 'Dist' ---
                # Calcoliamo le medie Distanza usando 'mdf', filtrando per i giocatori validi identificati sopra
                
                # RED TEAM
                if not real_players_red.empty:
                    valid_red_ids = real_players_red['Player'].unique()
                    atr = mdf[(mdf['Player'].isin(valid_red_ids)) & (mdf['Type']=='Total')]['Dist'].mean()
                    aro = mdf[(mdf['Player'].isin(valid_red_ids)) & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                    asr = real_players_red['Speed'].mean()
                    apr = real_players_red['Poss'].mean()
                else:
                    atr = 0; aro = 0; asr = 0; apr = 0

                # WHITE TEAM
                if not real_players_white.empty:
                    valid_white_ids = real_players_white['Player'].unique()
                    awt = mdf[(mdf['Player'].isin(valid_white_ids)) & (mdf['Type']=='Total')]['Dist'].mean()
                    awo = mdf[(mdf['Player'].isin(valid_white_ids)) & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                    asw = real_players_white['Speed'].mean()
                    apw = real_players_white['Poss'].mean()
                else:
                    awt = 0; awo = 0; asw = 0; apw = 0
                
                k1, k2 = st.columns(2)
                k1.info(f"🔴 **Red Avg**: Dist **{atr:.1f}m** (Off: {aro:.1f}m), Speed **{asr:.2f} m/s**, Poss **{apr:.1f}s**")
                k2.info(f"⚪ **White Avg**: Dist **{awt:.1f}m** (Off: {awo:.1f}m), Speed **{asw:.2f} m/s**, Poss **{apw:.1f}s**")
                
                # --- GRAFICI BARRE CON LINEE MEDIE (AGGIORNATO) ---
                valid_players = pd.concat([real_players_red, real_players_white])['Player'].unique()
                mdf_clean = mdf[mdf['Player'].isin(valid_players)]
                spdf_clean = spdf[spdf['Player'].isin(valid_players)]

                col_g1, col_g2 = st.columns(2)
                
                # GRAFICO WORKLOAD (con linee medie aggiunte)
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=mdf_clean, x='Player', y='Dist', hue='Type', palette={'Total':'gray', 'Off-Ball':'limegreen'}, ax=ax2)
                
                # LINEE MEDIE AGGIUNTE QUI
                if atr > 0: ax2.axhline(atr, c='darkred', ls='--', label=f"Avg R: {atr:.1f}m")
                if awt > 0: ax2.axhline(awt, c='darkblue', ls='--', label=f"Avg W: {awt:.1f}m")
                
                ax2.set_title("Workload (Meters)"); ax2.tick_params(axis='x', rotation=90)
                ax2.legend(fontsize='small', loc='upper right') # Mostra legenda linee
                col_g1.pyplot(fig2)
                
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=spdf_clean, x='Player', y='Poss', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax3)
                ax3.set_title("Possession Time (s)"); ax3.tick_params(axis='x', rotation=90); col_g2.pyplot(fig3)

            # --- 3. ANALISI DINAMICA VELOCITÀ (COERENTE) ---
            st.markdown("---")
            st.markdown("### ⚡ Analisi Dinamica Velocità (Speed vs Time)")
            
            # Usiamo la lista pulita dei giocatori
            clean_players_list = sorted(pd.concat([real_players_red, real_players_white])['Player'].unique())
            target_player = st.selectbox("Seleziona Giocatore:", clean_players_list)

            if target_player:
                p_data = sub[sub['player_unique_id'] == target_player].sort_values('frame_id').copy()
                # Ricalcolo identico a sopra per il grafico
                try:
                    xm = savgol_filter(p_data['x_meters'], 15, 2)
                    ym = savgol_filter(p_data['y_meters'], 15, 2)
                except:
                    xm = p_data['x_meters'].values; ym = p_data['y_meters'].values

                dx = np.diff(xm, prepend=xm[0])
                dy = np.diff(ym, prepend=ym[0])
                dist_per_frame = np.sqrt(dx**2 + dy**2)
                raw_speed = dist_per_frame * PHYSICS_FPS 
                
                # Rolling per la visualizzazione (solo estetica, il valore medio numerico usiamo quello Calcolato prima)
                speed_series = pd.Series(raw_speed)
                smooth_speed = speed_series.rolling(window=12, min_periods=1, center=True).mean()
                smooth_speed[smooth_speed < 0.2] = 0
                p_data['speed_m_s'] = smooth_speed.to_numpy()
                
                fig_speed = px.line(p_data, x='frame_id', y='speed_m_s', title=f"Velocità: {target_player}", labels={'speed_m_s':'m/s'}, template="plotly_dark")
                fig_speed.add_hrect(y0=0, y1=2, line_width=0, fillcolor="green", opacity=0.2, annotation_text="Walk")
                fig_speed.add_hrect(y0=2, y1=4.5, line_width=0, fillcolor="yellow", opacity=0.2, annotation_text="Jog")
                fig_speed.add_hrect(y0=4.5, y1=10, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Sprint")
                st.plotly_chart(fig_speed, width="stretch")
                
                # RECUPERA IL VALORE UFFICIALE CALCOLATO NEL LOOP PRECEDENTE
                # Così il numero combacia PERFETTAMENTE con la media del team e le barre
                official_avg = spdf[spdf['Player'] == target_player]['Speed'].values[0]
                
                c_avg, c_max = st.columns(2)
                c_avg.metric("Velocità Media (Totale)", f"{official_avg:.2f} m/s")
                c_max.metric("Picco Velocità", f"{smooth_speed.max():.2f} m/s")

            # --- GIF & HEATMAP ---
            ball_mask = df['team'] == 'Ball'
            if ball_mask.any():
                 df.loc[ball_mask, 'x_meters'] = df.loc[ball_mask, 'x_meters'].rolling(window=5, min_periods=1, center=True).mean()
                 df.loc[ball_mask, 'y_meters'] = df.loc[ball_mask, 'y_meters'].rolling(window=5, min_periods=1, center=True).mean()
            
            st.markdown("### 🌀 GIF Voronoi")
            # Mostriamo uno spinner mentre la funzione (eventualmente) lavora
            with st.spinner("Caricamento Animazione Tattica..."):
                # Passiamo 'sub' che contiene solo i frame selezionati (start-end)
                gif_path = get_cached_voronoi_gif(sub, file_prefix="voronoi_cache")
            
            if gif_path and os.path.exists(gif_path):
                st.image(gif_path, width="stretch")
            else:
                st.warning("Nessun dato sufficiente per generare la Voronoi Map.")
            
            st.markdown("### 🔥 Heatmap")
            h1, h2 = st.columns(2)
            st_red = sub[sub['team']=='Red']; st_white = sub[sub['team']=='White']
            if not st_red.empty:
                fig, ax = plt.subplots(figsize=(6, 4)); draw_mpl_court(ax)
                sns.kdeplot(x=st_red['x_meters'], y=st_red['y_meters'], fill=True, cmap='Reds', alpha=0.6, levels=10, ax=ax)
                ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off'); ax.set_title("Red Heatmap"); h1.pyplot(fig)
            if not st_white.empty:
                fig, ax = plt.subplots(figsize=(6, 4)); draw_mpl_court(ax)
                sns.kdeplot(x=st_white['x_meters'], y=st_white['y_meters'], fill=True, cmap='Blues', alpha=0.6, levels=10, ax=ax)
                ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off'); ax.set_title("White Heatmap"); h2.pyplot(fig)