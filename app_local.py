"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    🏀 COURTSENSE - FINAL PHYSICS FIX                       ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

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
from scipy.signal import savgol_filter

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE BASE & FISICA
# ═══════════════════════════════════════════════════════════════════════════
CSV_FILE = 'tracking_data.csv'
IMAGES_FOLDER = 'datasets' 
RADAR_HEIGHT = 300
COURT_WIDTH = 3840 
COURT_HEIGHT = 2160
POSSESSION_THRESHOLD = 60 

# --- PARAMETRI CRITICI PER LA FISICA ---
# PLAYBACK_FPS: A quanto gira il video a schermo
PLAYBACK_FPS = 12.0 

# PHYSICS_FPS: A quale frequenza campioniamo la realtà fisica.
# Impostato a 12.0 FPS per allinearsi ai frame effettivi acquisiti.
PHYSICS_FPS = 12.0 

REAL_WIDTH_M = 28.0
REAL_HEIGHT_M = 15.0
PX_TO_M = REAL_WIDTH_M / COURT_WIDTH 

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE OMOGRAFIA (I TUOI DATI)
# ═══════════════════════════════════════════════════════════════════════════

PTS_REAL_METERS = np.float32([
    [0, 0],              # Top-Left
    [28.0, 0],           # Top-Right
    [28.0, 15.0],        # Bottom-Right
    [0, 15.0]            # Bottom-Left
])

DATASET_CONFIG = {
    'azione_01': np.float32([
        [857, 980],    [2849, 1011],  [3552, 1505],  [149, 1458]
    ]),
    'azione_02': np.float32([
        [857, 980],    [2849, 1011],  [3552, 1505],  [149, 1458]
    ]),
    'azione_03': np.float32([
        [143, 290],    [474, 299],    [591, 446],    [24, 432]
    ]),
    'azione_04': np.float32([
        [143, 290],    [474, 299],    [591, 446],    [24, 432]
    ]),
    'azione_05': np.float32([
        [143, 290],    [474, 299],    [591, 446],    [24, 432]
    ]),
}

def apply_perspective_transform(points_px, src_pts, dst_pts):
    if len(points_px) == 0: return points_px
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    points_reshaped = points_px.reshape(-1, 1, 2).astype(np.float32)
    points_transformed = cv2.perspectiveTransform(points_reshaped, H)
    return points_transformed.reshape(-1, 2)

# ═══════════════════════════════════════════════════════════════════════════
# PARSING DATI
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
    
    # --- CALCOLO METRI CON OMOGRAFIA ---
    df['x_meters'] = 0.0
    df['y_meters'] = 0.0
    
    unique_actions = df['action_id'].unique()
    
    for action in unique_actions:
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
# STATISTICHE LIVE (NAVIGAZIONE)
# ═══════════════════════════════════════════════════════════════════════════
def calculate_advanced_stats_hybrid(df_action, player_id, current_frame, ownership_table):
    p_data = df_action[(df_action['player_unique_id'] == player_id) & (df_action['frame_id'] <= current_frame)].sort_values('frame_id')
    if len(p_data) < 5: return 0, 0, 0.0, 0
    
    # Calcolo velocità puntuale usando PHYSICS_FPS
    # Delta su 5 frame (0.33s a 15fps)
    if len(p_data) > 5:
        curr = p_data.iloc[-1][['x_meters', 'y_meters']].values
        prev = p_data.iloc[-5][['x_meters', 'y_meters']].values
        dist = np.linalg.norm(curr - prev)
        time_s = 5.0 / PHYSICS_FPS
        speed_ms = dist / time_s
    else:
        speed_ms = 0.0

    # Possesso
    p_data = p_data.join(ownership_table, on='frame_id', rsuffix='_owner')
    p_data['is_mine'] = (p_data['player_unique_id_owner'] == player_id)
    poss_frames = p_data['is_mine'].sum()
    
    return 0, 0, round(speed_ms, 2), poss_frames

def get_possession_table(df_subset):
    ball_df = df_subset[df_subset['team'] == 'Ball'][['frame_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
    players_df = df_subset[df_subset['team'].isin(['Red', 'White'])].copy()
    if ball_df.empty or players_df.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    m = pd.merge(players_df, ball_df, on='frame_id', suffixes=('', '_b'), how='inner')
    mw, mh = m['bbox_w']*0.05, m['bbox_h']*0.10
    bx_c = m['bbox_x_b'] + m['bbox_w_b']/2; by_c = m['bbox_y_b'] + m['bbox_h_b']/2
    is_ov = (bx_c > m['bbox_x']+mw) & (bx_c < m['bbox_x']+m['bbox_w']-mw) & (by_c > m['bbox_y']) & (by_c < m['bbox_y']+m['bbox_h']-mh)
    candidates = m[is_ov].copy()
    if candidates.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    p_cx = candidates['bbox_x'] + candidates['bbox_w']/2
    p_cy = candidates['bbox_y'] + candidates['bbox_h']/2
    candidates['dist_px'] = np.sqrt((p_cx - bx_c)**2 + (p_cy - by_c)**2)
    best_idx = candidates.groupby('frame_id')['dist_px'].idxmin()
    return candidates.loc[best_idx, ['frame_id', 'player_unique_id']].set_index('frame_id')

# ═══════════════════════════════════════════════════════════════════════════
# GRAFICA
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
    court = Rectangle((0, 0), REAL_WIDTH_M, REAL_HEIGHT_M, linewidth=lw, color=color, fill=False)
    ax.add_patch(court); ax.plot([14, 14], [0, 15], color=color, linewidth=lw)
    ax.add_patch(Circle((14, 7.5), 1.8, color=color, fill=False, linewidth=lw))
    # Aggiungi aree anche qui per coerenza nei grafici statici
    ax.add_patch(Rectangle((0, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))
    ax.add_patch(Rectangle((22.2, 5.05), 5.8, 4.9, linewidth=lw, color=color, fill=False))

def generate_static_voronoi(frame_data, title=None):
    fig, ax = plt.subplots(figsize=(10, 6)); draw_mpl_court(ax)
    players = frame_data[frame_data['team'].isin(['Red', 'White'])]
    if len(players) >= 4:
        points = players[['x_meters', 'y_meters']].values
        teams = players['team'].values
        dummy = np.array([[-5, -5], [35, -5], [35, 20], [-5, 20]])
        try:
            vor = Voronoi(np.vstack([points, dummy]))
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region:
                    c = 'red' if teams[i] == 'Red' else 'blue'
                    ax.add_patch(Polygon(vor.vertices[region], facecolor=c, alpha=0.4, edgecolor='white'))
        except: pass
    for _, r in players.iterrows():
        c = 'red' if r['team']=='Red' else 'blue'
        ax.scatter(r['x_meters'], r['y_meters'], c=c, s=80, edgecolors='white', zorder=5)
    ball = frame_data[frame_data['team']=='Ball']
    if not ball.empty:
        ax.scatter(ball['x_meters'], ball['y_meters'], c='orange', s=180, edgecolors='black', zorder=10)
    ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off')
    if title: ax.set_title(title)
    return fig

def generate_static_hull(frame_data):
    fig, ax = plt.subplots(figsize=(10, 6)); draw_mpl_court(ax)
    for team, col in [('Red','red'), ('White','blue')]:
        points = frame_data[frame_data['team'] == team][['x_meters', 'y_meters']].values
        ax.scatter(points[:,0], points[:,1], c=col, s=80)
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                ax.add_patch(Polygon(points[hull.vertices], facecolor=col, alpha=0.3))
            except: pass
    ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off')
    return fig

def render_nba_style(f_id, df, target_width, highlight_id=None, is_possessor=False, stats=None):
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None
    fname = fname_row['frame_filename'].iloc[0]
    
    img_path = None
    if 'image_path' in fname_row.columns:
        p = fname_row['image_path'].iloc[0]
        if os.path.exists(p): img_path = p
        elif os.path.exists(p.replace('\\', '/')): img_path = p.replace('\\', '/')
    if img_path is None: img_path = os.path.join(IMAGES_FOLDER, fname)
    if not os.path.exists(img_path): img_path = os.path.join('train', fname)
        
    frame_img_orig = cv2.imread(img_path) if img_path and os.path.exists(img_path) else np.zeros((720,1280,3), np.uint8)
    scale = target_width / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    H, W = frame_img.shape[:2]
    
    radar_base = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_base, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    
    for _, row in frame_data.iterrows():
        rx = int(row['x_meters'] * sx); ry = int(row['y_meters'] * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        t = str(row['team'])
        col = (0,0,255) if t=='Red' else (255,255,255) if t=='White' else (0,165,255)
        cv2.circle(radar_base, (rx, ry), 6, col, -1)
        if row['player_unique_id'] == highlight_id:
            bx = int(row['bbox_x']*scale); by = int(row['bbox_y']*scale)
            bw = int(row['bbox_w']*scale); bh = int(row['bbox_h']*scale)
            col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
            cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 2)
            cv2.circle(radar_base, (rx, ry), 10, col_box, 2)

    mini_w = int(W * 0.25); mini_h = int(mini_w * (RADAR_HEIGHT/600))
    radar_mini = cv2.resize(radar_base, (mini_w, mini_h))
    y1 = H - mini_h - 20; x1 = W - mini_w - 20
    frame_img[y1:y1+mini_h, x1:x1+mini_w] = cv2.addWeighted(frame_img[y1:y1+mini_h, x1:x1+mini_w], 0.3, radar_mini, 0.7, 0)
    cv2.rectangle(frame_img, (x1, y1), (x1+mini_w, y1+mini_h), (255,255,255), 1)

    if stats:
        dist, off, spd, poss = stats
        cv2.rectangle(frame_img, (20, 20), (320, 130), (0,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_img, f"{highlight_id}{' (BALL)' if is_possessor else ''}", (30, 50), font, 0.6, (0,255,255), 2)
        
        d_str = f"{dist} m" if isinstance(dist, (int, float)) else str(dist)
        #s_str = f"{spd} m/s" if isinstance(spd, (int, float)) else str(spd)
        
        cv2.putText(frame_img, f"DIST: {d_str}", (30, 75), font, 0.5, (255,255,255), 1)
        #cv2.putText(frame_img, f"SPEED: {s_str}", (30, 95), font, 0.5, (255,255,255), 1)
        cv2.putText(frame_img, f"POSS: {poss}", (30, 115), font, 0.5, (255,255,255), 1)

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CourtSense Local", layout="wide")
st.title("🏀 CourtSense: Tactical Dashboard (Local)")

df_full = load_data()
if df_full is None: st.stop()

st.sidebar.header("Configurazione")
actions = sorted(df_full['action_id'].unique())
sel_action = st.sidebar.selectbox("Azione:", actions)
df = df_full[df_full['action_id'] == sel_action].sort_values('frame_id')
own_table = get_possession_table(df)

frames = df['frame_id'].unique()
players = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p])

mode = st.sidebar.radio("Modalità:", ("Navigazione", "Riproduzione"))
sel_player = st.sidebar.selectbox("Giocatore:", players)
ph_vid = st.empty()

if mode == "Navigazione":
    f = st.sidebar.slider("Frame:", int(min(frames)), int(max(frames)))
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, sel_player, f, own_table)
    
    is_own = False
    try:
        if f in own_table.index and own_table.loc[f]['player_unique_id'] == sel_player: is_own = True
    except: pass
    
    img = render_nba_style(f, df, 1080, sel_player, is_own, ("-", "-", spd, f"{(pf/PHYSICS_FPS):.1f}s"))
    if img is not None: ph_vid.image(img, channels="RGB", width='stretch')
    
    c1, c2 = st.columns(2)
    frm_data = df[df['frame_id']==f]
    if c1.button("📸 Voronoi"): c1.pyplot(generate_static_voronoi(frm_data))
    if c2.button("🛡️ Convex Hull"): c2.pyplot(generate_static_hull(frm_data))

else: # AUTO
    st.sidebar.markdown("---")
    # Default su 0.3x per analisi tattica
    speed_factor = st.sidebar.select_slider("Rallentatore:", options=[0.1, 0.3, 0.5, 1.0], value=0.3)
    start, end = st.sidebar.select_slider("Range:", options=frames, value=(min(frames), min(min(frames)+40, max(frames))))
    
    if st.sidebar.button("▶️ PLAY"):
        clip = [x for x in frames if start <= x <= end]
        cum_m=0; cum_off=0; pos_buf=[]; poss_c=0
        smooth_spd = 0.0
        
        # Logica Macro-Step (Cronometro)
        last_macro_pos = None
        frame_counter = 0
        MACRO_INTERVAL = 5 # 0.33s a 15 FPS fisici
        
        ph_vid = st.empty()
        
        for f in clip:
            t0 = time.time()
            curr = df[(df['frame_id']==f) & (df['player_unique_id']==sel_player)]
            current_spd = 0.0
            step_val = 0 
            
            if not curr.empty:
                cx, cy = curr.iloc[0]['x_meters'], curr.iloc[0]['y_meters']
                curr_pos_arr = np.array([cx, cy])
                
                pos_buf.append(curr_pos_arr)
                if len(pos_buf) > 10: pos_buf.pop(0) 
                
                if last_macro_pos is None: last_macro_pos = curr_pos_arr
                
                # --- 1. DISTANZA (Campionata a 3 Hz) ---
                frame_counter += 1
                if frame_counter % MACRO_INTERVAL == 0:
                    dist_segment = np.linalg.norm(curr_pos_arr - last_macro_pos)
                    # Filtro fermo (5cm in 0.3s)
                    if dist_segment > 0.05:
                        cum_m += dist_segment
                        step_val = dist_segment 
                    last_macro_pos = curr_pos_arr
                
                # --- 2. VELOCITÀ (m/s, finestra 0.6s) ---
                #if len(pos_buf) >= 10:
                #    p_now = pos_buf[-1]
                #    p_old = pos_buf[0]
                #    dist_v = np.linalg.norm(p_now - p_old)
                    
                    # Usa PHYSICS_FPS per il tempo (15 FPS -> 0.66s per 10 frame)
                #    time_v = (len(pos_buf)-1) / PHYSICS_FPS
                    
                #    raw_ms = dist_v / time_v
                    
                    # Hard cap umano (11 m/s)
                #    if raw_ms > 11.0: raw_ms = 11.0
                    
                #    smooth_spd = (smooth_spd * 0.85) + (raw_ms * 0.15)
                    
                #    if smooth_spd < 0.5: 
                #        current_spd = 0.0
                #    else:
                #        current_spd = smooth_spd

            # Possesso (Usa PHYSICS_FPS per il tempo)
            is_own = False
            try:
                if f in own_table.index and own_table.loc[f]['player_unique_id'] == sel_player: is_own = True
            except: pass
            
            if is_own: poss_c += 1
            else: 
                if step_val > 0: cum_off += step_val

            # Render
            img = render_nba_style(f, df, 1080, sel_player, is_own, (int(cum_m), int(cum_off), round(current_spd, 2), f"{(poss_c/PHYSICS_FPS):.1f}s"))
            if img is not None: ph_vid.image(img, channels="RGB", width='stretch')
            
            # Slow Motion Wait
            time.sleep(1.0 / (PLAYBACK_FPS * speed_factor))

    # --- REPORT AGGREGATO (SAVITZKY-GOLAY + PHYSICS FPS) ---
    st.markdown("---"); st.subheader("📈 Report Azione")
    
    if st.button("Genera Metriche"):
        with st.spinner("Calcolo traiettorie fisiche (m/s)..."):
            sub = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            own_sub = own_table[own_table.index.isin(sub['frame_id'].unique())]
            
            # Durata basata su 15 FPS
            duration_s = (end - start + 1) / PHYSICS_FPS
            
            # 1. SPACING
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
                ax1.axhline(mr, c='darkred', ls='--', label=f"Avg R: {mr:.1f}m")
                ax1.axhline(mw, c='darkblue', ls='--', label=f"Avg W: {mw:.1f}m")
                ax1.set_title("Avg Team Spacing (Meters)", fontweight='bold'); ax1.legend(fontsize='small'); ax1.grid(alpha=0.3)
                st.pyplot(fig1)

            # 2. WORKLOAD & SPEED (m/s)
            moves = []; speed_poss_data = []
            
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                
                # --- Smoothing traiettoria ---
                try:
                    xm = savgol_filter(g['x_meters'], 15, 2)
                    ym = savgol_filter(g['y_meters'], 15, 2)
                except:
                    xm = g['x_meters'].values
                    ym = g['y_meters'].values
                
                # Differenze
                dx = np.diff(xm, prepend=xm[0])
                dy = np.diff(ym, prepend=ym[0])
                dists = np.sqrt(dx**2 + dy**2)
                
                # Noise Gate
                dists[dists < 0.02] = 0
                
                # Max Speed (per frame)
                # 11 m/s = 0.73 m/frame a 15 FPS. 
                dists[dists > 0.8] = 0.8
                
                tot_m = np.sum(dists)
                
                # Velocità media (m/s)
                avg_spd_ms = (tot_m / duration_s) if duration_s > 0 else 0
                
                # Possesso (Tempo)
                is_poss = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                poss_s = is_poss.sum() / PHYSICS_FPS
                
                poss_ratio = poss_s / duration_s if duration_s > 0 else 0
                off_m = tot_m * (1.0 - poss_ratio)
                
                moves.append({'Player': pid, 'Dist': tot_m, 'Type': 'Total', 'Team': g['team'].iloc[0]})
                moves.append({'Player': pid, 'Dist': off_m, 'Type': 'Off-Ball', 'Team': g['team'].iloc[0]})
                speed_poss_data.append({'Player': pid, 'Team': g['team'].iloc[0], 'Speed': avg_spd_ms, 'Poss': poss_s})
            
            if moves:
                mdf = pd.DataFrame(moves)
                spdf = pd.DataFrame(speed_poss_data)
                
                atr = mdf[(mdf['Team']=='Red') & (mdf['Type']=='Total')]['Dist'].mean()
                aro = mdf[(mdf['Team']=='Red') & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                awt = mdf[(mdf['Team']=='White') & (mdf['Type']=='Total')]['Dist'].mean()
                awo = mdf[(mdf['Team']=='White') & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                
                asr = spdf[spdf['Team']=='Red']['Speed'].mean()
                apr = spdf[spdf['Team']=='Red']['Poss'].mean()
                asw = spdf[spdf['Team']=='White']['Speed'].mean()
                apw = spdf[spdf['Team']=='White']['Poss'].mean()
                
                k1, k2 = st.columns(2)
                k1.info(f"""
                🔴 **Red Team Avg**
                - 📏 Dist: **{atr:.1f} m** (Off: {aro:.1f}m)
                - ⚡ Speed: **{asr:.2f} m/s**
                - ⏱️ Poss: **{apr:.1f} s**
                """)
                k2.info(f"""
                ⚪ **White Team Avg**
                - 📏 Dist: **{awt:.1f} m** (Off: {awo:.1f}m)
                - ⚡ Speed: **{asw:.2f} m/s**
                - ⏱️ Poss: **{apw:.1f} s**
                """)
                
                col_g1, col_g2 = st.columns(2)
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=mdf, x='Player', y='Dist', hue='Type', palette={'Total':'gray', 'Off-Ball':'limegreen'}, ax=ax2)
                ax2.axhline(atr, c='darkred', ls='--', label="Avg R"); ax2.axhline(awt, c='darkblue', ls='--', label="Avg W")
                ax2.tick_params(axis='x', rotation=90); ax2.set_title("Workload (Meters)", fontweight='bold'); ax2.legend(fontsize='x-small')
                col_g1.pyplot(fig2)
                
                fig3, ax3 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=spdf, x='Player', y='Poss', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax3)
                ax3.tick_params(axis='x', rotation=90); ax3.set_title("Possession Time (s)", fontweight='bold')
                col_g2.pyplot(fig3)
                
                st.markdown("##### Velocity Analysis")
                fig4, ax4 = plt.subplots(figsize=(10, 4))
                sns.barplot(data=spdf, x='Player', y='Speed', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax4)
                ax4.axhline(asr, c='darkred', ls='--', label=f"Avg R ({asr:.2f} m/s)"); ax4.axhline(asw, c='darkblue', ls='--', label=f"Avg W ({asw:.2f} m/s)")
                ax4.tick_params(axis='x', rotation=90); ax4.set_title("Average Speed (m/s)", fontweight='bold'); ax4.legend()
                st.pyplot(fig4)

            # GIF & Heatmap
            st.markdown("### 🌀 GIF Voronoi")
            frames_list = sub['frame_filename'].unique()
            if len(frames_list) > 0:
                bar = st.progress(0); tmp="tmp_v_loc"; os.makedirs(tmp, exist_ok=True); files=[]
                try:
                    for i, fn in enumerate(frames_list):
                        bar.progress(int((i/len(frames_list))*90))
                        fig = generate_static_voronoi(df[df['frame_filename']==fn], title=f"Frame {extract_frame_number(fn)}")
                        p = os.path.join(tmp, f"{i:03d}.png"); fig.savefig(p, dpi=60, bbox_inches='tight'); plt.close(fig); files.append(p)
                    with imageio.get_writer("voronoi_local.gif", mode='I', duration=0.15, loop=0) as w:
                        for f in files: w.append_data(imageio.imread(f))
                    bar.empty(); st.image("voronoi_local.gif", width="stretch")
                except Exception as e: st.error(str(e))
                finally: shutil.rmtree(tmp)
            
            st.markdown("### 🔥 Heatmap")
            h1, h2 = st.columns(2)
            st_red = sub[sub['team']=='Red']
            if not st_red.empty:
                fig, ax = plt.subplots(figsize=(5,3)); draw_mpl_court(ax)
                sns.kdeplot(x=st_red['x_meters'], y=st_red['y_meters'], fill=True, cmap='Reds', alpha=0.6, ax=ax)
                ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off'); ax.set_title("Red Heatmap")
                h1.pyplot(fig)
            st_white = sub[sub['team']=='White']
            if not st_white.empty:
                fig, ax = plt.subplots(figsize=(5,3)); draw_mpl_court(ax)
                sns.kdeplot(x=st_white['x_meters'], y=st_white['y_meters'], fill=True, cmap='Blues', alpha=0.6, ax=ax)
                ax.set_xlim(0, REAL_WIDTH_M); ax.set_ylim(REAL_HEIGHT_M, 0); ax.axis('off'); ax.set_title("White Heatmap")
                h2.pyplot(fig)