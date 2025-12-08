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
import tempfile

# --- CONFIGURAZIONE ---
CSV_FILE = 'tracking_data.csv'
IMAGES_FOLDER = 'datasets' 
RADAR_HEIGHT = 300
COURT_WIDTH = 3840 
COURT_HEIGHT = 2160
POSSESSION_THRESHOLD = 60 

# --- PARAMETRI FISICI ---
FPS = 30.0
REAL_WIDTH_M = 28.0
REAL_HEIGHT_M = 15.0
PX_TO_M = REAL_WIDTH_M / COURT_WIDTH 
MAX_PIXEL_STEP = 100 
SMOOTHING_WINDOW = 5
MIN_SPEED_THRESHOLD = 3.0

# --- FUNZIONI UTILI ---
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', str(filename))
    if match: return int(match.group(1))
    return 0

def draw_radar_court(img, width, height, color=(200, 200, 200)):
    thick = 2
    ppm = width / REAL_WIDTH_M 
    cv2.line(img, (int(width/2), 0), (int(width/2), height), color, thick)
    cv2.circle(img, (int(width/2), int(height/2)), int(1.8 * ppm), color, thick)
    paint_w = int(5.8 * ppm); paint_h = int(4.9 * ppm)
    y_top = int((height - paint_h) / 2); y_bot = int((height + paint_h) / 2)
    cv2.rectangle(img, (0, y_top), (paint_w, y_bot), color, thick)
    cv2.circle(img, (paint_w, int(height/2)), int(1.8 * ppm), color, thick)
    cv2.rectangle(img, (width - paint_w, y_top), (width, y_bot), color, thick)
    cv2.circle(img, (width - paint_w, int(height/2)), int(1.8 * ppm), color, thick)
    three_pt = int(6.75 * ppm)
    cv2.ellipse(img, (0, int(height/2)), (int(width*0.35), int(height*0.8)), 0, -90, 90, color, thick)
    cv2.ellipse(img, (width, int(height/2)), (int(width*0.35), int(height*0.8)), 0, 90, 270, color, thick)
    return img

def draw_mpl_court(ax, color='black', lw=2):
    court = Rectangle((0, 0), COURT_WIDTH, COURT_HEIGHT, linewidth=lw, color=color, fill=False)
    ax.add_patch(court)
    ax.plot([COURT_WIDTH/2, COURT_WIDTH/2], [0, COURT_HEIGHT], color=color, linewidth=lw)
    ax.add_patch(Circle((COURT_WIDTH/2, COURT_HEIGHT/2), 180, color=color, fill=False, linewidth=lw))
    ax.add_patch(Rectangle((0, COURT_HEIGHT/2 - 250), 550, 500, linewidth=lw, color=color, fill=False))
    ax.add_patch(Rectangle((COURT_WIDTH-550, COURT_HEIGHT/2 - 250), 550, 500, linewidth=lw, color=color, fill=False))

@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE): return None
    df = pd.read_csv(CSV_FILE)
    if 'action_id' not in df.columns:
        df['action_id'] = df['frame_filename'].apply(lambda x: x.split('_frame')[0] if '_frame' in x else 'unknown')
    df['frame_id'] = df['frame_filename'].apply(extract_frame_number)
    df['player_unique_id'] = df['team'] + "_" + df['number'].astype(str)
    if 'x_meters' not in df.columns:
        df['x_meters'] = df['x_feet'] * PX_TO_M
        df['y_meters'] = df['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT) 
    return df

# --- LOGICA POSSESSO ---
def get_possession_table(df_subset):
    ball_df = df_subset[df_subset['team'] == 'Ball'][['frame_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
    players_df = df_subset[df_subset['team'].isin(['Red', 'White'])].copy()
    if ball_df.empty or players_df.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    m = pd.merge(players_df, ball_df, on='frame_id', suffixes=('', '_b'), how='inner')
    mw, mh = m['bbox_w']*0.05, m['bbox_h']*0.10
    bx_c = m['bbox_x_b'] + m['bbox_w_b']/2; by_c = m['bbox_y_b'] + m['bbox_h_b']/2
    px1, px2 = m['bbox_x']+mw, m['bbox_x']+m['bbox_w']-mw
    py1, py2 = m['bbox_y'], m['bbox_y']+m['bbox_h']-mh
    m['is_overlap'] = (bx_c > px1) & (bx_c < px2) & (by_c > py1) & (by_c < py2)
    candidates = m[m['is_overlap']].copy()
    if candidates.empty: return pd.DataFrame(columns=['frame_id', 'player_unique_id'])
    
    p_cx = candidates['bbox_x'] + candidates['bbox_w']/2
    p_cy = candidates['bbox_y'] + candidates['bbox_h']/2
    candidates['dist_px'] = np.sqrt((p_cx - bx_c)**2 + (p_cy - by_c)**2)
    best_idx = candidates.groupby('frame_id')['dist_px'].idxmin()
    return candidates.loc[best_idx, ['frame_id', 'player_unique_id']].set_index('frame_id')

# --- CALCOLO STATISTICHE IBRIDE ---
def calculate_advanced_stats_hybrid(df_action, player_id, current_frame, ownership_table):
    p_data = df_action[(df_action['player_unique_id'] == player_id) & (df_action['frame_id'] <= current_frame)].sort_values('frame_id')
    if len(p_data) < 5: return 0, 0, 0, 0
    
    p_data['xp'] = p_data['x_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    p_data['yp'] = p_data['y_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    dx = p_data['xp'].diff().fillna(0); dy = p_data['yp'].diff().fillna(0)
    step_px = np.sqrt(dx**2 + dy**2); step_px[step_px > MAX_PIXEL_STEP] = 0
    step_px[step_px < 2.0] = 0
    total_dist_m = step_px.sum() * PX_TO_M
    
    speed_kmh = 0.0
    if len(p_data) > 15:
        dist_last_15_px = step_px.tail(15).sum()
        time_s = 15 / FPS
        speed_kmh = (dist_last_15_px * PX_TO_M / time_s) * 3.6
        if speed_kmh < MIN_SPEED_THRESHOLD: speed_kmh = 0.0

    p_data = p_data.join(ownership_table, on='frame_id', rsuffix='_owner')
    p_data['is_mine'] = (p_data['player_unique_id_owner'] == player_id)
    is_mine_buf = p_data['is_mine'].rolling(window=3, center=True, min_periods=1).sum() >= 2
    off_ball_px = step_px[~is_mine_buf].sum()
    return int(total_dist_m), int(off_ball_px * PX_TO_M), round(speed_kmh, 1), is_mine_buf.sum()

# --- RENDERING VIDEO ---
def render_dual_view(f_id, df, quality_mode, highlight_id=None, is_possessor=False):
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None, None, 0, 0, 0, False
    fname = fname_row['frame_filename'].iloc[0]
    
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

    target_w = 3840 if quality_mode == "Massima (4K)" else 960
    scale = target_w / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    
    radar_img = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_img, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    ball_row = frame_data[frame_data['team'] == 'Ball']

    for _, row in frame_data.iterrows():
        xm = row['x_feet'] * PX_TO_M; ym = row['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)
        rx = int(xm * sx); ry = int(ym * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        
        t_str = str(row['team']); raw_c = str(row.get('raw_class', '')).lower()
        
        if t_str == 'Ball':
            cv2.circle(radar_img, (rx, ry), 8, (0, 165, 255), -1)
        elif 'Ref' in t_str or 'ref' in raw_c:
            cv2.circle(radar_img, (rx, ry), 5, (0, 255, 0), -1)
        elif t_str in ['Red', 'White']:
            c = (0, 0, 255) if t_str == 'Red' else (255, 255, 255)
            if t_str == 'Red': c = (0, 0, 255)
            cv2.circle(radar_img, (rx, ry), 6, c, -1)
            
            if highlight_id and row['player_unique_id'] == highlight_id:
                bx, by = int(row['bbox_x']*scale), int(row['bbox_y']*scale)
                bw, bh = int(row['bbox_w']*scale), int(row['bbox_h']*scale)
                col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
                cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 3)
                cv2.putText(frame_img, row['player_unique_id'], (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_box, 2)
                cv2.circle(radar_img, (rx, ry), 12, col_box, 2)

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB), 0, 0, 0, is_possessor

# --- MAIN ---
st.set_page_config(page_title="CourtSense", layout="wide")
st.title("🏀 CourtSense: Tactical Dashboard")

df_full = load_data()
if df_full is None: st.error("CSV non trovato."); st.stop()

st.sidebar.header("⚙️ Pannello di Controllo")
available_actions = sorted(df_full['action_id'].unique())
idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("📂 Seleziona Azione:", available_actions, index=idx)

df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
ownership_table = get_possession_table(df)

unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

analysis_mode = st.sidebar.radio("Modalità:", ("🕹️ Navigazione (Manuale)", "▶️ Genera Video (Auto)"))
st.sidebar.markdown("---")
selected_player = st.sidebar.selectbox("Traccia Giocatore:", player_list)

col_main, col_side = st.columns([3, 1])
video_ph = col_main.empty(); radar_ph = col_side.empty(); stats_ph = col_side.empty()

if analysis_mode == "🕹️ Navigazione (Manuale)":
    sel_frame = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, selected_player, sel_frame, ownership_table)
    
    is_owner = False
    try:
        if sel_frame in ownership_table.index and ownership_table.loc[sel_frame]['player_unique_id'] == selected_player: is_owner = True
    except: pass
    
    vid, rad, _, _, _, hold = render_dual_view(sel_frame, df, "Ottimizzata (HD)", selected_player, is_owner)
    
    if vid is not None:
        video_ph.image(vid, channels="RGB", width="stretch")
        radar_ph.image(rad, channels="RGB", caption="Tactical Board", width="stretch")
        icon = "🏀" if is_owner else ""
        stats_ph.markdown(f"### Frame {sel_frame}\n**Player:** `{selected_player}` {icon}\n📏 **Dist:** {dt}m\n⚡ **Speed:** {spd} km/h")

    # (Grafici statici rimossi per brevità)

else: # AUTO (GENERAZIONE VIDEO)
    st.info("ℹ️ Genera un video fluido per evitare blocchi del browser.")
    start, end = st.sidebar.select_slider("Clip:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    
    if st.sidebar.button("🎥 GENERA VIDEO ANALISI"):
        frames = [f for f in unique_frames if start <= f <= end]
        output_file = "analysis_output.mp4"
        prog_bar = st.progress(0, "Rendering in corso...")
        
        cum_m = 0; pos_buffer = []
        video_frames = []
        
        for i, f_id in enumerate(frames):
            prog_bar.progress(int(i / len(frames) * 100))
            
            # Calcoli (Semplificati per speed rendering)
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            spd = 0.0
            if not curr.empty:
                cx, cy = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buffer.append((cx, cy))
                if len(pos_buffer)>15: pos_buffer.pop(0)
                if len(pos_buffer)>=2:
                    step = np.sqrt((cx-pos_buffer[-2][0])**2 + (cy-pos_buffer[-2][1])**2)
                    if step < MAX_PIXEL_STEP: cum_m += step * PX_TO_M
                if len(pos_buffer)>=10:
                     d = np.sqrt((pos_buffer[-1][0]-pos_buffer[0][0])**2 + (pos_buffer[-1][1]-pos_buffer[0][1])**2)
                     spd = (d * PX_TO_M / ((len(pos_buffer)-1)/FPS)) * 3.6

            is_owner = False
            try:
                if f_id in ownership_table.index and ownership_table.loc[f_id]['player_unique_id'] == selected_player: is_owner = True
            except: pass
            
            # Rendering Frame
            vid, rad, _, _, _, _ = render_dual_view(f_id, df, "Ottimizzata (HD)", selected_player, is_owner)
            
            if vid is not None:
                # Componi immagine unica (Video + Radar + Stats)
                h, w, _ = vid.shape
                # Resize radar to match height
                rad_res = cv2.resize(rad, (int(rad.shape[1] * (h / rad.shape[0])), h))
                
                # Canvas
                combined = np.hstack((vid, rad_res))
                
                # Overlay Text
                icon = " (BALL)" if is_owner else ""
                text = f"Ply: {selected_player}{icon} | Dist: {int(cum_m)}m | Spd: {spd:.1f} km/h"
                cv2.rectangle(combined, (0, 0), (w, 60), (0,0,0), -1)
                cv2.putText(combined, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                video_frames.append(combined)

        prog_bar.progress(90, "Salvataggio MP4...")
        imageio.mimwrite(output_file, video_frames, fps=30, macro_block_size=1)
        
        prog_bar.empty()
        st.success("Video Generato!")
        st.video(output_file)

    # --- REPORT FINALE ---
    st.markdown("---"); st.subheader("📈 Report")
    if st.button("Genera Metriche"):
        # (Copia qui il blocco report del codice precedente se serve)
        pass