import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, ConvexHull
from matplotlib.patches import Polygon, Rectangle, Circle
from itertools import combinations
import imageio.v2 as imageio

# --- CONFIGURAZIONE ---
CSV_FILE = 'tracking_data.csv'
IMAGES_FOLDER = 'datasets' 
# Dimensioni base per il calcolo (non visualizzazione)
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
        dist_last_15_m = dist_last_15_px * PX_TO_M
        time_s = 15 / FPS
        speed_kmh = (dist_last_15_m / time_s) * 3.6
        if speed_kmh < MIN_SPEED_THRESHOLD: speed_kmh = 0.0

    p_data = p_data.join(ownership_table, on='frame_id', rsuffix='_owner')
    p_data['is_mine'] = (p_data['player_unique_id_owner'] == player_id)
    is_mine_buf = p_data['is_mine'].rolling(window=3, center=True, min_periods=1).sum() >= 2
    off_ball_px = step_px[~is_mine_buf].sum()
    off_ball_m = off_ball_px * PX_TO_M
    poss_frames = is_mine_buf.sum()
    return int(total_dist_m), int(off_ball_m), round(speed_kmh, 1), poss_frames

# --- RENDERING VIDEO (BROADCAST OVERLAY) ---
def render_broadcast_view(f_id, df, quality_mode, highlight_id=None, is_possessor=False, stats_info=None):
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None
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

    # 1. Video Base (Resize per fluidità)
    target_w = 3840 if quality_mode == "Massima (4K)" else 800
    scale = target_w / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    H_vid, W_vid = frame_img.shape[:2]
    
    # 2. Genera Radar ad Alta Definizione (poi lo ridimensioniamo)
    radar_base = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_base, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    red_c, white_c, ref_c = 0, 0, 0
    ball_row = frame_data[frame_data['team'] == 'Ball']

    for _, row in frame_data.iterrows():
        xm = row['x_feet'] * PX_TO_M; ym = row['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)
        rx = int(xm * sx); ry = int(ym * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        t_str = str(row['team']); raw_c = str(row.get('raw_class', '')).lower()
        
        if t_str == 'Ball':
            cv2.circle(radar_base, (rx, ry), 8, (0, 165, 255), -1)
        elif 'Ref' in t_str or 'ref' in raw_c:
            cv2.circle(radar_base, (rx, ry), 5, (0, 255, 0), -1); ref_c += 1
        elif t_str in ['Red', 'White']:
            c = (0, 0, 255) if t_str == 'Red' else (255, 255, 255)
            if t_str == 'Red': c = (0, 0, 255)
            cv2.circle(radar_base, (rx, ry), 6, c, -1)
            
            if highlight_id and row['player_unique_id'] == highlight_id:
                # Disegna Box sul video
                bx, by = int(row['bbox_x']*scale), int(row['bbox_y']*scale)
                bw, bh = int(row['bbox_w']*scale), int(row['bbox_h']*scale)
                col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
                cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 2)
                cv2.putText(frame_img, row['player_unique_id'], (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_box, 2)
                # Evidenzia su radar
                cv2.circle(radar_base, (rx, ry), 12, col_box, 2)

    # 3. OVERLAY RADAR (Minimappa in basso a destra)
    # Dimensione minimappa: 25% larghezza video
    mini_w = int(W_vid * 0.25)
    mini_h = int(mini_w * (RADAR_HEIGHT/600))
    radar_mini = cv2.resize(radar_base, (mini_w, mini_h))
    
    # Bordo e Sfondo semi-trasparente per il radar
    # Creiamo una regione di interesse (ROI) nel video
    pad = 20
    y1 = H_vid - mini_h - pad; y2 = H_vid - pad
    x1 = W_vid - mini_w - pad; x2 = W_vid - pad
    
    if y1 > 0 and x1 > 0:
        roi = frame_img[y1:y2, x1:x2]
        # Blend radar con sfondo (semi-trasparenza 80%)
        blended = cv2.addWeighted(roi, 0.2, radar_mini, 0.8, 0)
        # Bordo bianco
        cv2.rectangle(blended, (0,0), (mini_w-1, mini_h-1), (200,200,200), 2)
        frame_img[y1:y2, x1:x2] = blended

    # 4. OVERLAY STATISTICHE (HUD in alto a sinistra)
    if stats_info:
        # (dist_tot, dist_off, poss_s, speed)
        d_tot, d_off, poss, spd = stats_info
        
        # Sfondo nero semi-trasparente per il testo
        overlay = frame_img.copy()
        cv2.rectangle(overlay, (20, 20), (350, 140), (0, 0, 0), -1)
        frame_img = cv2.addWeighted(overlay, 0.6, frame_img, 0.4, 0)
        
        # Testo
        col_txt = (0, 255, 255) # Giallo
        icon = " (BALL)" if is_possessor else ""
        
        cv2.putText(frame_img, f"PLAYER: {highlight_id}{icon}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame_img, f"DIST: {d_tot} m", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 1)
        cv2.putText(frame_img, f"SPEED: {spd} km/h", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 1)
        cv2.putText(frame_img, f"POSS: {poss}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_txt, 1)

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

# --- GRAFICI STATICI (Invariati) ---
def generate_static_voronoi(frame_data, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    players = frame_data[frame_data['team'].isin(['Red', 'White'])]
    if len(players) >= 4:
        points = players[['x_feet', 'y_feet']].values
        dummy = np.array([[-200, -200], [4000, -200], [4000, 2400], [-200, 2400]])
        try:
            vor = Voronoi(np.vstack([points, dummy]))
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    c = 'red' if players.iloc[i]['team'] == 'Red' else 'blue'
                    ax.add_patch(Polygon(vor.vertices[region], facecolor=c, alpha=0.4, edgecolor='white'))
        except: pass
    for _, r in players.iterrows():
        c = 'red' if r['team']=='Red' else 'blue'
        ax.scatter(r['x_feet'], r['y_feet'], c=c, s=80, edgecolors='white', zorder=5)
    ax.set_xlim(0, COURT_WIDTH); ax.set_ylim(COURT_HEIGHT, 0); ax.axis('off')
    if title: ax.set_title(title, fontsize=15)
    return fig

def generate_static_hull(frame_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    colors = {'Red': 'red', 'White': 'blue'}; fill = {'Red': 'salmon', 'White': 'lightblue'}
    for team in ['Red', 'White']:
        points = frame_data[frame_data['team'] == team][['x_feet', 'y_feet']].values
        ax.scatter(points[:,0], points[:,1], c=colors[team], s=80, zorder=5)
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                poly = Polygon(points[hull.vertices], facecolor=fill[team], edgecolor=colors[team], alpha=0.3, lw=2, linestyle='--')
                ax.add_patch(poly)
            except: pass
    ax.set_xlim(0, COURT_WIDTH); ax.set_ylim(COURT_HEIGHT, 0); ax.axis('off')
    return fig

# --- MAIN ---
st.set_page_config(page_title="CourtSense", layout="wide")
st.title("🏀 CourtSense: Broadcast Analytics")

df_full = load_data()
if df_full is None: st.error("CSV non trovato."); st.stop()

st.sidebar.header("⚙️ Configurazione")
available_actions = sorted(df_full['action_id'].unique())
idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("📂 Azione:", available_actions, index=idx)

df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
ownership_table = get_possession_table(df)

unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

analysis_mode = st.sidebar.radio("Modalità:", ("🕹️ Navigazione (Manuale)", "▶️ Genera Video (Auto)"))
selected_player = st.sidebar.selectbox("Traccia Giocatore:", player_list)

video_ph = st.empty() # Placeholder unico a tutta larghezza (schermo intero)

if analysis_mode == "🕹️ Navigazione (Manuale)":
    sel_frame = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, selected_player, sel_frame, ownership_table)
    
    is_owner = False
    try:
        if sel_frame in ownership_table.index and ownership_table.loc[sel_frame]['player_unique_id'] == selected_player: is_owner = True
    except: pass
    
    # Renderizza UNICA immagine composta
    frame_final = render_broadcast_view(sel_frame, df, "Ottimizzata (HD)", selected_player, is_owner, 
                                      stats_info=(dt, do, f"{(pf/FPS):.1f} s", spd))
    
    if frame_final is not None:
        video_ph.image(frame_final, channels="RGB", use_container_width=True)

    st.markdown("---"); c1, c2 = st.columns(2)
    fdata = df[df['frame_id'] == sel_frame]
    if c1.button("📸 Voronoi (Pixel)"): c1.pyplot(generate_static_voronoi(fdata))
    if c2.button("🛡️ Convex Hull (Pixel)"): c2.pyplot(generate_static_hull(fdata))

else: # AUTO
    st.sidebar.info("Genera un video MP4 fluido con overlay tattico.")
    start, end = st.sidebar.select_slider("Clip:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    
    if st.sidebar.button("🎥 GENERA VIDEO TV-STYLE"):
        frames = [f for f in unique_frames if start <= f <= end]
        output_file = "broadcast_analysis.mp4"
        prog_bar = st.progress(0, "Rendering Broadcast...")
        
        cum_m = 0; cum_off_m = 0; pos_buffer = [] 
        curr_poss_frames = 0
        video_frames = []
        
        for i, f_id in enumerate(frames):
            prog_bar.progress(int(i / len(frames) * 100))
            
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            step_m = 0; speed_kmh = 0.0
            
            if not curr.empty:
                cx, cy = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buffer.append((cx, cy)); 
                if len(pos_buffer)>15: pos_buffer.pop(0)
                
                if len(pos_buffer)>=2:
                    cx_s = np.mean([p[0] for p in pos_buffer[-3:]]); cy_s = np.mean([p[1] for p in pos_buffer[-3:]])
                    if i > 0: # prev exists in loop logic implicitly by buffer
                        prev = np.mean([p for p in pos_buffer[-4:-1]], axis=0) if len(pos_buffer)>3 else pos_buffer[0]
                        raw_step = np.sqrt((cx_s-prev[0])**2 + (cy_s-prev[1])**2)
                        if raw_step < MAX_PIXEL_STEP:
                            step_m = raw_step * PX_TO_M
                            cum_m += step_m

                if len(pos_buffer)>=10:
                    d = np.sqrt((pos_buffer[-1][0]-pos_buffer[0][0])**2 + (pos_buffer[-1][1]-pos_buffer[0][1])**2)
                    tm = (len(pos_buffer)-1)/FPS
                    raw_spd = (d * PX_TO_M / tm) * 3.6
                    if raw_spd > MIN_SPEED_THRESHOLD: speed_kmh = min(raw_spd, 36.0)

            is_owner = False
            try:
                if f_id in ownership_table.index and ownership_table.loc[f_id]['player_unique_id'] == selected_player: is_owner = True
            except: pass
            
            if is_owner: curr_poss_frames += 1
            else: cum_off_m += step_m
            
            # Rendering Broadcast Frame
            stats_tuple = (int(cum_m), int(cum_off_m), f"{(curr_poss_frames/FPS):.1f}s", round(speed_kmh, 1))
            final_img = render_broadcast_view(f_id, df, "Ottimizzata (HD)", selected_player, is_owner, stats_tuple)
            
            if final_img is not None:
                video_frames.append(final_img)

        prog_bar.progress(90, "Salvataggio MP4...")
        imageio.mimwrite(output_file, video_frames, fps=30, macro_block_size=1)
        
        prog_bar.empty()
        st.success("Video Broadcast Pronto!")
        st.video(output_file)

    st.markdown("---"); st.subheader("📈 Report")
    if st.button("Genera Metriche"):
        with st.spinner("Calcolo Analisi Completa..."):
            # Filtra i dati nel range selezionato
            sub = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            ball_sub = sub[sub['team'] == 'Ball']
            duration_s = (end - start + 1) / FPS
            
            # --- 1. SPACING CHART ---
            spac = []
            for f, g in players.groupby('frame_id'):
                for t in ['Red', 'White']:
                    tg = g[g['team'] == t]
                    if len(tg) >= 2:
                        # Distanza media tra compagni (in pixel) convertita in metri
                        dists_px = [np.linalg.norm(a-b) for a,b in combinations(tg[['x_feet','y_feet']].values, 2)]
                        dists_m = [d * PX_TO_M for d in dists_px]
                        spac.append({'f': f, 't': t, 'v': np.mean(dists_m)})
            
            if spac:
                sdf = pd.DataFrame(spac)
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.lineplot(data=sdf, x='f', y='v', hue='t', palette={'Red':'red','White':'blue'}, ax=ax1)
                
                # Linee Medie
                mr = sdf[sdf['t']=='Red']['v'].mean()
                mw = sdf[sdf['t']=='White']['v'].mean()
                ax1.axhline(mr, c='darkred', ls='--', label=f"Avg R: {mr:.1f}m")
                ax1.axhline(mw, c='darkblue', ls='--', label=f"Avg W: {mw:.1f}m")
                
                ax1.set_title("Avg Team Spacing (Meters)")
                ax1.set_xlabel("Frame")
                ax1.set_ylabel("Meters")
                ax1.legend(fontsize='small')
                ax1.grid(True, alpha=0.3)
                
                # Visualizza Grafico 1
                c1, c2 = st.columns(2)
                c1.pyplot(fig1)

            # --- 2. WORKLOAD, SPEED & POSSESSION ---
            moves = []
            speed_poss_data = []
            
            # Pre-calcolo possesso per il range selezionato
            own_sub = ownership_table[ownership_table.index.isin(sub['frame_id'].unique())]
            
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                
                # Calcolo Distanza (Pixel -> Metri)
                # Smoothing per il report
                g['xm'] = g['x_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
                g['ym'] = g['y_feet'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
                
                steps_px = np.sqrt(np.diff(g['xm'], prepend=g['xm'].iloc[0])**2 + np.diff(g['ym'], prepend=g['ym'].iloc[0])**2)
                
                # Filtro fisico (Pixel)
                steps_px = np.where(steps_px > MAX_PIXEL_STEP, 0, steps_px)
                
                # Noise Gate su velocità istantanea stimata
                # Se in questo frame la velocità era < 3 km/h, azzera lo step
                inst_speed = (steps_px * FPS * PX_TO_M) * 3.6
                steps_px[inst_speed < MIN_SPEED_THRESHOLD] = 0
                
                steps_m = steps_px * PX_TO_M
                
                # Possesso
                is_poss = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                
                tot_m = np.sum(steps_m)
                off_m = np.sum(steps_m[~is_poss])
                poss_s = is_poss.sum() / FPS
                avg_spd = (tot_m / duration_s) * 3.6 if duration_s > 0 else 0
                
                moves.append({'Player': pid, 'Dist': tot_m, 'Type': 'Total', 'Team': g['team'].iloc[0]})
                moves.append({'Player': pid, 'Dist': off_m, 'Type': 'Off-Ball', 'Team': g['team'].iloc[0]})
                
                speed_poss_data.append({'Player': pid, 'Team': g['team'].iloc[0], 'Speed': avg_spd, 'Poss': poss_s})
            
            if moves:
                mdf = pd.DataFrame(moves)
                sp_df = pd.DataFrame(speed_poss_data)
                
                # KPI Calculation
                atr = mdf[(mdf['Team']=='Red') & (mdf['Type']=='Total')]['Dist'].mean()
                aro = mdf[(mdf['Team']=='Red') & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                awt = mdf[(mdf['Team']=='White') & (mdf['Type']=='Total')]['Dist'].mean()
                awo = mdf[(mdf['Team']=='White') & (mdf['Type']=='Off-Ball')]['Dist'].mean()
                
                asr = sp_df[sp_df['Team']=='Red']['Speed'].mean()
                apr = sp_df[sp_df['Team']=='Red']['Poss'].mean()
                asw = sp_df[sp_df['Team']=='White']['Speed'].mean()
                apw = sp_df[sp_df['Team']=='White']['Poss'].mean()
                
                # KPI Display
                st.markdown("### 📊 Team Averages")
                k1, k2 = st.columns(2)
                k1.info(f"🔴 **Red:**\n- Dist: {atr:.1f}m\n- Off-Ball: {aro:.1f}m\n- Speed: {asr:.1f} km/h\n- Poss: {apr:.1f}s")
                k2.info(f"⚪ **White:**\n- Dist: {awt:.1f}m\n- Off-Ball: {awo:.1f}m\n- Speed: {asw:.1f} km/h\n- Poss: {apw:.1f}s")
                
                # Grafico Workload (c2 del blocco Spacing o nuova riga)
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=mdf, x='Player', y='Dist', hue='Type', palette={'Total':'gray', 'Off-Ball':'limegreen'}, ax=ax2)
                # Linee medie
                ax2.axhline(atr, c='darkred', ls='--', lw=1); ax2.axhline(aro, c='red', ls=':', lw=1.5)
                ax2.axhline(awt, c='darkblue', ls='--', lw=1); ax2.axhline(awo, c='blue', ls=':', lw=1.5)
                
                ax2.tick_params(axis='x', rotation=90)
                ax2.set_title("Workload (Meters)")
                ax2.legend(fontsize='x-small')
                ax2.grid(True, axis='y', alpha=0.3)
                c2.pyplot(fig2)
                
                # Nuova riga per Speed e Possesso
                r2c1, r2c2 = st.columns(2)
                
                # Grafico Possesso
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=sp_df, x='Player', y='Poss', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax3)
                ax3.tick_params(axis='x', rotation=90)
                ax3.set_title("Possession Time (s)")
                ax3.grid(True, axis='y', alpha=0.3)
                r2c1.pyplot(fig3)
                
                # Grafico Velocità
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=sp_df, x='Player', y='Speed', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax4)
                # Linee medie speed
                ax4.axhline(asr, c='darkred', ls='--', label='Avg R')
                ax4.axhline(asw, c='darkblue', ls='--', label='Avg W')
                
                ax4.tick_params(axis='x', rotation=90)
                ax4.set_title("Avg Speed (km/h)")
                ax4.grid(True, axis='y', alpha=0.3)
                ax4.legend(fontsize='small')
                r2c2.pyplot(fig4)