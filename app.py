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
IMAGES_FOLDER = 'datasets' # Root folder
RADAR_HEIGHT = 300
COURT_WIDTH = 3840 
COURT_HEIGHT = 2160
POSSESSION_THRESHOLD = 60 # Pixel (Box-in-Box)

# --- PARAMETRI FISICI ---
FPS = 30.0
REAL_WIDTH_M = 28.0
REAL_HEIGHT_M = 15.0
PX_TO_M = REAL_WIDTH_M / COURT_WIDTH 
MAX_PIXEL_STEP = 100 
SMOOTHING_WINDOW = 5
MIN_SPEED_THRESHOLD = 3.0
STREAMING_WIDTH = 800 # Larghezza video ridotta per fluidità cloud

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
    bx_c = m['bbox_x_b'] + m['bbox_w_b']/2
    by_c = m['bbox_y_b'] + m['bbox_h_b']/2
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
    
    dx = p_data['xp'].diff().fillna(0)
    dy = p_data['yp'].diff().fillna(0)
    step_px = np.sqrt(dx**2 + dy**2)
    step_px[step_px > MAX_PIXEL_STEP] = 0
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

# --- GRAFICI STATICI ---
def generate_static_voronoi(frame_data, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    players = frame_data[frame_data['team'].isin(['Red', 'White'])]
    if len(players) >= 4:
        points = players[['x_feet', 'y_feet']].values
        teams = players['team'].values
        dummy = np.array([[-200, -200], [4000, -200], [4000, 2400], [-200, 2400]])
        try:
            vor = Voronoi(np.vstack([points, dummy]))
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    c = 'red' if teams[i] == 'Red' else 'blue'
                    ax.add_patch(Polygon(vor.vertices[region], facecolor=c, alpha=0.4, edgecolor='white'))
        except: pass
    for t, c in [('Red', 'red'), ('White', 'blue')]:
        tp = players[players['team'] == t]
        ax.scatter(tp['x_feet'], tp['y_feet'], c=c, s=80, edgecolors='white', zorder=5)
    ax.set_xlim(0, COURT_WIDTH); ax.set_ylim(COURT_HEIGHT, 0); ax.axis('off')
    if title: ax.set_title(title, fontsize=15)
    return fig

def generate_static_hull(frame_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_mpl_court(ax)
    colors = {'Red': 'red', 'White': 'blue'}
    fill = {'Red': 'salmon', 'White': 'lightblue'}
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

# --- RENDERING VIDEO (BLINDATO) ---
def render_dual_view(f_id, df, quality_mode, highlight_id=None, is_possessor=False):
    # --- FIX 1: INIZIALIZZAZIONE VARIABILE ---
    is_holding_ball = False # Default
    
    fname_row = df[df['frame_id'] == f_id]
    if fname_row.empty: return None, None, 0, 0, 0, False
    fname = fname_row['frame_filename'].iloc[0]
    
    # --- FIX 2: PATH FINDING ROBUSTO ---
    possible_paths = []
    if 'image_path' in fname_row.columns:
        raw_p = fname_row['image_path'].iloc[0]
        possible_paths.append(raw_p)
        possible_paths.append(raw_p.replace('\\', '/'))
        possible_paths.append(raw_p.replace('/', '\\'))
        
    if 'action_id' in fname_row.columns:
        act = fname_row['action_id'].iloc[0]
        possible_paths.append(os.path.join('datasets', act, 'train', fname))
        
    possible_paths.append(os.path.join(IMAGES_FOLDER, fname))
    possible_paths.append(os.path.join('train', fname))
    
    frame_img_orig = None
    for p in possible_paths:
        if os.path.exists(p):
            frame_img_orig = cv2.imread(p)
            if frame_img_orig is not None: break
            
    if frame_img_orig is None: 
        h, w = 720, 1280
        frame_img_orig = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame_img_orig, "IMG NOT FOUND", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    target_w = 3840 if quality_mode == "Massima (4K)" else STREAMING_WIDTH
    scale = target_w / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    
    radar_img = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_img, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    red_c, white_c, ref_c = 0, 0, 0
    ball_row = frame_data[frame_data['team'] == 'Ball']

    for _, row in frame_data.iterrows():
        xm = row['x_feet'] * PX_TO_M
        ym = row['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)
        rx = int(xm * sx); ry = int(ym * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        
        t_str = str(row['team'])
        raw_c = str(row.get('raw_class', '')).lower()
        
        if t_str == 'Ball':
            cv2.circle(radar_img, (rx, ry), 8, (0, 165, 255), -1); cv2.circle(radar_img, (rx, ry), 10, (255, 255, 255), 1)
        elif 'Ref' in t_str or 'ref' in raw_c:
            cv2.circle(radar_img, (rx, ry), 5, (0, 255, 0), -1); ref_c += 1
        elif t_str in ['Red', 'White']:
            c = (0, 0, 255) if t_str == 'Red' else (255, 255, 255)
            if t_str == 'Red': red_c += 1
            else: white_c += 1
            cv2.circle(radar_img, (rx, ry), 6, c, -1)
            
            if highlight_id and row['player_unique_id'] == highlight_id:
                bx, by = int(row['bbox_x']*scale), int(row['bbox_y']*scale)
                bw, bh = int(row['bbox_w']*scale), int(row['bbox_h']*scale)
                
                col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
                cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 2)
                cv2.putText(frame_img, row['player_unique_id'], (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_box, 2)
                cv2.circle(radar_img, (rx, ry), 12, col_box, 2)
                
                if not ball_row.empty:
                    b = ball_row.iloc[0]; bcx, bcy = b['bbox_x']+b['bbox_w']/2, b['bbox_y']+b['bbox_h']/2
                    mw, mh = row['bbox_w']*0.05, row['bbox_h']*0.10
                    if (row['bbox_x']+mw < bcx < row['bbox_x']+row['bbox_w']-mw) and (row['bbox_y'] < bcy < row['bbox_y']+row['bbox_h']-mh):
                        is_holding_ball = True # Viene calcolato ma non usato per il return del possesso globale

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB), red_c, white_c, ref_c, is_possessor

# --- MAIN ---
st.set_page_config(page_title="CourtSense Dashboard", layout="wide")
st.title("🏀 CourtSense: Tactical Dashboard")

df_full = load_data()
if df_full is None: st.error("CSV non trovato."); st.stop()

st.sidebar.header("⚙️ Pannello di Controllo")
available_actions = sorted(df_full['action_id'].unique())
if not available_actions: st.error("Nessuna azione trovata."); st.stop()

idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("📂 Seleziona Azione:", available_actions, index=idx)

df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
ownership_table = get_possession_table(df)

unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

analysis_mode = st.sidebar.radio("Modalità:", ("🕹️ Navigazione (Manuale)", "▶️ Riproduzione (Auto)"))
quality_mode = st.sidebar.radio("Qualità:", ("Ottimizzata (HD)", "Massima (4K)"))
st.sidebar.markdown("---")
selected_player = st.sidebar.selectbox("Traccia Giocatore:", player_list)

col_main, col_side = st.columns([3, 1])
video_ph = col_main.empty(); radar_ph = col_side.empty(); stats_ph = col_side.empty()

def update_ui_elements(fid, dist_tot, dist_off, poss_frames, speed):
    is_owner = False
    try:
        if fid in ownership_table.index and ownership_table.loc[fid]['player_unique_id'] == selected_player: is_owner = True
    except: pass
    
    vid, rad, rc, wc, ref, hold = render_dual_view(fid, df, quality_mode, selected_player, is_owner)
    
    if vid is not None:
        success, buffer = cv2.imencode(".jpg", vid, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if success:
            video_ph.image(buffer.tobytes(), channels="RGB", width="stretch")
        else:
            video_ph.image(vid, channels="RGB", width="stretch")
            
        radar_ph.image(rad, channels="RGB", caption="Tactical Board (Meters)", width="stretch")
        icon = "🏀" if is_owner else ""
        stats_ph.markdown(f"""
        ### Frame: {fid}
        **Focus:** `{selected_player}` {icon}
        📏 **Dist:** {int(dist_tot)} m
        🏃 **Off-Ball:** {int(dist_off)} m
        ⚡ **Speed:** {speed} km/h
        ⏱️ **Poss:** {(poss_frames/FPS):.1f} s
        ---
        🔴 R:{rc} | ⚪ W:{wc}
        """)
    return hold

if analysis_mode == "🕹️ Navigazione (Manuale)":
    st.sidebar.markdown("---")
    sel_frame = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, selected_player, sel_frame, ownership_table)
    update_ui_elements(sel_frame, dt, do, pf, spd)
    
    st.markdown("---"); c1, c2 = st.columns(2)
    fdata = df[df['frame_id'] == sel_frame]
    if c1.button("📸 Voronoi"): c1.pyplot(generate_static_voronoi(fdata))
    if c2.button("🛡️ Convex Hull"): c2.pyplot(generate_static_hull(fdata))

else: # AUTO
    start, end = st.sidebar.select_slider("Clip:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    
    if st.sidebar.button("▶️ PLAY", type="primary"):
        frames = [f for f in unique_frames if start <= f <= end]
        cum_m = 0; cum_off_m = 0; prev_x_px = None; prev_y_px = None
        pos_buffer = [] 
        curr_poss_frames = 0
        
        for f_id in frames:
            t0 = time.time()
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            step_m = 0; speed_kmh = 0.0
            
            if not curr.empty:
                cx_px, cy_px = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buffer.append((cx_px, cy_px)); 
                if len(pos_buffer)>15: pos_buffer.pop(0)
                
                if len(pos_buffer)>=2:
                    cx_s = np.mean([p[0] for p in pos_buffer[-3:]]); cy_s = np.mean([p[1] for p in pos_buffer[-3:]])
                    if prev_x_px is not None:
                        raw_step_px = np.sqrt((cx_s-prev_x_px)**2 + (cy_s-prev_y_px)**2)
                        if raw_step_px < MAX_PIXEL_STEP:
                            step_m = raw_step_px * PX_TO_M
                            cum_m += step_m
                    prev_x_px, prev_y_px = cx_s, cy_s

                if len(pos_buffer)>=10:
                    d_px = np.sqrt((pos_buffer[-1][0]-pos_buffer[0][0])**2 + (pos_buffer[-1][1]-pos_buffer[0][1])**2)
                    tm = (len(pos_buffer)-1)/FPS
                    raw_spd = (d_px * PX_TO_M / tm) * 3.6
                    if raw_spd > MIN_SPEED_THRESHOLD: speed_kmh = min(raw_spd, 36.0)

            is_owner = False
            try:
                if f_id in ownership_table.index and ownership_table.loc[f_id]['player_unique_id'] == selected_player: is_owner = True
            except: pass
            
            if is_owner: curr_poss_frames += 1
            else: cum_off_m += step_m
            
            update_ui_elements(f_id, int(cum_m), int(cum_off_m), curr_poss_frames, round(speed_kmh, 1))
            time.sleep(max(0.04, (1.0/FPS) - (time.time()-t0)))

    st.markdown("---"); st.subheader("📈 Report Azione")
    c1, c2, c3 = st.columns(3)
    if c1.button("Genera Metriche"):
        with st.spinner("Calcolo..."):
            sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            duration_s = (end - start + 1) / FPS
            
            spac = []
            for f, g in players.groupby('frame_id'):
                for t in ['Red', 'White']:
                    tg = g[g['team']==t]
                    if len(tg)>=2:
                        dists_px = [np.linalg.norm(a-b) for a,b in combinations(tg[['x_feet','y_feet']].values, 2)]
                        dists_m = [d*PX_TO_M for d in dists_px]
                        spac.append({'f':f, 't':t, 'v':np.mean(dists_m)})
            if spac:
                sdf = pd.DataFrame(spac)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.lineplot(data=sdf, x='f', y='v', hue='t', palette={'Red':'red','White':'blue'}, ax=ax)
                mr = sdf[sdf['t']=='Red']['v'].mean(); mw = sdf[sdf['t']=='White']['v'].mean()
                ax.axhline(mr, c='darkred', ls='--', label=f"R:{mr:.1f}m"); ax.axhline(mw, c='darkblue', ls='--', label=f"W:{mw:.1f}m")
                ax.set_title("Avg Spacing (Meters)"); ax.legend(fontsize='small'); c1.pyplot(fig)
            
            moves = []; speed_poss = []
            own_sub = ownership_table[ownership_table.index.isin(sub['frame_id'].unique())]
            
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                g['xm'] = g['x_feet'].rolling(SMOOTHING_WINDOW).mean(); g['ym'] = g['y_feet'].rolling(SMOOTHING_WINDOW).mean()
                steps_px = np.sqrt(np.diff(g['xm'], prepend=g['xm'].iloc[0])**2 + np.diff(g['ym'], prepend=g['ym'].iloc[0])**2)
                steps_px = np.where(steps_px > MAX_PIXEL_STEP, 0, steps_px)
                steps_m = steps_px * PX_TO_M
                
                is_p = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                tot = np.nansum(steps_m); off = np.nansum(steps_m[~is_p])
                poss_s = is_p.sum()/FPS; avg_spd = (tot/duration_s)*3.6
                
                moves.append({'Player':pid, 'Dist':tot, 'Type':'Total', 'Team':g['team'].iloc[0]})
                moves.append({'Player':pid, 'Dist':off, 'Type':'Off-Ball', 'Team':g['team'].iloc[0]})
                speed_poss.append({'Player':pid, 'Team':g['team'].iloc[0], 'Speed':avg_spd, 'Poss':poss_s})
            
            if moves:
                mdf = pd.DataFrame(moves); spdf = pd.DataFrame(speed_poss)
                ar_t = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Total')]['Dist'].mean()
                ar_o = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
                awt = mdf[(mdf['Team']=='White')&(mdf['Type']=='Total')]['Dist'].mean()
                awo = mdf[(mdf['Team']=='White')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
                asr=spdf[spdf['Team']=='Red']['Speed'].mean(); apr=spdf[spdf['Team']=='Red']['Poss'].mean()
                asw=spdf[spdf['Team']=='White']['Speed'].mean(); apw=spdf[spdf['Team']=='White']['Poss'].mean()
                
                k1, k2 = st.columns(2)
                k1.info(f"🔴 **Red Avg:** Tot {ar_t:.1f}m | Off {ar_o:.1f}m | Spd {asr:.1f}km/h | Poss {apr:.1f}s")
                k2.info(f"⚪ **White Avg:** Tot {awt:.1f}m | Off {awo:.1f}m | Spd {asw:.1f}km/h | Poss {apw:.1f}s")
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=mdf, x='Player', y='Dist', hue='Type', palette={'Total':'gray', 'Off-Ball':'limegreen'}, ax=ax2)
                ax2.axhline(ar_t, c='darkred', ls='--'); ax2.axhline(ar_o, c='red', ls=':')
                ax2.axhline(awt, c='darkblue', ls='--'); ax2.axhline(awo, c='blue', ls=':')
                ax2.tick_params(axis='x', rotation=90); ax2.set_title("Workload (m)"); c1.pyplot(fig2)
                
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=spdf, x='Player', y='Poss', hue='Team', palette={'Red':'red','White':'blue'}, ax=ax3)
                ax3.tick_params(axis='x', rotation=90); ax3.set_title("Possession (s)"); c2.pyplot(fig3)
                
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=spdf, x='Player', y='Speed', hue='Team', palette={'Red':'red', 'White':'blue'}, ax=ax4)
                ax4.axhline(asr, c='darkred', ls='--'); ax4.axhline(asw, c='darkblue', ls='--')
                ax4.tick_params(axis='x', rotation=90); ax4.set_title("Avg Speed (km/h)"); c3.pyplot(fig4)

    if c2.button("GIF Voronoi"):
        frames_list = df[(df['frame_id'] >= start) & (df['frame_id'] <= end)]['frame_filename'].unique()
        bar = st.progress(0, "Rendering..."); tmp="tmp_v"; os.makedirs(tmp, exist_ok=True); files=[]
        try:
            for i, fn in enumerate(frames_list):
                bar.progress(int((i/len(frames_list))*90))
                fig = generate_static_voronoi(df[df['frame_filename']==fn], title=f"Frame {extract_frame_number(fn)}")
                p = os.path.join(tmp, f"{i:03d}.png"); fig.savefig(p, dpi=50); plt.close(fig); files.append(p)
            with imageio.get_writer("action_voronoi.gif", mode='I', duration=0.15) as w:
                for f in files: w.append_data(imageio.imread(f))
            bar.empty(); st.image("action_voronoi.gif")
        except Exception as e: st.error(str(e))
        finally: shutil.rmtree(tmp)

    if c3.button("Heatmap Azione"):
        sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
        colors_map = {'Red':'Reds', 'White':'Blues'}
        for t in ['Red', 'White']:
            st_t = sub[sub['team']==t]
            if not st_t.empty:
                fig, ax = plt.subplots(figsize=(5,3)); draw_mpl_court(ax)
                sns.kdeplot(x=st_t['x_feet'], y=st_t['y_feet'], fill=True, cmap=colors_map.get(t, 'Greys'), alpha=0.6, ax=ax)
                ax.set_xlim(0, COURT_WIDTH); ax.set_ylim(COURT_HEIGHT, 0); ax.axis('off'); c3.pyplot(fig)