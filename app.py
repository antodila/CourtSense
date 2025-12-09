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
import shutil

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
# Risoluzione streaming live
STREAMING_WIDTH = 800 

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
    # Campo in PIXEL per Heatmap/Voronoi Statici
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
    off_ball_m = off_ball_px * PX_TO_M
    poss_frames = is_mine_buf.sum()
    
    return int(total_dist_m), int(off_ball_m), round(speed_kmh, 1), poss_frames

# --- FUNZIONI GRAFICHE ---
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
        
    for _, r in players.iterrows():
        c = 'red' if r['team']=='Red' else 'blue'
        ax.scatter(r['x_feet'], r['y_feet'], c=c, s=80, edgecolors='white', zorder=5)
    
    # PALLA NELLA GIF
    ball = frame_data[frame_data['team'] == 'Ball']
    if not ball.empty:
        ax.scatter(ball['x_feet'], ball['y_feet'], c='orange', s=180, edgecolors='black', linewidth=1.5, zorder=10)

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

# --- RENDERING NBA STYLE (OVERLAY) ---
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

    # 1. Resize Video
    scale = target_width / frame_img_orig.shape[1]
    frame_img = cv2.resize(frame_img_orig, (0, 0), fx=scale, fy=scale)
    H, W = frame_img.shape[:2]
    
    # 2. Genera Radar
    radar_base = np.zeros((RADAR_HEIGHT, 600, 3), dtype=np.uint8)
    draw_radar_court(radar_base, 600, RADAR_HEIGHT)
    
    frame_data = df[df['frame_id'] == f_id]
    sx, sy = 600 / REAL_WIDTH_M, RADAR_HEIGHT / REAL_HEIGHT_M
    
    for _, row in frame_data.iterrows():
        xm = row['x_feet'] * PX_TO_M; ym = row['y_feet'] * (REAL_HEIGHT_M / COURT_HEIGHT)
        rx = int(xm * sx); ry = int(ym * sy)
        rx = max(0, min(rx, 600-1)); ry = max(0, min(ry, RADAR_HEIGHT-1))
        
        t = str(row['team'])
        col = (200,200,200)
        if t=='Red': col=(0,0,255)
        elif t=='White': col=(255,255,255)
        elif t=='Ball': col=(0,165,255)
        
        rad = 5 if t!='Ball' else 7
        cv2.circle(radar_base, (rx, ry), rad, col, -1)
        
        if row['player_unique_id'] == highlight_id:
            bx = int(row['bbox_x']*scale); by = int(row['bbox_y']*scale)
            bw = int(row['bbox_w']*scale); bh = int(row['bbox_h']*scale)
            col_box = (0, 165, 255) if is_possessor else (0, 255, 255)
            cv2.rectangle(frame_img, (bx, by), (bx+bw, by+bh), col_box, 2)
            cv2.circle(radar_base, (rx, ry), 10, col_box, 2)

    # 3. OVERLAY RADAR (Basso Destra)
    mini_w = int(W * 0.25)
    mini_h = int(mini_w * (RADAR_HEIGHT/600))
    radar_mini = cv2.resize(radar_base, (mini_w, mini_h))
    
    margin = 20
    y1 = H - mini_h - margin; y2 = H - margin
    x1 = W - mini_w - margin; x2 = W - margin
    
    if y1 > 0 and x1 > 0:
        roi = frame_img[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi, 0.3, radar_mini, 0.7, 0)
        frame_img[y1:y2, x1:x2] = blended
        cv2.rectangle(frame_img, (x1, y1), (x2, y2), (255,255,255), 1)

    # 4. STATS BOX (Alto Sinistra)
    if stats:
        dist, off, spd, poss = stats
        cv2.rectangle(frame_img, (20, 20), (300, 130), (0,0,0), -1)
        
        head_col = (0, 255, 255)
        txt_col = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        icon = " (BALL)" if is_possessor else ""
        cv2.putText(frame_img, f"{highlight_id}{icon}", (30, 50), font, 0.6, head_col, 2)
        cv2.putText(frame_img, f"DIST: {dist} m", (30, 75), font, 0.5, txt_col, 1)
        cv2.putText(frame_img, f"SPEED: {spd} km/h", (30, 95), font, 0.5, txt_col, 1)
        cv2.putText(frame_img, f"POSS: {poss:.1f} s", (30, 115), font, 0.5, txt_col, 1)

    return cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

# --- MAIN ---
st.set_page_config(page_title="CourtSense", layout="wide")
st.title("🏀 CourtSense: Broadcast Analytics")

df_full = load_data()
if df_full is None: st.error("CSV non trovato."); st.stop()

st.sidebar.header("Configurazione")
available_actions = sorted(df_full['action_id'].unique())
idx = available_actions.index('out13') if 'out13' in available_actions else 0
selected_action = st.sidebar.selectbox("Azione:", available_actions, index=idx)

df = df_full[df_full['action_id'] == selected_action].copy().sort_values('frame_id')
ownership_table = get_possession_table(df)

unique_frames = df['frame_id'].unique()
min_f, max_f = int(min(unique_frames)), int(max(unique_frames))
player_list = sorted([p for p in df['player_unique_id'].unique() if "Ball" not in p and "Ref" not in p])

analysis_mode = st.sidebar.radio("Modalità:", ("Navigazione", "Genera Video"))
selected_player = st.sidebar.selectbox("Giocatore:", player_list)

preview_ph = st.empty()

if analysis_mode == "Navigazione":
    sel_frame = st.sidebar.slider("Frame:", min_f, max_f, min_f)
    dt, do, spd, pf = calculate_advanced_stats_hybrid(df, selected_player, sel_frame, ownership_table)
    
    is_owner = False
    try:
        if sel_frame in ownership_table.index and ownership_table.loc[sel_frame]['player_unique_id'] == selected_player: is_owner = True
    except: pass
    
    # Preview (light)
    img = render_nba_style(sel_frame, df, 800, selected_player, is_owner, (dt, do, spd, pf/FPS))
    if img is not None:
        preview_ph.image(img, channels="RGB", use_container_width=True)

    st.markdown("---"); c1, c2 = st.columns(2)
    fdata = df[df['frame_id'] == sel_frame]
    if c1.button("📸 Voronoi"): c1.pyplot(generate_static_voronoi(fdata))
    if c2.button("🛡️ Convex Hull"): c2.pyplot(generate_static_hull(fdata))

else: # GENERAZIONE VIDEO
    st.info("Genera un video MP4 fluido in alta definizione (1280p).")
    start, end = st.sidebar.select_slider("Clip Range:", options=unique_frames, value=(min_f, min(min_f+40, max_f)))
    
    if st.sidebar.button("🎥 GENERA VIDEO"):
        frames = [f for f in unique_frames if start <= f <= end]
        output_file = "analysis_broadcast.mp4"
        prog_bar = st.progress(0, "Rendering...")
        
        cum_m = 0; cum_off = 0; pos_buf = []; poss_c = 0
        video_frames = []
        
        for i, f_id in enumerate(frames):
            prog_bar.progress(int(i/len(frames)*100))
            
            curr = df[(df['frame_id']==f_id) & (df['player_unique_id']==selected_player)]
            spd = 0.0
            if not curr.empty:
                cx, cy = curr.iloc[0]['x_feet'], curr.iloc[0]['y_feet']
                pos_buf.append((cx, cy)); 
                if len(pos_buf)>15: pos_buf.pop(0)
                if len(pos_buf)>=2:
                    step_px = np.sqrt((cx-pos_buf[-2][0])**2 + (cy-pos_buf[-2][1])**2)
                    if step_px < MAX_PIXEL_STEP: 
                        step_m = step_px * PX_TO_M
                        cum_m += step_m
                if len(pos_buf)>=10:
                     d_px = np.sqrt((pos_buf[-1][0]-pos_buf[0][0])**2 + (pos_buf[-1][1]-pos_buf[0][1])**2)
                     tm = (len(pos_buf)-1)/FPS
                     raw_spd = (d_px * PX_TO_M / tm) * 3.6
                     if raw_spd > MIN_SPEED_THRESHOLD: spd = min(raw_spd, 32.0)

            is_owner = False
            try:
                if f_id in ownership_table.index and ownership_table.loc[f_id]['player_unique_id'] == selected_player: is_owner = True
            except: pass
            
            if is_owner: poss_c += 1
            else: cum_off += (step_m if 'step_m' in locals() else 0)
            
            # Render HD
            img = render_nba_style(f_id, df, 1280, selected_player, is_owner, (int(cum_m), int(cum_off), round(spd, 1), poss_c/FPS))
            if img is not None: video_frames.append(img)
            
        prog_bar.progress(95, "Salvataggio MP4...")
        imageio.mimwrite(output_file, video_frames, fps=30, macro_block_size=1)
        
        prog_bar.empty()
        st.success("Video Pronto!")
        st.video(output_file)

    st.markdown("---"); st.subheader("📈 Report")
    if st.button("Genera Metriche"):
        with st.spinner("Calcolo..."):
            sub = df[(df['frame_id']>=start) & (df['frame_id']<=end)]
            players = sub[sub['team'].isin(['Red', 'White'])]
            duration_s = (end - start + 1) / FPS
            
            # --- 1. SPACING ---
            spac = []
            for f, g in players.groupby('frame_id'):
                for t in ['Red', 'White']:
                    tg = g[g['team']==t]
                    if len(tg)>=2:
                        d = [np.linalg.norm(a-b) for a,b in combinations(tg[['x_feet','y_feet']].values, 2)]
                        spac.append({'f':f, 't':t, 'v':np.mean(d)*PX_TO_M})
            if spac:
                sdf = pd.DataFrame(spac)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.lineplot(data=sdf, x='f', y='v', hue='t', palette={'Red':'red','White':'blue'}, ax=ax)
                mr = sdf[sdf['t']=='Red']['v'].mean(); mw = sdf[sdf['t']=='White']['v'].mean()
                ax.axhline(mr, c='darkred', ls='--', label=f"R:{mr:.1f}m")
                ax.axhline(mw, c='darkblue', ls='--', label=f"W:{mw:.1f}m")
                ax.set_title("Avg Spacing (Meters)"); ax.legend(fontsize='small'); preview_ph.pyplot(fig) # Show in main

            # --- 2. WORKLOAD, SPEED & POSS ---
            moves = []; speed_poss = []
            own_sub = ownership_table[ownership_table.index.isin(sub['frame_id'].unique())]
            for pid, g in players.groupby('player_unique_id'):
                g = g.sort_values('frame_id')
                steps_px = np.sqrt(np.diff(g['x_feet'], prepend=g['x_feet'].iloc[0])**2 + np.diff(g['y_feet'], prepend=g['y_feet'].iloc[0])**2)
                steps_px = np.where(steps_px > MAX_PIXEL_STEP, 0, steps_px)
                steps_m = steps_px * PX_TO_M
                
                is_p = g['frame_id'].isin(own_sub[own_sub['player_unique_id'] == pid].index).values
                tot = np.nansum(steps_m); off = np.nansum(steps_m[~is_p])
                poss_s = is_p.sum()/FPS; avg_spd = (tot/duration_s)*3.6
                
                moves.append({'Player':pid, 'Dist':tot, 'Type':'Total', 'Team':g['team'].iloc[0]})
                moves.append({'Player':pid, 'Dist':off, 'Type':'Off-Ball', 'Team':g['team'].iloc[0]})
                speed_poss.append({'Player':pid, 'Team':g['team'].iloc[0], 'Speed':avg_spd, 'Poss':poss_s})
            
            mdf = pd.DataFrame(moves); spdf = pd.DataFrame(speed_poss)
            
            # KPI
            atr = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Total')]['Dist'].mean()
            aro = mdf[(mdf['Team']=='Red')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
            awt = mdf[(mdf['Team']=='White')&(mdf['Type']=='Total')]['Dist'].mean()
            awo = mdf[(mdf['Team']=='White')&(mdf['Type']=='Off-Ball')]['Dist'].mean()
            asr=spdf[spdf['Team']=='Red']['Speed'].mean(); apr=spdf[spdf['Team']=='Red']['Poss'].mean()
            asw=spdf[spdf['Team']=='White']['Speed'].mean(); apw=spdf[spdf['Team']=='White']['Poss'].mean()
            
            k1, k2 = st.columns(2)
            k1.info(f"🔴 **Red Avg:** Tot {atr:.1f}m | Off {aro:.1f}m | Spd {asr:.1f}km/h | Poss {apr:.1f}s")
            k2.info(f"⚪ **White Avg:** Tot {awt:.1f}m | Off {awo:.1f}m | Spd {asw:.1f}km/h | Poss {apw:.1f}s")

            # Charts
            c1, c2 = st.columns(2)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.barplot(data=mdf, x='Player', y='Dist', hue='Type', ax=ax2)
            ax2.axhline(atr, c='darkred', ls='--'); ax2.axhline(awt, c='darkblue', ls='--')
            ax2.tick_params(axis='x', rotation=90); ax2.set_title("Workload (m)"); c1.pyplot(fig2)
            
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=spdf, x='Player', y='Poss', hue='Team', ax=ax3)
            ax3.tick_params(axis='x', rotation=90); ax3.set_title("Possession (s)"); c2.pyplot(fig3)
            
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=spdf, x='Player', y='Speed', hue='Team', ax=ax4)
            ax4.tick_params(axis='x', rotation=90); ax4.set_title("Avg Speed (km/h)"); st.pyplot(fig4)
            
            # --- GIF VORONOI ---
            st.markdown("### GIF Voronoi")
            frames_list = sub['frame_filename'].unique()
            if len(frames_list) > 0:
                bar = st.progress(0, "Rendering GIF..."); tmp="tmp_v"; os.makedirs(tmp, exist_ok=True); files=[]
                try:
                    for i, fn in enumerate(frames_list):
                        bar.progress(int((i/len(frames_list))*90))
                        fig = generate_static_voronoi(df[df['frame_filename']==fn], title=f"Frame {extract_frame_number(fn)}")
                        p = os.path.join(tmp, f"{i:03d}.png"); fig.savefig(p, dpi=60, bbox_inches='tight'); plt.close(fig); files.append(p)
                    
                    with imageio.get_writer("action_voronoi.gif", mode='I', duration=0.15, loop=0) as w:
                        for f in files: w.append_data(imageio.imread(f))
                    bar.empty(); st.image("action_voronoi.gif")
                except Exception as e: st.error(f"Errore GIF: {str(e)}")
                finally: shutil.rmtree(tmp)
            
            # --- HEATMAP ---
            st.markdown("### Heatmap")
            for t in ['Red', 'White']:
                st_t = sub[sub['team']==t]
                if not st_t.empty:
                    fig, ax = plt.subplots(figsize=(5,3)); draw_mpl_court(ax)
                    sns.kdeplot(x=st_t['x_feet'], y=st_t['y_feet'], fill=True, cmap='Reds' if t=='Red' else 'Blues', alpha=0.6, ax=ax)
                    ax.set_xlim(0, COURT_WIDTH); ax.set_ylim(COURT_HEIGHT, 0); ax.axis('off'); st.pyplot(fig)