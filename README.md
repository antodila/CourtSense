# 🏀 CourtSense - Tactical Dashboard

## Panoramica

**CourtSense** è un'applicazione **Streamlit** per l'analisi tattica e il tracking di sport su campo (basket, calcio, etc.). Utilizza video e dati di tracking per generare metriche avanzate come:

- 📏 **Distanza percorsa** (totale + off-ball in metri)
- ⚡ **Velocità media** (km/h)
- 🏀 **Possesso palla** (rilevamento automatico)
- 📊 **Spacing** (distanza media tra giocatori)
- 🔷 **Voronoi Diagram** (spazio controllato)
- 🛡️ **Convex Hull** (forma squadra)
- 🔥 **Heatmap** (densità movimento)

---

## 🎯 Caratteristiche Principali

### Sistema Ibrido Pixel ↔ Metri
- Calcoli in **PIXEL** (per robustezza su rilevamento video)
- Output in **METRI** (per confronto reale)
- Fattore di conversione: `1 pixel = 0.0073 m`

### Algoritmo Box-in-Box per Possesso
- Rileva chi ha la palla basato su sovrapposizione bounding box
- Risolve conflitti con distanza euclidea
- Buffer 3-frame per stabilità

### Doppia Visualizzazione
- **Frame video** con highlight giocatore e box
- **Radar tattico** con campo sovrapposto e giocatori in scala metrica

### Tre Modalità di Analisi
1. **🕹️ Navigazione Manuale**: Slider per frame singoli + Voronoi/Hull su richiesta
2. **▶️ Riproduzione Auto**: Simula video con accumulazione metriche
3. **📈 Report**: Statistiche aggregate (spacing, workload, velocità, possesso)

---

## 📋 Struttura File

```
CourtSense/
├── app.py                    # ✅ Main app (con commenti completi)
├── json_to_csv.py            # Convertitore COCO JSON → CSV
├── .gitignore                # Configurazione Git
├── README.md                 # Questo file
│
├── datasets/                 # ✅ Dataset principali (sarà in .gitignore)
│   ├── azione_01/
│   │   └── train/_annotations.coco.json
│   └── azione_02/
│       └── train/_annotations.coco.json
│
├── not_dataset/              # ✅ Dataset secondari (sarà in .gitignore)
│   ├── azione_03/
│   ├── azione_04/
│   └── azione_05/
│
├── app_pixel.py              # ⚠️ Versione vecchia (in .gitignore)
├── app_metri.py              # ⚠️ Versione vecchia (in .gitignore)
├── tracking_data.csv         # ⚠️ Generato dinamicamente (in .gitignore)
│
└── temp_voronoi_gif/         # 📁 Temp (in .gitignore)
```

---

## 🚀 Installazione

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/CourtSense.git
cd CourtSense
```

### 2. Crea Ambiente Virtuale
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Installa Dipendenze
```bash
pip install -r requirements.txt
```

---

## 📦 Dipendenze

```
streamlit>=1.28.0
pandas>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
imageio>=2.32.0
```

Genera il file con:
```bash
pip freeze > requirements.txt
```

---

## ⚙️ Setup Iniziale

### 1. Prepara i Dataset
I file JSON COCO devono trovarsi in:
```
datasets/azione_01/train/_annotations.coco.json
datasets/azione_02/train/_annotations.coco.json
...
```

### 2. Genera CSV da JSON
```bash
python json_to_csv.py
```

Questo crea `tracking_data.csv` con tutte le posizioni (frame-by-frame).

### 3. Avvia Streamlit
```bash
streamlit run app.py
```

Apri browser → `http://localhost:8501`

---

## 🎮 Come Usare

### Modalità Navigazione Manuale
1. Seleziona azione dal dropdown
2. Usa slider per esplorare frame singoli
3. Clicca "Voronoi" o "Convex Hull" per visualizzare

### Modalità Riproduzione Auto
1. Seleziona range di frame (start-end)
2. Regola FPS riproduzione
3. Premi **PLAY** per simulare video
4. Metriche si accumulano in tempo reale

### Report Azione
- **Genera Metriche**: Statistiche aggregate + grafici
- **GIF Voronoi**: Crea animazione spazio controllato
- **Heatmap**: Densità movimento per team

---

## 📊 Output Metriche

### Per Giocatore
| Metrica | Unità | Descrizione |
|---------|-------|------------|
| **Dist** | m | Distanza totale percorsa |
| **Off-Ball** | m | Movimento senza palla |
| **Speed** | km/h | Velocità media (ultimi 15 frame) |
| **Poss** | s | Tempo possesso palla |

### Per Squadra (Aggregate)
| Metrica | Descrizione |
|---------|------------|
| **Avg Spacing** | Distanza media intra-squadra |
| **Workload** | Distanza per giocatore (bar chart) |
| **Possession** | Tempo palla (bar chart) |
| **Avg Speed** | Velocità media (bar chart) |

---

## 🔧 Configurazione

Modifica `app.py` sezione CONFIGURAZIONE:

```python
# Campo reale (metri)
REAL_WIDTH_M = 28.0    # Basket: 28m, Calcio: 105m
REAL_HEIGHT_M = 15.0   # Basket: 15m, Calcio: 68m

# Video (pixel)
COURT_WIDTH = 3840     # 4K width
COURT_HEIGHT = 2160    # 4K height

# Filtri robustezza
MAX_PIXEL_STEP = 100   # Teletrasporto threshold
SMOOTHING_WINDOW = 5   # Media mobile frame
```

---

## 🐛 Troubleshooting

### CSV non trovato
```bash
# Verifica DATASETS_ROOT in json_to_csv.py
# Default: DATASETS_ROOT = 'datasets'
python json_to_csv.py
```

### Immagini non caricate
- Verifica path in CSV (`image_path` colonna)
- Prova: `IMAGES_FOLDER = '.'` in `app.py`

### Performance lenta
- Usa qualità "Ottimizzata (HD)" invece di "Massima (4K)"
- Riduci FPS slider in riproduzione auto

---

## 📁 .gitignore Spiegazione

| Pattern | Motivo |
|---------|--------|
| `tracking_data.csv` | Generato da `json_to_csv.py` |
| `datasets/` | Dataset troppo grandi (~GB) |
| `not_dataset/` | Dataset secondari non usati |
| `app_pixel.py` / `app_metri.py` | Versioni vecchie/test |
| `temp_voronoi_gif/` | Cartella temporanea |
| `.streamlit/` | Cache Streamlit |
| `venv/` | Ambiente virtuale locale |

**Include nel repo:**
- ✅ `app.py` (main app)
- ✅ `json_to_csv.py` (script conversione)
- ✅ `.gitignore` (questo file)
- ✅ `README.md` (documentazione)
- ✅ `requirements.txt` (dipendenze)

---

## 🔄 Workflow Git Consigliato

```bash
# 1. Init repository
git init
git add .gitignore README.md app.py json_to_csv.py requirements.txt
git commit -m "Initial commit: CourtSense Tactical Dashboard"

# 2. Aggiungi remote
git remote add origin https://github.com/yourusername/CourtSense.git
git branch -M main
git push -u origin main

# 3. Future updates
git add app.py
git commit -m "Add feature: X"
git push
```

---

## 👤 Autore

**CourtSense** - Analisi Tattica Sport  
Magistrale Sport Tech | Università di Trento

---

## 📝 Licenza

MIT License - Vedi LICENSE file

---

## 🤝 Contributi

Per problemi o suggerimenti, apri una **Issue** o **Pull Request**!
