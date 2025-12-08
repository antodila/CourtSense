# 📌 GUIDA GITHUB - CourtSense

## ✅ File Creati

Nella cartella `CourtSense/` hai ora:

```
✅ .gitignore          ← Esclude: app_pixel.py, app_metri.py, datasets/, 
                         not_dataset/, tracking_data.csv, etc.

✅ README.md           ← Documentazione completa (ti leggerà GitHub)

✅ requirements.txt    ← Dipendenze Python

✅ setup_github.sh     ← Script setup (opzionale)

✅ app.py              ← Main app (con commenti completi)

✅ json_to_csv.py      ← Convertitore COCO → CSV
```

---

## 🚀 Procedura GitHub - 5 Minuti

### STEP 1: Apri GitHub e crea repository
1. Vai su https://github.com/new
2. **Repository name**: `CourtSense`
3. **Description**: "Tactical Dashboard for Sports Analysis - Streamlit App"
4. **Public** o **Private** (tua scelta)
5. ⚠️ **NON** selezionare "Initialize this repository with:"
6. Clicca "Create repository"

### STEP 2: Copia il comando da GitHub
Dopo aver creato il repo, GitHub mostra qualcosa tipo:

```
...or push an existing repository from the command line

git remote add origin https://github.com/YOUR_USERNAME/CourtSense.git
git branch -M main
git push -u origin main
```

### STEP 3: Esegui comandi nel terminale

Apri terminale nella cartella `CourtSense`:

```bash
# 1️⃣ Inizializza git (se non lo hai già fatto)
git init

# 2️⃣ Configura identità (una volta sola)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 3️⃣ Aggiungi tutti i file (rispetta .gitignore)
git add .

# 4️⃣ Commit iniziale
git commit -m "Initial commit: CourtSense Tactical Dashboard

- Main Streamlit application with comprehensive documentation
- COCO JSON to CSV conversion script
- Box-in-Box possession detection algorithm
- Hybrid Pixel-to-Meters conversion system
- Support for Voronoi, Convex Hull, and Heatmap analysis"

# 5️⃣ Aggiungi remote (COPIA DA GITHUB - cambia YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/CourtSense.git

# 6️⃣ Rinomina branch
git branch -M main

# 7️⃣ PUSH! 🚀
git push -u origin main
```

---

## ✨ Risultato Finale

Dopo il push, su GitHub vedrai:

```
🏀 CourtSense
   "Tactical Dashboard for Sports Analysis - Streamlit App"

📁 Files:
   ✅ .gitignore
   ✅ README.md
   ✅ requirements.txt
   ✅ app.py
   ✅ json_to_csv.py

📊 Commits: 1 initial commit
```

---

## 🔄 Per Future Updates

Ogni volta che modifichi un file:

```bash
# Aggiungi cambiamenti
git add app.py

# Commit con messaggio descrittivo
git commit -m "Add feature: Heatmap visualization"

# Push
git push
```

---

## 📋 Cosa NON sarà su GitHub (per .gitignore)

```
❌ app_pixel.py         ← Versione vecchia
❌ app_metri.py         ← Versione vecchia
❌ tracking_data.csv    ← Generato da json_to_csv.py
❌ datasets/            ← Troppo grande
❌ not_dataset/         ← Non usato
❌ venv/                ← Ambiente virtuale
❌ __pycache__/         ← Cache Python
❌ .streamlit/          ← Cache Streamlit
```

**Ma questi file non saranno persi!** Restano nel tuo computer.
Se qualcuno clona il repo, può rigenerare tutto con:

```bash
# Clona il repo
git clone https://github.com/YOUR_USERNAME/CourtSense.git

# Installa dipendenze
pip install -r requirements.txt

# Genera CSV dai dataset
python json_to_csv.py

# Lancia app
streamlit run app.py
```

---

## ⚙️ Configurazioni Utili (FACOLTATIVO)

### Ignora file già committed
Se hai commesso errore e vuoi escludere file già tracciati:

```bash
git rm --cached tracking_data.csv
git commit -m "Remove tracking_data.csv from tracking"
git push
```

### Verifica cosa verrà committato
```bash
git add .
git status
```

Mostra file pronti al commit (verificherai che app_pixel.py NON c'è 😉)

---

## 🎯 Checklist Finale

- [x] File `.gitignore` creato ✅
- [x] File `README.md` creato ✅
- [x] File `requirements.txt` creato ✅
- [x] Repository GitHub creato
- [x] Comandi eseguiti nel terminale
- [x] Git push completato

---

## ❓ Domande Comuni

**P: Posso includere i dataset su GitHub?**  
A: No (sono in .gitignore). Se necessario, usa servizi come:
   - Google Drive (condividere link)
   - Dropbox
   - GitHub Large File Storage (LFS)

**P: Come fanno altri a clonare il repo?**  
A: 
```bash
git clone https://github.com/YOUR_USERNAME/CourtSense.git
cd CourtSense
pip install -r requirements.txt
```

**P: Posso rendere il repo privato?**  
A: Sì, vai in Settings → Private (su GitHub)

---

## 🆘 Help

Se hai errori:

```bash
# Verifica status
git status

# Vedi log dei commit
git log

# Annulla ultimo commit (non ancora pushato)
git reset --soft HEAD~1
```

---

**Buon lavoro! 🚀 Il tuo CourtSense sarà presto su GitHub!**
