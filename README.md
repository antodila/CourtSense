# 🏟️ CourtSense | Sport Tech Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://courtsense.streamlit.app/)

**CourtSense** is an advanced analytics dashboard for Football Tracking Data, developed as part of the **Sport Tech 2025-26** course. 

The application provides interactive insights into player performance, focusing on dynamic velocity smoothing and game phase recognition.

## 🚀 Live Demo
Access the web application directly here:  
👉 **[https://courtsense.streamlit.app/](https://courtsense.streamlit.app/)**

---

## 📊 Key Features

* **Savitzky-Golay Smoothing:** Implementation of advanced signal processing filters to calculate **Dynamic Velocity** from noisy raw tracking data.
* **Game Phases Analysis:** Automatic segmentation of the match into distinct phases for tactical evaluation.
* **Interactive Visualization:** Built with **Plotly** to allow zooming, panning, and detailed inspection of player metrics.
* **Team & Player Filtering:** Dynamic selection of teams ("Red" vs "White") and individual players.

---

## 📂 Repository Structure

This repository contains the source code and datasets used for the application:

* `app.py`: The main application script (Streamlit entry point).
* `json_to_csv.py`: Data engineering script used to parse raw JSON tracking logs into structured CSV format.
* `tracking_data.csv`: The processed dataset used by the live application.
* `README.md`: Project documentation.


---

**Author:** Antonio Di Lauro
**Course:** Information Engineering - Sport Tech Course