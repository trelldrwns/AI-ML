# AI & ML Assignments Portfolio

This repository contains two distinct desktop applications demonstrating applied Artificial Intelligence and Machine Learning techniques. The projects cover graph-based pathfinding with Generative AI integration and sports analytics using statistical modeling.

## 1. CU Anveshan (BotBrain Campus Navigator)
**Directory:** `assignments/CU Anveshan BotBrain Campus Navigator/`

**Author:** Abhilash N S Reddy

A desktop application designed to provide intelligent pathfinding and location-based information for a university campus. It bridges traditional algorithms with modern Large Language Models (LLMs) for a natural user experience.

### Key Features
* **Interactive Visual Map:** A graphical interface where users can select start and end points to visualize routes.
* **Pathfinding Algorithms:** Implementation of four classic graph traversal algorithms allowing users to optimize for distance or steps:
    * Breadth-First Search (BFS)
    * Uniform Cost Search (UCS)
    * Dijkstra's Algorithm
    * A* Search
* **AI-Powered NLU:** Integrates a local LLM (**phi3:mini** via Ollama) to parse natural language queries like "Show me the way from the library to the sports complex".
* **RAG System:** A simple Retrieval-Augmented Generation system that fetches and displays specific building details (e.g., descriptions, operational details) upon arrival.

### Technical Architecture
* **GUI:** Built with **PyQt5** for rendering the map and controls.
* **AI Engine:** Uses **Ollama** running the `phi3:mini` model for local inference.
* **Concurrency:** Multithreaded `AIWorker` ensures the GUI remains responsive during LLM processing.

### Setup & Usage
1.  **Prerequisites:**
    * Python 3.x
    * `pip install PyQt5`
    * **Ollama** installed and running `phi3:mini` (`ollama pull phi3:mini`).
2.  **Run the Application:**
    Navigate to the project directory and execute:
    ```bash
    python main_app.py
    ```
   

---

## 2. FIFA World Cup 2026 Winner Predictor
**Directory:** `assignments/Fifa2026WinnerPredicter/`

A comprehensive machine learning application that analyzes team playstyles and predicts football match outcomes. It simulates the entire knockout stage of the 2026 World Cup using a two-phase AI approach.

### Methodology
* **Phase 1: Unsupervised Learning (Clustering)**
    * Uses **K-Means Clustering** to identify distinct "play styles" (Team DNA) based on metrics like Expected Goals (xG), Possession, and Shot-Creating Actions.
    * Determines natural groupings of teams to understand performance archetypes.
* **Phase 2: Supervised Learning (Prediction)**
    * Uses a **Random Forest Classifier** (ensemble of Decision Trees) to predict match outcomes (Home Win, Away Win, Draw).
    * **Feature Engineering:** Utilizes Win Rate Difference, Home Advantage, Knockout Stage flags, and One-Hot Encoded team identities.

### Application Features
* **Tournament Simulator:** Automatically simulates the bracket from the Round of 16 to the Final, advancing winners based on predicted probabilities.
* **Match Odds:** Displays calculated probabilities for Home Win, Draw, and Away Win for every simulated match.
* **Data Visualization Tab:** Interactive dashboard using `matplotlib` and `seaborn` to visualize team stats, heatmaps, and cluster data.

### Setup & Usage
1.  **Requirements:**
    * Python 3.x
    * `pandas`, `numpy`, `joblib`, `scikit-learn`
    * `PyQt5`, `matplotlib`, `seaborn`
2.  **Configuration:**
    * Ensure the model paths in `Final_v7.py` (e.g., `RF_MODEL_PATH`) point to the correct location of your `.joblib` and `.csv` files.
3.  **Run the Application:**
    ```bash
    python Final_v7.py
    ```
