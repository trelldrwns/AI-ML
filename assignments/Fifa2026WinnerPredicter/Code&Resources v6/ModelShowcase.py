import sys
import pandas as pd
import numpy as np
import joblib
import json
import random
import subprocess  # For running external scripts
import re          # For parsing script output
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFrame,
    QGridLayout, QGraphicsView, QGraphicsScene, QGraphicsTextItem,
    QSizePolicy, QDialog, QProgressDialog
)
from PyQt5.QtGui import QColor, QPalette, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF, QThread, pyqtSignal

# --- Configuration ---
COLOR_DARK_BG = "#393f50"
COLOR_MEDIUM_BLUE = "#435892"
COLOR_OFF_WHITE = "#f2eee7"
COLOR_LIGHT_GREEN = "#bacaa8"

# --- Button Styles ---
TEAM_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {COLOR_LIGHT_GREEN};
        color: {COLOR_DARK_BG};
        border-radius: 35px;
        font-size: 10pt;
        font-weight: bold;
        border: 2px solid {COLOR_LIGHT_GREEN};
    }}
    QPushButton:hover {{
        background-color: #d0e0c0;
    }}
"""
TEAM_BUTTON_SELECTED_STYLE = f"""
    QPushButton {{
        background-color: {COLOR_LIGHT_GREEN};
        color: {COLOR_DARK_BG};
        border-radius: 35px;
        font-size: 10pt;
        font-weight: bold;
        border: 3px solid #FFD700; /* Gold border for selection */
    }}
"""

# --- Model and Data Paths ---
RF_MODEL_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/rf_classifier_v3.joblib'
PREPROCESSOR_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/preprocessor_v3.joblib'
LABEL_ENCODER_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/label_encoder_v3.joblib'
TEAM_WIN_RATE_LOOKUP_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/team_win_rate_lookup.json'
KMEANS_DATA_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/final_team_data.csv'
KMEANS_MODEL_SCRIPT = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/initialkmeanclustering.py'


# --- K-Means Worker Thread (FIXED PARSER) ---
class KMeansWorker(QThread):
    """
    Runs the external K-Means script in a separate thread
    and parses its pretty-print output.
    """
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, script_path, python_executable):
        super().__init__()
        self.script_path = script_path
        self.python_executable = python_executable

    def run(self):
        stdout = ""
        try:
            # Run the python script
            process = subprocess.run(
                [self.python_executable, self.script_path],
                capture_output=True, text=True, check=True
            )
            stdout = process.stdout
            
            # --- 1. Parse the Cluster Summary (Pretty-Print) ---
            in_summary = False
            data_lines = []
            for line in stdout.splitlines():
                if line.strip() == "--- Cluster Profiles (Mean Values) ---":
                    in_summary = True
                    continue
                if in_summary:
                    # Stop when we hit the end of the table
                    if line.strip().startswith('['):
                        in_summary = False
                        break # <--- This was the missing logic
                    # Skip header lines ('... xg ...') and ('cluster ...')
                    if line.strip() and not line.strip().startswith('cluster') and not 'xg_plus_minus_per90' in line:
                        data_lines.append(line)
            
            if not data_lines:
                raise ValueError("Could not parse cluster summary data lines from script output.")

            best_cluster_id = -1
            max_metric = -float('inf')

            for line in data_lines:
                # --- THIS IS THE FIX ---
                # Check for the summary line *again* just in case it wasn't caught
                if line.strip().startswith('['):
                    continue 
                    
                parts = line.split()
                if not parts:
                    continue
                
                cluster_id = int(parts[0]) 
                metric_val = float(parts[1])
                
                if metric_val > max_metric:
                    max_metric = metric_val
                    best_cluster_id = cluster_id

            if best_cluster_id == -1:
                raise ValueError("Could not determine the best cluster.")

            # --- 2. Parse the Team Assignments by Cluster ---
            cluster_teams = {}
            current_cluster_id = -1
            in_team_assignments = False
            
            for line in stdout.splitlines():
                if line.strip() == "--- Team Assignments by Cluster ---":
                    in_team_assignments = True
                    continue
                
                if not in_team_assignments:
                    continue

                if line.startswith("=== Cluster"):
                    current_cluster_id = int(re.search(r'\d+', line).group())
                elif current_cluster_id != -1 and line.strip() and not line.startswith("==="):
                    teams = [team.strip() for team in line.split(', ')]
                    cluster_teams[current_cluster_id] = teams
                    current_cluster_id = -1

            top_teams = cluster_teams.get(best_cluster_id)
            
            if not top_teams:
                raise ValueError(f"Could not find team list for best cluster {best_cluster_id}.")

            self.finished.emit(top_teams)

        except subprocess.CalledProcessError as e:
            self.error.emit(f"K-Means script failed:\n{e.stderr}")
        except Exception as e:
            self.error.emit(f"Error parsing K-Means output: {e}\n\nFull output was:\n{stdout}")


# --- Loading Dialog (Unchanged) ---
class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setModal(True) 
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Running K-Means Model...\n\nThis may take a moment.\n\nPlease wait.", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14pt; padding: 20px;")
        self.layout.addWidget(self.label)
        self.setFixedSize(300, 150)


# --- GraphicsWidget for the Bracket (Unchanged) ---
class BracketView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setStyleSheet(f"background-color: {COLOR_OFF_WHITE}; border-radius: 8px;")
        
        self.line_pen = QPen(QColor(COLOR_DARK_BG), 2)
        self.team_font = QFont("Arial", 10)
        self.winner_font = QFont("Arial", 12, QFont.Bold)
        self.team_brush = QBrush(QColor(COLOR_DARK_BG))
        self.match_brush = QBrush(QColor(COLOR_LIGHT_GREEN))

    def clear_bracket(self):
        self.scene.clear()

    def draw_match(self, x, y1, y2, team1, team2, winner=None):
        box_width, box_height = 120, 25 
        
        self.scene.addRect(x, y1, box_width, box_height, self.line_pen, self.match_brush)
        self.scene.addRect(x, y2, box_width, box_height, self.line_pen, self.match_brush)
        
        text1 = QGraphicsTextItem(team1)
        text1.setFont(self.team_font)
        text1.setPos(x + 5, y1 + 3)
        self.scene.addItem(text1)
        
        text2 = QGraphicsTextItem(team2)
        text2.setFont(self.team_font)
        text2.setPos(x + 5, y2 + 3)
        self.scene.addItem(text2)
        
        mid_y = (y1 + y2 + box_height) / 2
        line_x = x + box_width
        
        self.scene.addLine(line_x, y1 + box_height / 2, line_x + 30, y1 + box_height / 2, self.line_pen)
        self.scene.addLine(line_x, y2 + box_height / 2, line_x + 30, y2 + box_height / 2, self.line_pen)
        self.scene.addLine(line_x + 30, y1 + box_height / 2, line_x + 30, y2 + box_height / 2, self.line_pen)
        self.scene.addLine(line_x + 30, mid_y, line_x + 60, mid_y, self.line_pen)
        
        if winner:
            winner_x = x + box_width + 60 
            winner_y = mid_y - box_height / 2
            self.scene.addRect(winner_x, winner_y, box_width, box_height, self.line_pen, self.match_brush)
            
            w_text = QGraphicsTextItem(winner)
            w_text.setFont(self.team_font)
            w_text.setPos(winner_x + 5, winner_y + 3)
            self.scene.addItem(w_text)
            
            if winner == team1:
                text1.setFont(QFont("Arial", 10, QFont.Bold))
            else:
                text2.setFont(QFont("Arial", 10, QFont.Bold))
                
    def draw_final(self, x, y, team1, team2, winner=None):
        box_width, box_height = 120, 25 
        
        self.scene.addRect(x, y, box_width, box_height, self.line_pen, self.match_brush)
        self.scene.addRect(x, y + box_height + 10, box_width, box_height, self.line_pen, self.match_brush)
        
        text1 = QGraphicsTextItem(team1)
        text1.setFont(self.team_font)
        text1.setPos(x + 5, y + 3)
        self.scene.addItem(text1)
        
        text2 = QGraphicsTextItem(team2)
        text2.setFont(self.team_font)
        text2.setPos(x + 5, y + box_height + 13)
        self.scene.addItem(text2)
        
        if winner:
            winner_x = x + box_width + 20
            winner_y = y + (box_height + 10) / 2
            self.scene.addRect(winner_x, winner_y, box_width + 40, box_height + 10, self.line_pen, QBrush(QColor(COLOR_LIGHT_GREEN)))
            
            w_text = QGraphicsTextItem(f"🏆 {winner} 🏆")
            w_text.setFont(self.winner_font)
            w_text.setPos(winner_x + 5, winner_y + 8) 
            self.scene.addItem(w_text)
            
            if winner == team1:
                text1.setFont(QFont("Arial", 10, QFont.Bold))
            else:
                text2.setFont(QFont("Arial", 10, QFont.Bold))

    def fit_view(self):
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


# --- Main Application ---
class FootballPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIFA Knockout Stage Simulator")
        self.setGeometry(100, 100, 1400, 900) 

        self.loaded_models = False
        self.rf_classifier = None
        self.preprocessor = None
        self.label_encoder = None
        self.team_win_rate_lookup = {}
        self.top_teams_pool = [] 
        
        self.selected_tournament_teams = []
        self.team_button_map = {} 
        
        self.kmeans_worker = None
        self.loading_dialog = LoadingDialog(self)

        self.init_ui()
        self.load_models_and_data() 

    def init_ui(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(COLOR_DARK_BG))
        self.setPalette(palette)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left Panel (Column 1) ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        controls_frame = QFrame(self)
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet(f"background-color: {COLOR_MEDIUM_BLUE}; border-radius: 8px;")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        
        title_controls = QLabel("Tournament Controls", controls_frame)
        title_controls.setStyleSheet(f"color: {COLOR_OFF_WHITE}; font-size: 16pt; font-weight: bold;")
        title_controls.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(title_controls)
        controls_layout.addSpacing(10)
        
        controls_layout.addWidget(QLabel("Selected Teams (Pick 8):", controls_frame, 
                                        styleSheet=f"color: {COLOR_OFF_WHITE}; font-size: 12pt;"))
        self.selected_teams_display = QTextEdit(controls_frame)
        self.selected_teams_display.setReadOnly(True)
        self.selected_teams_display.setStyleSheet(f"background-color: {COLOR_OFF_WHITE}; color: {COLOR_DARK_BG}; font-size: 10pt; border-radius: 5px;")
        self.selected_teams_display.setFixedHeight(150) 
        controls_layout.addWidget(self.selected_teams_display)
        
        self.auto_select_button = QPushButton("Randomize Bracket", controls_frame)
        self.auto_select_button.setStyleSheet(f"""
            QPushButton {{ background-color: {COLOR_OFF_WHITE}; color: {COLOR_DARK_BG}; 
                          font-size: 12pt; padding: 8px; border-radius: 5px; }}
            QPushButton:hover {{ background-color: #e0dcdc; }}
        """)
        self.auto_select_button.clicked.connect(self.auto_select_teams)
        controls_layout.addWidget(self.auto_select_button)
        
        self.start_tournament_button = QPushButton("Start Tournament", controls_frame)
        self.start_tournament_button.setStyleSheet(f"""
            QPushButton {{ background-color: {COLOR_LIGHT_GREEN}; color: {COLOR_DARK_BG}; 
                          font-size: 14pt; padding: 10px; border-radius: 5px; margin-top: 5px; }}
            QPushButton:hover {{ background-color: #d0e0c0; }}
        """)
        self.start_tournament_button.clicked.connect(self.start_tournament_simulation)
        controls_layout.addWidget(self.start_tournament_button)
        
        self.reset_button = QPushButton("Reset Selections", controls_frame)
        self.reset_button.setStyleSheet(f"""
            QPushButton {{ background-color: {COLOR_OFF_WHITE}; color: {COLOR_DARK_BG}; 
                          font-size: 12pt; padding: 8px; border-radius: 5px; margin-top: 5px; }}
            QPushButton:hover {{ background-color: #e0dcdc; }}
        """)
        self.reset_button.clicked.connect(self.reset_all)
        controls_layout.addWidget(self.reset_button)
        controls_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) 
        left_panel.addWidget(controls_frame)
        
        team_grid_frame = QFrame(self)
        team_grid_frame.setFrameShape(QFrame.StyledPanel)
        team_grid_frame.setStyleSheet(f"background-color: {COLOR_MEDIUM_BLUE}; border-radius: 8px;")
        
        team_grid_main_layout = QVBoxLayout(team_grid_frame) 
        team_grid_main_layout.setContentsMargins(15, 15, 15, 15)
        
        team_grid_label = QLabel("Team Pool (from K-Means):", team_grid_frame)
        team_grid_label.setStyleSheet(f"color: {COLOR_OFF_WHITE}; font-size: 12pt; margin-bottom: 10px;")
        team_grid_label.setAlignment(Qt.AlignCenter) 
        team_grid_main_layout.addWidget(team_grid_label)
        
        self.team_grid_layout = QGridLayout() 
        self.team_grid_layout.setSpacing(10)
        
        for i in range(12): 
            btn = QPushButton("...", team_grid_frame)
            btn.setFixedSize(70, 70) 
            btn.setStyleSheet(TEAM_BUTTON_STYLE)
            self.team_grid_layout.addWidget(btn, i // 4, i % 4)
            btn.hide() 
            
        team_grid_main_layout.addLayout(self.team_grid_layout) 
        team_grid_main_layout.addStretch() 
        
        team_grid_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_panel.addWidget(team_grid_frame)
        
        main_layout.addLayout(left_panel, 2) 

        # --- Right Panel (Column 2) ---
        right_main_panel = QVBoxLayout()
        right_main_panel.setSpacing(10)

        # --- Top-Right Panel (Log) ---
        info_dump_frame = QFrame(self)
        info_dump_frame.setFrameShape(QFrame.StyledPanel)
        info_dump_frame.setStyleSheet(f"background-color: {COLOR_OFF_WHITE}; border-radius: 8px;")
        info_dump_layout = QVBoxLayout(info_dump_frame)
        title_info = QLabel("Simulation Log", info_dump_frame)
        title_info.setStyleSheet(f"color: {COLOR_DARK_BG}; font-size: 16pt; font-weight: bold;")
        title_info.setAlignment(Qt.AlignCenter)
        info_dump_layout.addWidget(title_info)
        self.output_info = QTextEdit(info_dump_frame)
        self.output_info.setReadOnly(True)
        self.output_info.setStyleSheet(f"background-color: {COLOR_OFF_WHITE}; color: {COLOR_DARK_BG}; font-size: 10pt;")
        info_dump_layout.addWidget(self.output_info)
        
        info_dump_frame.setFixedHeight(250) 
        right_main_panel.addWidget(info_dump_frame) 

        # --- Bottom-Right Panel (Bracket) ---
        main_show_frame = QFrame(self)
        main_show_frame.setFrameShape(QFrame.StyledPanel)
        main_show_frame.setStyleSheet(f"background-color: {COLOR_DARK_BG}; border-radius: 8px;") # Frame bg
        main_show_layout = QVBoxLayout(main_show_frame)
        main_show_layout.setContentsMargins(10, 10, 10, 10)
        
        title_main_show = QLabel("Tournament Bracket", main_show_frame)
        title_main_show.setStyleSheet(f"color: {COLOR_OFF_WHITE}; font-size: 16pt; font-weight: bold;")
        title_main_show.setAlignment(Qt.AlignCenter)
        main_show_layout.addWidget(title_main_show)
        
        self.bracket_view = BracketView()
        main_show_layout.addWidget(self.bracket_view)
        
        main_show_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_main_panel.addWidget(main_show_frame) 

        main_layout.addLayout(right_main_panel, 5)

    def load_models_and_data(self):
        try:
            self.rf_classifier = joblib.load(RF_MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            with open(TEAM_WIN_RATE_LOOKUP_PATH, 'r') as f:
                self.team_win_rate_lookup = json.load(f)
            self.loaded_models = True
            self.output_info.append("Prediction models and lookup data loaded successfully.")
            
            self.run_kmeans_to_get_top_teams()
            
        except Exception as e:
            self.output_info.append(f"Error loading files: {e}")
            self.loaded_models = False

    def run_kmeans_to_get_top_teams(self):
        self.output_info.append("Starting K-Means script to find top teams...")
        self.loading_dialog.show() 
        
        self.kmeans_worker = KMeansWorker(KMEANS_MODEL_SCRIPT, sys.executable)
        self.kmeans_worker.finished.connect(self.on_kmeans_finished)
        self.kmeans_worker.error.connect(self.on_kmeans_error)
        self.kmeans_worker.start()

    def on_kmeans_finished(self, top_teams):
        self.loading_dialog.hide() 
        self.top_teams_pool = top_teams
        self.output_info.append(f"K-Means complete. Found {len(self.top_teams_pool)} elite teams.")
        self.update_team_buttons() 
        
        self.output_info.append(f"\nYour {len(self.top_teams_pool)} elite teams are loaded.\n\n"
                                "1. Click 8 teams in the order you want them to play (1 vs 2, 3 vs 4...).\n\n"
                                "2. Or, click 'Randomize Bracket'.")

    def on_kmeans_error(self, error_message):
        self.loading_dialog.hide()
        self.output_info.append(f"--- K-MEANS SCRIPT FAILED ---")
        self.output_info.append(error_message)
        self.output_info.append("Please check script and file paths.")
        self.auto_select_button.setEnabled(False)
        self.start_tournament_button.setEnabled(False)

    def update_team_buttons(self):
        # Clear old placeholder buttons
        for i in reversed(range(self.team_grid_layout.count())): 
            widget = self.team_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.team_button_map.clear()
        
        # Add new buttons from live data
        for i, team_name in enumerate(self.top_teams_pool):
            btn = QPushButton(team_name[:3].upper(), self)
            btn.setFixedSize(70, 70) 
            btn.setToolTip(team_name) 
            btn.setStyleSheet(TEAM_BUTTON_STYLE)
            btn.clicked.connect(lambda checked, name=team_name: self.team_button_clicked(name))
            
            row, col = divmod(i, 4) 
            self.team_grid_layout.addWidget(btn, row, col)
            self.team_button_map[team_name] = btn
            btn.show()
            
        # Add stretch to push buttons to the top
        self.team_grid_layout.setRowStretch(self.team_grid_layout.rowCount(), 1)

    def team_button_clicked(self, team_name):
        if team_name in self.selected_tournament_teams:
            self.selected_tournament_teams.remove(team_name)
            self.team_button_map[team_name].setStyleSheet(TEAM_BUTTON_STYLE) 
            self.output_info.append(f"Removed {team_name} from tournament.")
        else:
            if len(self.selected_tournament_teams) < 8:
                self.selected_tournament_teams.append(team_name)
                self.team_button_map[team_name].setStyleSheet(TEAM_BUTTON_SELECTED_STYLE) 
                self.output_info.append(f"Added {team_name} to tournament.")
            else:
                self.output_info.append(f"Cannot add team: You have already selected all 8 teams.")
        self.selected_teams_display.setText(
            f"Selected: {len(self.selected_tournament_teams)} / 8\n\n" + 
            "\n".join(self.selected_tournament_teams)
        )

    def auto_select_teams(self):
        self.reset_all() 
        if len(self.top_teams_pool) < 8:
            self.output_info.append("Randomize failed: Not 8 teams in the pool.")
            return
        self.output_info.append("Randomizing bracket...")
        
        # Select all 8 teams from the pool in a random order
        selected_teams = random.sample(self.top_teams_pool, 8)
        
        self.auto_select_list = selected_teams
        self.auto_select_timer = QTimer(self)
        self.auto_select_timer.timeout.connect(self.add_auto_select_team)
        self.auto_select_timer.start(150)

    def add_auto_select_team(self):
        if self.auto_select_list:
            team_to_add = self.auto_select_list.pop(0)
            self.team_button_clicked(team_to_add)
        else:
            self.auto_select_timer.stop()
            self.output_info.append("Bracket randomized. Click 'Start Tournament'.")

    def predict_match(self, team_a, team_b, is_knockout):
        if not self.loaded_models:
            return None
        try:
            team_a_rate = self.team_win_rate_lookup.get(team_a, 0.5) 
            team_b_rate = self.team_win_rate_lookup.get(team_b, 0.5)
            win_rate_diff = team_a_rate - team_b_rate
            knockout_stage = 1 if is_knockout else 0
            is_home_advantage = 0 
            input_data = pd.DataFrame({
                'Win_Rate_Difference': [win_rate_diff],
                'Knockout Stage': [knockout_stage],
                'Is_Home_Advantage': [is_home_advantage],
                'Home Team Name': [team_a],
                'Away Team Name': [team_b]
            })
            input_transformed = self.preprocessor.transform(input_data)
            probabilities = self.rf_classifier.predict_proba(input_transformed)[0]
            result = {
                'team_a': team_a,
                'team_b': team_b,
                'home_win_prob': probabilities[self.label_encoder.transform(['home team win'])[0]],
                'away_win_prob': probabilities[self.label_encoder.transform(['away team win'])[0]],
                'draw_prob': probabilities[self.label_encoder.transform(['draw'])[0]],
                'predicted_outcome': self.label_encoder.inverse_transform([probabilities.argmax()])[0]
            }
            return result
        except Exception as e:
            self.output_info.append(f"Error during prediction: {e}")
            return None

    def get_winner(self, prediction_result):
        if prediction_result['predicted_outcome'] == 'home team win':
            return prediction_result['team_a']
        elif prediction_result['predicted_outcome'] == 'away team win':
            return prediction_result['team_b']
        else:
            if prediction_result['home_win_prob'] > prediction_result['away_win_prob']:
                return prediction_result['team_a']
            else:
                return prediction_result['team_b']

    def start_tournament_simulation(self):
        if len(self.selected_tournament_teams) != 8:
            self.output_info.append("Tournament start failed: 8 teams not selected.")
            self.bracket_view.clear_bracket()
            return

        self.output_info.append("Starting 8-team knockout simulation...")
        self.bracket_view.clear_bracket()
        
        teams = self.selected_tournament_teams.copy()
        
        # --- Quarter-Finals ---
        self.output_info.append("--- QUARTER-FINALS ---")
        qf_winners = []
        qf_matches = [(teams[0], teams[1]), (teams[2], teams[3]), (teams[4], teams[5]), (teams[6], teams[7])]
        
        y_step = 90 
        qf_y_positions = [20, 20 + y_step, 20 + 2*y_step, 20 + 3*y_step]
        
        for i, (team_a, team_b) in enumerate(qf_matches):
            result = self.predict_match(team_a, team_b, is_knockout=True)
            winner = self.get_winner(result)
            qf_winners.append(winner)
            self.output_info.append(f"{team_a} vs {team_b} -> Winner: {winner}")
            
            y1 = qf_y_positions[i]
            y2 = y1 + 40 
            self.bracket_view.draw_match(20, y1, y2, team_a, team_b, winner)
        
        # --- Semi-Finals ---
        self.output_info.append("--- SEMI-FINALS ---")
        sf_winners = []
        sf_matches = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
        sf_y_positions = [ (qf_y_positions[0] + qf_y_positions[1] + 40) / 2 + 5,
                           (qf_y_positions[2] + qf_y_positions[3] + 40) / 2 + 5 ]

        for i, (team_a, team_b) in enumerate(sf_matches):
            result = self.predict_match(team_a, team_b, is_knockout=True)
            winner = self.get_winner(result)
            sf_winners.append(winner)
            self.output_info.append(f"{team_a} vs {team_b} -> Winner: {winner}")
            
            y1 = sf_y_positions[i]
            y2 = y1 + 40 
            self.bracket_view.draw_match(20 + 120 + 60, y1, y2, team_a, team_b, winner)

        # --- Final ---
        self.output_info.append("--- FINAL ---")
        final_match = (sf_winners[0], sf_winners[1])
        final_y_pos = (sf_y_positions[0] + sf_y_positions[1] + 40) / 2 + 5
        
        result = self.predict_match(final_match[0], final_match[1], is_knockout=True)
        final_winner = self.get_winner(result)
        self.output_info.append(f"{final_match[0]} vs {final_match[1]} -> Winner: {final_winner}")

        self.bracket_view.draw_final(20 + (120 + 60) * 2, final_y_pos, final_match[0], final_match[1], final_winner)
        self.output_info.append(f"🏆 TOURNAMENT CHAMPION: {final_winner} 🏆")
        
        self.bracket_view.fit_view()

    def reset_all(self):
        self.output_info.clear()
        self.bracket_view.clear_bracket() 
        
        self.selected_tournament_teams = []
        self.selected_teams_display.clear()
        
        for team_name, button in self.team_button_map.items():
            button.setStyleSheet(TEAM_BUTTON_STYLE)
            
        self.output_info.append("Selections reset. Ready to start a new tournament.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FootballPredictorApp()
    ex.show()
    sys.exit(app.exec_())
