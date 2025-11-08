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
    QSizePolicy, QDialog, QTabWidget, QTableView, QScrollArea,
    QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import (
    QColor, QPalette, QPen, QBrush, QFont,
    QStandardItemModel, QStandardItem, QPainter, QPixmap
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QThread, pyqtSignal

# --- NEW: Matplotlib Imports ---
import matplotlib
matplotlib.use('Qt5Agg') # Set the backend for PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


# --- Color Theme Dictionaries (Unchanged) ---
DARK_THEME = {
    "WINDOW_BG": "#1e2025",
    "CONTENT_BG": "#2c2e36",
    "ACCENT_BLUE": "#4a69bd",
    "PRIMARY_TEXT": "#e1e1e1",
    "HIGHLIGHT_GREEN": "#6be9a0",
    "BUTTON_TEXT": "#1e2025" 
}
LIGHT_THEME = {
    "WINDOW_BG": "#e1e1e1",
    "CONTENT_BG": "#f0f0f0",
    "ACCENT_BLUE": "#4a69bd",
    "PRIMARY_TEXT": "#1e2025",
    "HIGHLIGHT_GREEN": "#6be9a0",
    "BUTTON_TEXT": "#1e2025"
}

# --- Model and Data Paths ---
RF_MODEL_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/rf_classifier_v3.joblib'
PREPROCESSOR_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/preprocessor_v3.joblib'
LABEL_ENCODER_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/label_encoder_v3.joblib'
TEAM_WIN_RATE_LOOKUP_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/team_win_rate_lookup.json'
KMEANS_DATA_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/final_team_data.csv'
KMEANS_MODEL_SCRIPT = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/initialkmeanclustering.py'
CLUSTER_IMAGE_PATH = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/v4/backenddata/cluster_visualization.png'


# --- K-Means Worker Thread ---
class KMeansWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, script_path, python_executable):
        super().__init__()
        self.script_path = script_path
        self.python_executable = python_executable

    def run(self):
        stdout = ""
        try:
            process = subprocess.run(
                [self.python_executable, self.script_path],
                capture_output=True, text=True, check=True
            )
            stdout = process.stdout
            
            in_summary = False
            data_lines = []
            found_data = False
            
            for line in stdout.splitlines():
                if line.strip() == "--- Cluster Profiles (Mean Values) ---":
                    in_summary = True
                    continue
                
                if in_summary:
                    if (not line.strip() or line.strip().startswith('[')) and found_data:
                        in_summary = False
                        break 
                        
                    if line.strip() and not line.strip().startswith('cluster') and 'xg_plus_minus_per90' not in line:
                        data_lines.append(line)
                        found_data = True 
            
            if not data_lines:
                raise ValueError("Could not parse cluster summary data lines from script output.")

            best_cluster_id = -1
            max_metric = -float('inf')

            for line in data_lines:
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


# --- Loading Dialog ---
class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setModal(True) 
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Running K-Means Model...\n\nThis may take a moment.\n\nPlease wait.", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.setFixedSize(300, 150)
    def set_theme(self, theme):
        self.setStyleSheet(f"background-color: {theme['CONTENT_BG']};")
        self.label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 14pt; padding: 20px;")


# --- GraphicsWidget for the Bracket ---
class BracketView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.set_theme(DARK_THEME) 
    def set_theme(self, theme):
        self.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border-radius: 8px;")
        self.line_pen = QPen(QColor(theme['PRIMARY_TEXT']), 2)
        self.team_font = QFont("Arial", 10)
        self.winner_font = QFont("Arial", 12, QFont.Bold)
        self.text_brush = QBrush(QColor(theme['PRIMARY_TEXT']))
        self.match_brush = QBrush(QColor(theme['HIGHLIGHT_GREEN']))
        self.match_text_brush = QBrush(QColor(theme['BUTTON_TEXT'])) 
        self.winner_box_brush = QBrush(QColor(theme['ACCENT_BLUE']))
        self.winner_text_brush = QBrush(QColor(theme['PRIMARY_TEXT']))
        if self.scene.items():
            pass 
    def clear_bracket(self):
        self.scene.clear()
    def draw_match(self, x, y1, y2, team1, team2, winner=None):
        box_width, box_height = 120, 25 
        self.scene.addRect(x, y1, box_width, box_height, self.line_pen, self.match_brush)
        self.scene.addRect(x, y2, box_width, box_height, self.line_pen, self.match_brush)
        text1 = QGraphicsTextItem(team1)
        text1.setFont(self.team_font)
        text1.setDefaultTextColor(QColor(self.match_text_brush.color())) 
        text1.setPos(x + 5, y1 + 3)
        self.scene.addItem(text1)
        text2 = QGraphicsTextItem(team2)
        text2.setFont(self.team_font)
        text2.setDefaultTextColor(QColor(self.match_text_brush.color())) 
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
            w_text.setDefaultTextColor(QColor(self.match_text_brush.color()))
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
        text1.setDefaultTextColor(QColor(self.match_text_brush.color()))
        text1.setPos(x + 5, y + 3)
        self.scene.addItem(text1)
        text2 = QGraphicsTextItem(team2)
        text2.setFont(self.team_font)
        text2.setDefaultTextColor(QColor(self.match_text_brush.color()))
        text2.setPos(x + 5, y + box_height + 13)
        self.scene.addItem(text2)
        if winner:
            winner_x = x + box_width + 20
            winner_y = y + (box_height + 10) / 2
            self.scene.addRect(winner_x, winner_y, box_width + 40, box_height + 10, self.line_pen, self.winner_box_brush)
            w_text = QGraphicsTextItem(f"🏆 {winner} 🏆")
            w_text.setFont(self.winner_font)
            w_text.setDefaultTextColor(QColor(self.winner_text_brush.color()))
            w_text.setPos(winner_x + 5, winner_y + 8) 
            self.scene.addItem(w_text)
            if winner == team1:
                text1.setFont(QFont("Arial", 10, QFont.Bold))
            else:
                text2.setFont(QFont("Arial", 10, QFont.Bold))
    def fit_view(self):
        try:
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        except Exception:
            pass 


# --- Odds Widgets ---
class OddsBarWidget(QWidget):
    def __init__(self, prob_a, prob_b, prob_draw, theme):
        super().__init__()
        self.prob_a = prob_a
        self.prob_b = prob_b
        self.prob_draw = prob_draw
        self.theme = theme
        self.setFixedHeight(20)
    def set_theme(self, theme):
        self.theme = theme
        self.update() 
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        total_width = self.width()
        color_a = QColor(self.theme['ACCENT_BLUE'])
        color_b = QColor(self.theme['HIGHLIGHT_GREEN'])
        color_draw = QColor("#888888") 
        width_a = int(total_width * self.prob_a)
        width_draw = int(total_width * self.prob_draw)
        width_b = total_width - width_a - width_draw 
        painter.fillRect(0, 0, width_a, self.height(), color_a)
        painter.fillRect(width_a, 0, width_draw, self.height(), color_draw)
        painter.fillRect(width_a + width_draw, 0, width_b, self.height(), color_b)

class MatchOddsWidget(QFrame):
    def __init__(self, title, result, theme):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.team_a = result['team_a']
        self.team_b = result['team_b']
        self.prob_a = result['home_win_prob']
        self.prob_b = result['away_win_prob']
        self.prob_draw = result['draw_prob']
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        self.title_label = QLabel(title)
        main_layout.addWidget(self.title_label)
        self.match_label = QLabel(f"<b>{self.team_a}</b> vs. <b>{self.team_b}</b>")
        main_layout.addWidget(self.match_label)
        self.bar = OddsBarWidget(self.prob_a, self.prob_b, self.prob_draw, theme)
        main_layout.addWidget(self.bar)
        legend_layout = QHBoxLayout()
        self.label_a = QLabel(f"{self.team_a} Win: {self.prob_a*100:.1f}%")
        self.label_draw = QLabel(f"Draw: {self.prob_draw*100:.1f}%")
        self.label_b = QLabel(f"{self.team_b} Win: {self.prob_b*100:.1f}%")
        legend_layout.addWidget(self.label_a, alignment=Qt.AlignLeft)
        legend_layout.addWidget(self.label_draw, alignment=Qt.AlignCenter)
        legend_layout.addWidget(self.label_b, alignment=Qt.AlignRight)
        main_layout.addLayout(legend_layout)
        self.set_theme(theme) 
    def set_theme(self, theme):
        self.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border: 1px solid {theme['ACCENT_BLUE']}; border-radius: 5px; padding: 5px;")
        self.title_label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 12pt; font-weight: bold; border: none; padding-bottom: 2px;")
        self.match_label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 10pt; border: none; padding-bottom: 5px;")
        self.label_a.setStyleSheet(f"color: {theme['ACCENT_BLUE']}; font-weight: bold; border: none;")
        self.label_draw.setStyleSheet(f"color: #888888; font-weight: bold; border: none;")
        self.label_b.setStyleSheet(f"color: {theme['HIGHLIGHT_GREEN']}; font-weight: bold; border: none;")
        self.bar.set_theme(theme)


# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# --- Data Visuals Tab Widget ---
class DataVisualsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.stats_df = None
        self.theme = DARK_THEME 
        self.feature_cols = [
            'xg_plus_minus_per90', 'xg', 'possession', 'gk_save_pct', 
            'sca_per90', 'shots_on_target_pct', 'passes_pct', 
            'progressive_passes', 'passes_into_penalty_area', 
            'interceptions', 'aerials_won_pct', 'fouls'
        ]
        self.team_map = {} 
        self.column_map = {}

        layout = QVBoxLayout(self)
        self.visuals_tabs = QTabWidget()
        layout.addWidget(self.visuals_tabs)

        self.init_cluster_tab()
        self.init_heatmap_tab()
        self.init_histogram_tab()
        self.init_scatter_tab()
        self.init_comparison_tab()

    def load_data(self, stats_df, column_map):
        self.stats_df = stats_df
        self.column_map = column_map
        
        display_features = [column_map[col] for col in self.feature_cols if col in column_map]
        
        self.hist_combo.addItems(display_features)
        self.scatter_combo_x.addItems(display_features)
        self.scatter_combo_y.addItems(display_features)
        
        try:
            xg_index = display_features.index(column_map['xg'])
            self.scatter_combo_y.setCurrentIndex(xg_index)
        except (ValueError, KeyError):
            self.scatter_combo_y.setCurrentIndex(1) 
        
        self.comp_stat_combo.addItems(display_features)
        
        self.team_list_widget.clear()
        self.team_map = {}
        for i, row in self.stats_df.iterrows():
            team_name = row['team'] 
            self.team_map[team_name] = i
            item = QListWidgetItem(team_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.team_list_widget.addItem(item)
            
        self.plot_heatmap()
        self.plot_histogram()
        self.plot_scatter()

    def set_theme(self, theme):
        self.theme = theme
        
        self.visuals_tabs.setStyleSheet(f"""
            QTabBar::tab {{
                background: {theme['WINDOW_BG']};
                color: {theme['PRIMARY_TEXT']};
                padding: 8px 15px;
                font-size: 10pt;
            }}
            QTabBar::tab:selected {{
                background: {theme['CONTENT_BG']};
                border-top: 2px solid {theme['HIGHLIGHT_GREEN']};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['CONTENT_BG']};
            }}
        """)
        
        for combo in [self.hist_combo, self.scatter_combo_x, self.scatter_combo_y, self.comp_stat_combo]:
            combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {theme['CONTENT_BG']};
                    color: {theme['PRIMARY_TEXT']};
                    padding: 5px;
                    border: 1px solid {theme['ACCENT_BLUE']};
                }}
            """)
        
        self.team_list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {theme['CONTENT_BG']};
                color: {theme['PRIMARY_TEXT']};
                border: 1px solid {theme['ACCENT_BLUE']};
            }}
        """)
        
        self.comp_plot_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {theme['ACCENT_BLUE']}; 
                color: {theme['PRIMARY_TEXT']}; 
                font-size: 10pt; padding: 8px; border-radius: 5px; 
            }}
            QPushButton:hover {{ background-color: {theme['HIGHLIGHT_GREEN']}; color: {theme['BUTTON_TEXT']}; }}
        """)

        self.plot_heatmap()
        self.plot_histogram()
        self.plot_scatter()
        self.plot_team_comparison()
        
        self.cluster_image_label.setStyleSheet(f"background-color: {theme['CONTENT_BG']};")


    def init_cluster_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.cluster_image_label = QLabel("Loading K-Means Cluster Image...")
        self.cluster_image_label.setAlignment(Qt.AlignCenter)
        
        pixmap = QPixmap(CLUSTER_IMAGE_PATH)
        if pixmap.isNull():
            self.cluster_image_label.setText(f"Error: Could not load image from:\n{CLUSTER_IMAGE_PATH}")
        else:
            self.cluster_image_label.setPixmap(pixmap.scaled(
                800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation 
            ))
            
        layout.addWidget(self.cluster_image_label)
        self.visuals_tabs.addTab(tab, "Cluster Plot")

    def init_heatmap_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.heatmap_canvas = MplCanvas(self)
        layout.addWidget(self.heatmap_canvas)
        self.visuals_tabs.addTab(tab, "Correlation Heatmap")

    def init_histogram_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Select Feature:"))
        self.hist_combo = QComboBox()
        self.hist_combo.currentIndexChanged.connect(self.plot_histogram)
        controls.addWidget(self.hist_combo)
        controls.addStretch()
        layout.addLayout(controls)
        
        self.hist_canvas = MplCanvas(self)
        layout.addWidget(self.hist_canvas)
        self.visuals_tabs.addTab(tab, "Histogram")

    def init_scatter_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel("X-Axis:"))
        self.scatter_combo_x = QComboBox()
        self.scatter_combo_x.currentIndexChanged.connect(self.plot_scatter)
        controls.addWidget(self.scatter_combo_x)
        
        controls.addWidget(QLabel("Y-Axis:"))
        self.scatter_combo_y = QComboBox()
        self.scatter_combo_y.currentIndexChanged.connect(self.plot_scatter)
        controls.addWidget(self.scatter_combo_y)
        controls.addStretch()
        layout.addLayout(controls)
        
        self.scatter_canvas = MplCanvas(self)
        layout.addWidget(self.scatter_canvas)
        self.visuals_tabs.addTab(tab, "Scatter Plot")

    def init_comparison_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("1. Select Statistic:"))
        self.comp_stat_combo = QComboBox()
        controls_layout.addWidget(self.comp_stat_combo)
        
        controls_layout.addWidget(QLabel("2. Select Teams:"))
        self.team_list_widget = QListWidget()
        self.team_list_widget.setSelectionMode(QListWidget.MultiSelection)
        controls_layout.addWidget(self.team_list_widget)
        
        self.comp_plot_button = QPushButton("Plot Comparison")
        self.comp_plot_button.clicked.connect(self.plot_team_comparison)
        controls_layout.addWidget(self.comp_plot_button)
        
        layout.addLayout(controls_layout, 1) 
        
        self.comp_canvas = MplCanvas(self)
        layout.addWidget(self.comp_canvas, 2) 
        
        self.visuals_tabs.addTab(tab, "Team Comparison")
        
    def _apply_plot_theme(self, fig, ax):
        fig.patch.set_facecolor(self.theme['CONTENT_BG'])
        ax.set_facecolor(self.theme['CONTENT_BG'])
        
        ax.tick_params(colors=self.theme['PRIMARY_TEXT'], which='both')
        ax.xaxis.label.set_color(self.theme['PRIMARY_TEXT'])
        ax.yaxis.label.set_color(self.theme['PRIMARY_TEXT'])
        ax.title.set_color(self.theme['PRIMARY_TEXT'])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(self.theme['PRIMARY_TEXT'])

    # --- FIX: Changed self.visuals_tab.column_map to self.column_map ---
    def _get_raw_col_name(self, display_name):
        """Helper to get original column name from display name."""
        for raw, display in self.column_map.items():
            if display == display_name:
                return raw
        return None

    def plot_heatmap(self):
        if self.stats_df is None: return
        try:
            self.heatmap_canvas.axes.clear()
            # Use renamed columns for heatmap labels
            renamed_df = self.stats_df[self.feature_cols].rename(columns=self.column_map)
            corr = renamed_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=self.heatmap_canvas.axes)
            self.heatmap_canvas.axes.tick_params(axis='x', rotation=45)
            self._apply_plot_theme(self.heatmap_canvas.fig, self.heatmap_canvas.axes)
            self.heatmap_canvas.fig.tight_layout()
            self.heatmap_canvas.draw()
        except Exception as e:
            print(f"Error plotting heatmap: {e}")

    def plot_histogram(self):
        if self.stats_df is None: return
        try:
            display_name = self.hist_combo.currentText()
            raw_name = self._get_raw_col_name(display_name)
            if not raw_name: return
            
            self.hist_canvas.axes.clear()
            sns.histplot(self.stats_df[raw_name], kde=True, ax=self.hist_canvas.axes, color=self.theme['ACCENT_BLUE'])
            self.hist_canvas.axes.set_title(f"Distribution of {display_name}")
            self.hist_canvas.axes.set_xlabel(display_name)
            self._apply_plot_theme(self.hist_canvas.fig, self.hist_canvas.axes)
            self.hist_canvas.fig.tight_layout()
            self.hist_canvas.draw()
        except Exception as e:
            print(f"Error plotting histogram: {e}")

    def plot_scatter(self):
        if self.stats_df is None: return
        try:
            x_display = self.scatter_combo_x.currentText()
            y_display = self.scatter_combo_y.currentText()
            x_raw = self._get_raw_col_name(x_display)
            y_raw = self._get_raw_col_name(y_display)
            if not x_raw or not y_raw: return

            self.scatter_canvas.axes.clear()
            sns.scatterplot(data=self.stats_df, x=x_raw, y=y_raw, ax=self.scatter_canvas.axes, color=self.theme['HIGHLIGHT_GREEN'])
            self.scatter_canvas.axes.set_title(f"{x_display} vs. {y_display}")
            self.scatter_canvas.axes.set_xlabel(x_display)
            self.scatter_canvas.axes.set_ylabel(y_display)
            self._apply_plot_theme(self.scatter_canvas.fig, self.scatter_canvas.axes)
            self.scatter_canvas.fig.tight_layout()
            self.scatter_canvas.draw()
        except Exception as e:
            print(f"Error plotting scatter: {e}")

    def plot_team_comparison(self):
        if self.stats_df is None: return
        try:
            display_name = self.comp_stat_combo.currentText()
            stat = self._get_raw_col_name(display_name)
            if not stat: return
            
            selected_teams = []
            for i in range(self.team_list_widget.count()):
                item = self.team_list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    selected_teams.append(item.text())
            
            if not selected_teams:
                self.comp_canvas.axes.clear()
                self.comp_canvas.draw()
                return

            plot_data = self.stats_df[self.stats_df['team'].isin(selected_teams)].sort_values(by=stat, ascending=False)

            self.comp_canvas.axes.clear()
            sns.barplot(data=plot_data, x=stat, y='team', ax=self.comp_canvas.axes, color=self.theme['ACCENT_BLUE'])
            self.comp_canvas.axes.set_title(f"Comparison of {display_name}")
            self.comp_canvas.axes.set_xlabel(display_name)
            self.comp_canvas.axes.set_ylabel("Team")
            self._apply_plot_theme(self.comp_canvas.fig, self.comp_canvas.axes)
            self.comp_canvas.fig.tight_layout()
            self.comp_canvas.draw()
        except Exception as e:
            print(f"Error plotting comparison: {e}")


# --- Main Application ---
class FootballPredictorApp(QWidget):
    # ... (init_ui, load_stats_data, all other methods are unchanged) ...
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIFA Knockout Stage Simulator")
        self.setGeometry(100, 100, 1400, 900) 

        self.current_theme = "dark" 
        self.DARK_THEME = DARK_THEME
        self.LIGHT_THEME = LIGHT_THEME
        
        self.TEAM_BUTTON_STYLE = ""
        self.TEAM_BUTTON_SELECTED_STYLE = ""

        self.loaded_models = False
        self.rf_classifier = None
        self.preprocessor = None
        self.label_encoder = None
        self.team_win_rate_lookup = {}
        self.top_teams_pool = [] 
        
        self.selected_tournament_teams = []
        self.team_button_map = {} 
        
        self.simulation_results = {}
        
        self.kmeans_worker = None
        self.loading_dialog = LoadingDialog(self)
        self.stats_df = None 

        self.init_ui()
        self.apply_stylesheet(self.DARK_THEME) 
        
        self.load_models_and_data() 
        self.load_stats_data() 

    def init_ui(self):
        main_app_layout = QVBoxLayout(self)
        main_app_layout.setContentsMargins(10, 10, 10, 10)
        
        top_bar_layout = QHBoxLayout()
        self.title_label = QLabel("FIFA Knockout Stage Simulator")
        top_bar_layout.addWidget(self.title_label)
        top_bar_layout.addStretch()
        self.theme_toggle_button = QPushButton("Light Mode")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        top_bar_layout.addWidget(self.theme_toggle_button)
        main_app_layout.addLayout(top_bar_layout)
        
        self.tabs = QTabWidget()
        main_app_layout.addWidget(self.tabs)

        self.simulator_tab = QWidget()
        self.stats_tab = QWidget()
        self.odds_tab = QWidget()
        self.visuals_tab = DataVisualsTab() 

        self.tabs.addTab(self.simulator_tab, "Tournament Simulator")
        self.tabs.addTab(self.stats_tab, "Team Data") 
        self.tabs.addTab(self.odds_tab, "Match Odds")
        self.tabs.addTab(self.visuals_tab, "Data Visuals") 
        
        self.init_simulator_tab()
        self.init_stats_tab()
        self.init_odds_tab()


    def init_simulator_tab(self):
        simulator_tab_layout = QHBoxLayout(self.simulator_tab)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        self.controls_frame = QFrame(self)
        self.controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QVBoxLayout(self.controls_frame)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        self.title_controls = QLabel("Tournament Controls", self.controls_frame)
        self.title_controls.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.title_controls)
        controls_layout.addSpacing(10)
        self.selected_teams_label = QLabel("Selected Teams (Pick 8):", self.controls_frame)
        controls_layout.addWidget(self.selected_teams_label)
        self.selected_teams_display = QTextEdit(self.controls_frame)
        self.selected_teams_display.setReadOnly(True)
        self.selected_teams_display.setFixedHeight(150) 
        controls_layout.addWidget(self.selected_teams_display)
        self.auto_select_button = QPushButton("Randomize Bracket", self.controls_frame)
        self.auto_select_button.clicked.connect(self.auto_select_teams)
        controls_layout.addWidget(self.auto_select_button)
        self.start_tournament_button = QPushButton("Start Tournament", self.controls_frame)
        self.start_tournament_button.clicked.connect(self.start_tournament_simulation)
        controls_layout.addWidget(self.start_tournament_button)
        self.reset_button = QPushButton("Reset Selections", self.controls_frame)
        self.reset_button.clicked.connect(self.reset_all)
        controls_layout.addWidget(self.reset_button)
        self.controls_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) 
        left_panel.addWidget(self.controls_frame)
        self.team_grid_frame = QFrame(self)
        self.team_grid_frame.setFrameShape(QFrame.StyledPanel)
        team_grid_main_layout = QVBoxLayout(self.team_grid_frame) 
        team_grid_main_layout.setContentsMargins(15, 15, 15, 15)
        self.team_grid_label = QLabel("Team Pool (from K-Means):", self.team_grid_frame)
        self.team_grid_label.setAlignment(Qt.AlignCenter) 
        team_grid_main_layout.addWidget(self.team_grid_label)
        self.team_grid_layout = QGridLayout() 
        self.team_grid_layout.setSpacing(10)
        for i in range(12): 
            btn = QPushButton("...", self.team_grid_frame)
            btn.setFixedSize(70, 70) 
            self.team_grid_layout.addWidget(btn, i // 4, i % 4)
            btn.hide() 
        team_grid_main_layout.addLayout(self.team_grid_layout) 
        team_grid_main_layout.addStretch() 
        self.team_grid_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_panel.addWidget(self.team_grid_frame)
        simulator_tab_layout.addLayout(left_panel, 2) 
        right_main_panel = QVBoxLayout()
        right_main_panel.setSpacing(10)
        self.info_dump_frame = QFrame(self)
        self.info_dump_frame.setFrameShape(QFrame.StyledPanel)
        info_dump_layout = QVBoxLayout(self.info_dump_frame)
        self.title_info = QLabel("Simulation Log", self.info_dump_frame)
        self.title_info.setAlignment(Qt.AlignCenter)
        info_dump_layout.addWidget(self.title_info)
        self.output_info = QTextEdit(self.info_dump_frame)
        self.output_info.setReadOnly(True)
        info_dump_layout.addWidget(self.output_info)
        self.info_dump_frame.setFixedHeight(250) 
        right_main_panel.addWidget(self.info_dump_frame) 
        self.main_show_frame = QFrame(self)
        self.main_show_frame.setFrameShape(QFrame.StyledPanel)
        main_show_layout = QVBoxLayout(self.main_show_frame)
        main_show_layout.setContentsMargins(10, 10, 10, 10)
        self.title_main_show = QLabel("Tournament Bracket", self.main_show_frame)
        self.title_main_show.setAlignment(Qt.AlignCenter)
        main_show_layout.addWidget(self.title_main_show)
        self.bracket_view = BracketView()
        main_show_layout.addWidget(self.bracket_view)
        self.main_show_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_main_panel.addWidget(self.main_show_frame) 
        simulator_tab_layout.addLayout(right_main_panel, 5)

    def init_stats_tab(self):
        stats_layout = QVBoxLayout(self.stats_tab)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        self.stats_title = QLabel("Team Performance Statistics")
        self.stats_title.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stats_title)
        self.stats_subtitle = QLabel("Data from final_team_data.csv. Click headers to sort.")
        self.stats_subtitle.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stats_subtitle)
        self.stats_table_view = QTableView()
        self.stats_table_view.setSortingEnabled(True)
        stats_layout.addWidget(self.stats_table_view)
        
    def init_odds_tab(self):
        odds_layout = QVBoxLayout(self.odds_tab)
        odds_layout.setContentsMargins(10, 10, 10, 10)
        self.odds_title = QLabel("Tournament Match Odds")
        self.odds_title.setAlignment(Qt.AlignCenter)
        odds_layout.addWidget(self.odds_title)
        self.odds_subtitle = QLabel("Prediction probabilities for each simulated match.")
        self.odds_subtitle.setAlignment(Qt.AlignCenter)
        odds_layout.addWidget(self.odds_subtitle)
        self.odds_scroll_area = QScrollArea()
        self.odds_scroll_area.setWidgetResizable(True)
        self.odds_scroll_widget = QWidget() 
        self.odds_layout = QVBoxLayout(self.odds_scroll_widget) 
        self.odds_layout.setAlignment(Qt.AlignTop)
        self.odds_scroll_area.setWidget(self.odds_scroll_widget)
        odds_layout.addWidget(self.odds_scroll_area)
        
    def toggle_theme(self):
        if self.current_theme == "dark":
            self.current_theme = "light"
            self.apply_stylesheet(self.LIGHT_THEME)
            self.theme_toggle_button.setText("Dark Mode")
        else:
            self.current_theme = "dark"
            self.apply_stylesheet(self.DARK_THEME)
            self.theme_toggle_button.setText("Light Mode")

    def apply_stylesheet(self, theme):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(theme["WINDOW_BG"]))
        self.setPalette(palette)
        
        self.title_label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 16pt; font-weight: bold;")
        self.theme_toggle_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {theme['ACCENT_BLUE']}; 
                color: {theme['PRIMARY_TEXT']}; 
                font-size: 9pt; 
                padding: 5px 10px; 
                border-radius: 5px; 
            }}
            QPushButton:hover {{ background-color: {theme['HIGHLIGHT_GREEN']}; color: {theme['BUTTON_TEXT']}; }}
        """)
        
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border-top: 2px solid {theme['ACCENT_BLUE']}; 
            }}
            QTabBar::tab {{
                background: {theme['WINDOW_BG']};
                color: {theme['PRIMARY_TEXT']};
                padding: 8px 8px; 
                font-size: 9pt; 
                min-width: 120px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {theme['ACCENT_BLUE']};
                border-bottom: none;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {theme['ACCENT_BLUE']};
                color: {theme['PRIMARY_TEXT']};
            }}
            QTabBar::tab:!selected:hover {{
                background: {theme['CONTENT_BG']};
            }}
        """)
        
        self.controls_frame.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border-radius: 8px;")
        self.title_controls.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 16pt; font-weight: bold;")
        self.selected_teams_label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 12pt;")
        self.selected_teams_display.setStyleSheet(f"background-color: {theme['WINDOW_BG']}; color: {theme['PRIMARY_TEXT']}; font-size: 10pt; border-radius: 5px;")
        button_style = f"""
            QPushButton {{ background-color: {theme['ACCENT_BLUE']}; color: {theme['PRIMARY_TEXT']}; 
                          font-size: 12pt; padding: 8px; border-radius: 5px; }}
            QPushButton:hover {{ background-color: {theme['HIGHLIGHT_GREEN']}; color: {theme['BUTTON_TEXT']}; }}
        """
        self.auto_select_button.setStyleSheet(button_style)
        self.reset_button.setStyleSheet(button_style)
        self.start_tournament_button.setStyleSheet(f"""
            QPushButton {{ background-color: {theme['HIGHLIGHT_GREEN']}; color: {theme['BUTTON_TEXT']}; 
                          font-size: 14pt; padding: 10px; border-radius: 5px; margin-top: 5px; }}
            QPushButton:hover {{ background-color: {theme['ACCENT_BLUE']}; color: {theme['PRIMARY_TEXT']}; }}
        """)
        self.team_grid_frame.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border-radius: 8px;")
        self.team_grid_label.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 12pt; margin-bottom: 10px;")
        self.TEAM_BUTTON_STYLE = f"""
            QPushButton {{
                background-color: {theme['HIGHLIGHT_GREEN']};
                color: {theme['BUTTON_TEXT']};
                border-radius: 35px; font-size: 10pt; font-weight: bold;
                border: 2px solid {theme['HIGHLIGHT_GREEN']};
            }}
            QPushButton:hover {{ background-color: {theme['ACCENT_BLUE']}; color: {theme['PRIMARY_TEXT']}; }}
        """
        self.TEAM_BUTTON_SELECTED_STYLE = f"""
            QPushButton {{
                background-color: {theme['HIGHLIGHT_GREEN']};
                color: {theme['BUTTON_TEXT']};
                border-radius: 35px; font-size: 10pt; font-weight: bold;
                border: 3px solid #FFD700;
            }}
        """
        for team, button in self.team_button_map.items():
            if team in self.selected_tournament_teams:
                button.setStyleSheet(self.TEAM_BUTTON_SELECTED_STYLE)
            else:
                button.setStyleSheet(self.TEAM_BUTTON_STYLE)
        self.info_dump_frame.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border-radius: 8px;")
        self.title_info.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 16pt; font-weight: bold;")
        self.output_info.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; color: {theme['PRIMARY_TEXT']}; font-size: 10pt; border: none;")
        self.main_show_frame.setStyleSheet(f"background-color: {theme['WINDOW_BG']}; border-radius: 8px;")
        self.title_main_show.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 16pt; font-weight: bold;")
        self.bracket_view.set_theme(theme) 
        
        self.stats_title.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 18pt; font-weight: bold;")
        self.stats_subtitle.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 11pt; margin-bottom: 10px;")
        self.stats_table_view.setStyleSheet(f"""
            QTableView {{ 
                background-color: {theme['CONTENT_BG']}; 
                color: {theme['PRIMARY_TEXT']}; 
                border: 1px solid {theme['ACCENT_BLUE']};
                border-radius: 8px;
                gridline-color: {theme['ACCENT_BLUE']};
            }}
            QHeaderView::section {{ 
                background-color: {theme['ACCENT_BLUE']}; 
                color: {theme['PRIMARY_TEXT']};
                padding: 6px;
                font-size: 10pt;
                font-weight: bold;
                border: 1px solid {theme['WINDOW_BG']};
            }}
            QTableView::item {{
                padding: 5px;
            }}
        """)
        
        self.odds_title.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 18pt; font-weight: bold;")
        self.odds_subtitle.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 11pt; margin-bottom: 10px;")
        self.odds_scroll_widget.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border-radius: 8px;")
        self.odds_scroll_area.setStyleSheet(f"background-color: {theme['CONTENT_BG']}; border: 1px solid {theme['ACCENT_BLUE']}; border-radius: 8px;")
        for i in range(self.odds_layout.count()):
            widget = self.odds_layout.itemAt(i).widget()
            if isinstance(widget, MatchOddsWidget):
                widget.set_theme(theme)
            elif isinstance(widget, QLabel):
                widget.setStyleSheet(f"color: {theme['PRIMARY_TEXT']}; font-size: 14pt; font-weight: bold; margin-top: 10px; border: none;")
        
        self.visuals_tab.set_theme(theme)
        
        self.loading_dialog.set_theme(theme)


    def load_stats_data(self):
        try:
            self.stats_df = pd.read_csv(KMEANS_DATA_PATH)
            
            column_map = {
                'team': 'Team', 'xg_plus_minus_per90': 'xG +/-', 'xg': 'xG',
                'possession': 'Possession %', 'gk_save_pct': 'Save %',
                'sca_per90': 'Shot Actions', 'shots_on_target_pct': 'On-Target %',
                'passes_pct': 'Pass %', 'progressive_passes': 'Prog. Passes',
                'passes_into_penalty_area': 'Box Passes', 'interceptions': 'Interceptions',
                'aerials_won_pct': 'Aerials Won %', 'fouls': 'Fouls'
            }
            useful_cols = [col for col in column_map.keys() if col in self.stats_df.columns]
            
            display_df = self.stats_df[useful_cols].copy()
            
            model = QStandardItemModel(len(display_df), len(useful_cols))
            model.setHorizontalHeaderLabels([column_map[col] for col in useful_cols])
            
            for i, row in display_df.iterrows():
                for j, col_name in enumerate(useful_cols):
                    item_value = row[col_name]
                    if isinstance(item_value, (float, np.floating)):
                        item_value = f"{item_value:.2f}"
                    
                    item = QStandardItem(str(item_value))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable) 
                    model.setItem(i, j, item)
            
            self.stats_table_view.setModel(model)
            self.stats_table_view.resizeColumnsToContents()
            
            self.visuals_tab.load_data(self.stats_df, column_map)
            
        except Exception as e:
            self.output_info.append(f"Error loading stats data: {e}")
            if self.stats_table_view:
                self.stats_table_view.setParent(None) 

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
                                "1. Click teams in the order you want them to play (1 vs 2, 3 vs 4...).\n\n"
                                "2. Or, click 'Randomize Bracket'.")

    def on_kmeans_error(self, error_message):
        self.loading_dialog.hide()
        self.output_info.append(f"--- K-MEANS SCRIPT FAILED ---")
        self.output_info.append(error_message)
        self.output_info.append("Please check script and file paths.")
        self.auto_select_button.setEnabled(False)
        self.start_tournament_button.setEnabled(False)

    def update_team_buttons(self):
        for i in reversed(range(self.team_grid_layout.count())): 
            widget = self.team_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.team_button_map.clear()
        
        for i, team_name in enumerate(self.top_teams_pool):
            btn = QPushButton(team_name[:3].upper(), self)
            btn.setFixedSize(70, 70) 
            btn.setToolTip(team_name) 
            btn.setStyleSheet(self.TEAM_BUTTON_STYLE) 
            btn.clicked.connect(lambda checked, name=team_name: self.team_button_clicked(name))
            
            row, col = divmod(i, 4) 
            self.team_grid_layout.addWidget(btn, row, col)
            self.team_button_map[team_name] = btn
            btn.show()
            
        self.team_grid_layout.setRowStretch(self.team_grid_layout.rowCount(), 1)

    def team_button_clicked(self, team_name):
        if team_name in self.selected_tournament_teams:
            self.selected_tournament_teams.remove(team_name)
            self.team_button_map[team_name].setStyleSheet(self.TEAM_BUTTON_STYLE) 
            self.output_info.append(f"Removed {team_name} from tournament.")
        else:
            if len(self.selected_tournament_teams) < len(self.top_teams_pool):
                self.selected_tournament_teams.append(team_name)
                self.team_button_map[team_name].setStyleSheet(self.TEAM_BUTTON_SELECTED_STYLE) 
                self.output_info.append(f"Added {team_name} to tournament.")
            else:
                self.output_info.append(f"Cannot add team: You have already selected all {len(self.top_teams_pool)} teams.")
        self.selected_teams_display.setText(
            f"Selected: {len(self.selected_tournament_teams)} / {len(self.top_teams_pool)}\n\n" + 
            "\n".join(self.selected_tournament_teams)
        )

    def auto_select_teams(self):
        self.reset_all() 
        if len(self.top_teams_pool) < 8:
            self.output_info.append("Randomize failed: Not at least 8 teams in the pool.")
            return
        self.output_info.append("Randomizing bracket...")
        
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
            self.clear_odds_widgets() 
            return

        self.output_info.append("Starting 8-team knockout simulation...")
        self.bracket_view.clear_bracket()
        
        self.simulation_results = {
            "quarter_finals": [],
            "semi_finals": [],
            "final": []
        }
        
        teams = self.selected_tournament_teams.copy()
        
        # --- Quarter-Finals ---
        self.output_info.append("--- QUARTER-FINALS ---")
        qf_winners = []
        qf_matches = [(teams[0], teams[1]), (teams[2], teams[3]), (teams[4], teams[5]), (teams[6], teams[7])]
        
        y_step = 90 
        qf_y_positions = [20, 20 + y_step, 20 + 2*y_step, 20 + 3*y_step]
        
        for i, (team_a, team_b) in enumerate(qf_matches):
            result = self.predict_match(team_a, team_b, is_knockout=True)
            self.simulation_results["quarter_finals"].append(result) 
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
            self.simulation_results["semi_finals"].append(result) 
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
        self.simulation_results["final"].append(result) 
        final_winner = self.get_winner(result)
        self.output_info.append(f"{final_match[0]} vs {final_match[1]} -> Winner: {final_winner}")

        self.bracket_view.draw_final(20 + (120 + 60) * 2, final_y_pos, final_match[0], final_match[1], final_winner)
        self.output_info.append(f"🏆 TOURNAMENT CHAMPION: {final_winner} 🏆")
        
        self.bracket_view.fit_view()
        
        self.display_odds_widgets()
        
    def display_odds_widgets(self):
        self.clear_odds_widgets() 
        
        if not self.simulation_results:
            return
        
        theme = self.DARK_THEME if self.current_theme == "dark" else self.LIGHT_THEME
        
        qf_label = QLabel("--- QUARTER-FINALS ---")
        self.odds_layout.addWidget(qf_label)
        for i, res in enumerate(self.simulation_results["quarter_finals"]):
            widget = MatchOddsWidget(f"Quarter-Final {i+1}", res, theme)
            self.odds_layout.addWidget(widget)

        sf_label = QLabel("--- SEMI-FINALS ---")
        self.odds_layout.addWidget(sf_label)
        for i, res in enumerate(self.simulation_results["semi_finals"]):
            widget = MatchOddsWidget(f"Semi-Final {i+1}", res, theme)
            self.odds_layout.addWidget(widget)

        final_label = QLabel("--- FINAL ---")
        self.odds_layout.addWidget(final_label)
        for res in self.simulation_results["final"]:
            widget = MatchOddsWidget("Final", res, theme)
            self.odds_layout.addWidget(widget)
        
        # Apply theme to the new labels
        self.apply_stylesheet(theme)

    def clear_odds_widgets(self):
        while self.odds_layout.count():
            child = self.odds_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def reset_all(self):
        self.output_info.clear()
        self.bracket_view.clear_bracket() 
        
        self.clear_odds_widgets() 
        self.simulation_results = {}
        
        self.selected_tournament_teams = []
        self.selected_teams_display.clear()
        
        for team_name, button in self.team_button_map.items():
            button.setStyleSheet(self.TEAM_BUTTON_STYLE)
            
        self.output_info.append("Selections reset. Ready to start a new tournament.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FootballPredictorApp()
    ex.show()
    sys.exit(app.exec_())