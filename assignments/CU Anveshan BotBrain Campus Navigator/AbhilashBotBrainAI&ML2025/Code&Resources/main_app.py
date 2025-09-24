import sys
import math
import os
from collections import deque
from queue import PriorityQueue
import heapq
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QGridLayout, QButtonGroup, QMessageBox, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsLineItem, QGraphicsTextItem,
    QLineEdit, QStyle, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QBrush, QPen, QColor, QFont, QPainter, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QObject, QThread

import botbrain_ai as ai

NODE_RADIUS = 40
COLOR_DEFAULT = QColor("#8B7D6B")
COLOR_START = QColor("#556B2F")
COLOR_GOAL = QColor("#8B0000")
COLOR_HOVER = QColor("#A0522D")
COLOR_EDGE = QColor("#999999")
COLOR_TEXT = QColor("#f0f0f0")
COLOR_PATH = QColor("#00BFFF")


class AIWorker(QObject):
    result_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            source, destination = ai.parse_user_query(self.query)
            self.result_ready.emit(source, destination)
        except Exception as e:
            self.error.emit(str(e))


class NodeItem(QGraphicsObject):
    node_clicked = pyqtSignal(str)
    node_hovered = pyqtSignal(str, bool)

    def __init__(self, name, x, y):
        super().__init__()
        self.setPos(x, y)
        self.name = name
        self.current_color = COLOR_DEFAULT
        self.setAcceptHoverEvents(True)

        self.label = QGraphicsTextItem(self.name, self)
        self.label.setDefaultTextColor(COLOR_TEXT)
        label_rect = self.label.boundingRect()
        self.label.setPos(-label_rect.width() / 2, -label_rect.height() / 2)
        self.label.setZValue(2)

    def set_color(self, color):
        self.current_color = color
        self.update()

    def boundingRect(self):
        return QRectF(-NODE_RADIUS - 1, -NODE_RADIUS - 1, NODE_RADIUS * 2 + 2, NODE_RADIUS * 2 + 2)

    def paint(self, painter, option, widget):
        painter.setPen(QPen(Qt.NoPen))
        if option.state & QStyle.State_MouseOver:
            painter.setBrush(COLOR_HOVER)
            painter.setPen(QPen(Qt.white, 2))
        else:
            painter.setBrush(self.current_color)
        painter.drawEllipse(-NODE_RADIUS, -NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)

    def mousePressEvent(self, event):
        self.node_clicked.emit(self.name)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self.node_hovered.emit(self.name, True)
        super().hoverEnterEvent(event)
        self.update() 

    def hoverLeaveEvent(self, event):
        self.node_hovered.emit(self.name, False)
        super().hoverLeaveEvent(event)
        self.update()

def get_campus_graph():
    return {
        "Main Gate": [("Junction1", 140), ("OuterJunction1", 275), ("OuterJunction4", 275)],
        "Junction1": [("Main Gate", 140), ("Admin Block", 90)],
        "Admin Block": [("Junction1", 90), ("Library", 50), ("Stamba", 90)],
        "Library": [("Admin Block", 50)],
        "Stamba": [("Admin Block", 90), ("AcadA", 90), ("AcadB", 90), ("AcadC", 90)],
        "AcadA": [("Stamba", 90)],
        "AcadB": [("Stamba", 90), ("Junction2", 140)],
        "AcadC": [("Stamba", 90)],
        "Junction2": [("AcadB", 140), ("Faculty Housing", 90), ("Foodcourt & Laundry", 90), ("OuterJunction2", 270), ("OuterJunction5", 270)],
        "Faculty Housing": [("Junction2", 90), ("Foodcourt & Laundry", 50)],
        "Foodcourt & Laundry": [("Junction2", 90), ("Hostel", 90), ("Sports Complex", 230), ("Faculty Housing", 50)],
        "Hostel": [("Foodcourt & Laundry", 90)],
        "Sports Complex": [("Foodcourt & Laundry", 230), ("Cricket Ground", 130), ("Other Sports Grounds", 90), ("OuterJunction3", 275), ("OuterJunction6", 275)],
        "Cricket Ground": [("Sports Complex", 130)],
        "Other Sports Grounds": [("Sports Complex", 90)],
        "OuterJunction1": [("Main Gate", 275), ("OuterJunction2", 275)],
        "OuterJunction2": [("OuterJunction1", 275), ("OuterJunction3", 275), ("Junction2", 270)],
        "OuterJunction3": [("OuterJunction2", 275), ("Sports Complex", 275)],
        "OuterJunction4": [("Main Gate", 275), ("OuterJunction5", 275)],
        "OuterJunction5": [("OuterJunction4", 275), ("OuterJunction6", 275), ("Junction2", 270)],
        "OuterJunction6": [("OuterJunction5", 275), ("Sports Complex", 275)],
    }
def get_node_positions():
    return {
            "Main Gate": (0, 0), "Junction1": (100, 0), "Admin Block": (200, 0), "Library": (200, -75),
            "Stamba": (300, 0), "AcadA": (300, -90), "AcadB": (400, 0), "AcadC": (300, 90),
            "Junction2": (500, 0), "Faculty Housing": (600, -90), "Foodcourt & Laundry": (600, 0),
            "Hostel": (700, 75), "Sports Complex": (800, 0), "Cricket Ground": (900, -75),
            "Other Sports Grounds": (900, 75),
            "OuterJunction1": (150, -200),
            "OuterJunction2": (400, -250),
            "OuterJunction3": (650, -200),
            "OuterJunction4": (150, 200),
            "OuterJunction5": (400, 250),
            "OuterJunction6": (650, 200),
            }

def ucs(graph, start, target):
    visited, pq = set(), PriorityQueue(); pq.put((0, start, [start]))
    while not pq.empty():
        cost, node, path = pq.get()
        if node == target: return path, cost
        if node not in visited: visited.add(node)
        for neighbour, edge_cost in graph.get(node, []):
            if neighbour not in visited: pq.put((cost + edge_cost, neighbour, path + [neighbour]))
    return [], float('inf')
def dijkstra(graph, start, target):
    heap, visited = [(0, start, [start])], {}
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node == target: return path, cost
        if node in visited and visited[node] <= cost: continue
        visited[node] = cost
        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited or cost + edge_cost < visited.get(neighbor, float('inf')):
                heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))
    return [], float('inf')
def bfs(graph, start, target):
    visited, queue = set(), deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == target: return path, len(path) - 1
        if node not in visited: visited.add(node)
        for neighbour, _ in graph.get(node, []):
            if neighbour not in visited: queue.append((neighbour, path + [neighbour]))
    return [], float('inf')
def create_heuristic(positions, target):
    target_pos = positions.get(target)
    if not target_pos: return lambda node: 0
    def heuristic_func(node):
        node_pos = positions.get(node)
        if not node_pos: return 0
        return math.sqrt((node_pos[0] - target_pos[0])**2 + (node_pos[1] - target_pos[1])**2)
    return heuristic_func
def astar(graph, start, target, heuristic_func):
    heap, visited = [(heuristic_func(start), 0, start, [start])], {}
    while heap:
        _, cost, node, path = heapq.heappop(heap)
        if node == target: return path, cost
        if node in visited and visited[node] <= cost: continue
        visited[node] = cost
        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                est = new_cost + heuristic_func(neighbor)
                heapq.heappush(heap, (est, new_cost, neighbor, path + [neighbor]))
    return [], float('inf')

class GraphicsGraphWidget(QGraphicsView):
    node_selected_on_map = pyqtSignal(str, bool)
    node_hover_changed = pyqtSignal(str, bool)

    def __init__(self, graph_data, positions, parent=None):
        super().__init__(parent); self.scene = QGraphicsScene(); self.setScene(self.scene)
        self.graph_data, self.positions = graph_data, positions
        self.node_items, self.start_node, self.goal_node = {}, None, None
        self.selecting_start, self.highlighted_edges = True, []
        self.path_pen = QPen(COLOR_PATH, 5)
        self.draw_graph(); self.setRenderHint(QPainter.Antialiasing); self.setStyleSheet("border: none;")
    def draw_graph(self):
        edge_pen = QPen(COLOR_EDGE, 2)
        drawn_edges = set()
        for start_node, edges in self.graph_data.items():
            for end_node, _ in edges:
                edge_key = tuple(sorted((start_node, end_node)))
                if start_node in self.positions and end_node in self.positions and edge_key not in drawn_edges:
                    pos1, pos2 = self.positions[start_node], self.positions[end_node]
                    line = QGraphicsLineItem(pos1[0], pos1[1], pos2[0], pos2[1])
                    line.setZValue(-1); line.setPen(edge_pen); self.scene.addItem(line)
                    drawn_edges.add(edge_key)

        for name, pos in self.positions.items():
            node = NodeItem(name, pos[0], pos[1])
            node.node_clicked.connect(self.handle_node_click)
            node.node_hovered.connect(self.handle_node_hover)
            self.scene.addItem(node); self.node_items[name] = node
    
    def handle_node_hover(self, node_name, is_hovering):
        self.node_hover_changed.emit(node_name, is_hovering) 

    def handle_node_click(self, node_name): self.node_selected_on_map.emit(node_name, self.selecting_start)
    def update_selection(self, start_node, goal_node):
        if self.start_node and self.start_node != start_node and self.start_node in self.node_items:
            self.node_items[self.start_node].set_color(COLOR_DEFAULT)
        if self.goal_node and self.goal_node != goal_node and self.goal_node in self.node_items:
            self.node_items[self.goal_node].set_color(COLOR_DEFAULT)
        self.start_node, self.goal_node = start_node, goal_node
        if self.start_node in self.node_items: self.node_items[self.start_node].set_color(COLOR_START)
        if self.goal_node in self.node_items: self.node_items[self.goal_node].set_color(COLOR_GOAL)
        self.selecting_start = self.goal_node is not None
    def highlight_path(self, path):
        for edge in self.highlighted_edges: self.scene.removeItem(edge)
        self.highlighted_edges.clear()
        if not path or len(path) < 2: return
        for i in range(len(path) - 1):
            p1, p2 = self.positions.get(path[i]), self.positions.get(path[i+1])
            if p1 and p2:
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]; dist = math.sqrt(dx**2 + dy**2)
                if dist == 0: continue
                ratio = NODE_RADIUS / dist
                sx, sy = p1[0] + dx * ratio, p1[1] + dy * ratio
                ex, ey = p2[0] - dx * ratio, p2[1] - dy * ratio
                line = QGraphicsLineItem(sx, sy, ex, ey); line.setPen(self.path_pen)
                line.setZValue(1); self.scene.addItem(line); self.highlighted_edges.append(line)
    def resizeEvent(self, event): self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio); super().resizeEvent(event)

class CampusNavigatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.graph = get_campus_graph()
        self.locations = sorted(self.graph.keys())
        self.selection_mode = 'Automatic'
        self.start_node = None
        self.goal_node = None
        self.ai_thread = None
        self.ai_worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CU Anveshan')
        self.setGeometry(100, 100, 1400, 900)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        ai_layout = QHBoxLayout()
        self.ai_query_input = QLineEdit()
        self.original_placeholder = "Ask BotBrain: 'Show me the path from the main gate to the hostel'"
        self.ai_query_input.setPlaceholderText(self.original_placeholder)
        self.ai_button = QPushButton("Ask BotBrain")
        ai_layout.addWidget(self.ai_query_input)
        ai_layout.addWidget(self.ai_button)

        mode_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.auto_button = QPushButton("Automatic")
        self.auto_button.setCheckable(True)
        self.auto_button.setChecked(True)
        manual_button = QPushButton("Manual")
        manual_button.setCheckable(True)
        self.mode_button_group.addButton(self.auto_button)
        self.mode_button_group.addButton(manual_button)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["BFS", "A*", "UCS", "Dijkstra"])
        self.algo_combo.setDisabled(True)
        mode_layout.addWidget(self.auto_button)
        mode_layout.addWidget(manual_button)
        mode_layout.addWidget(self.algo_combo)
        mode_layout.addStretch(1)

        central_panel_layout = QHBoxLayout()
        left_panel_layout = QVBoxLayout()
        
        controls_grid = QGridLayout()
        controls_grid.setSpacing(10)
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(3, 1)

        controls_grid.addWidget(QLabel('Start'), 0, 0)
        self.start_combo = QComboBox()
        self.start_combo.addItems([""] + self.locations)
        controls_grid.addWidget(self.start_combo, 1, 0, 1, 2)

        controls_grid.addWidget(QLabel('Goal'), 2, 0)
        self.goal_combo = QComboBox()
        self.goal_combo.addItems([""] + self.locations)
        controls_grid.addWidget(self.goal_combo, 3, 0, 1, 2)
        
        info_labels_layout = QVBoxLayout()
        info_labels_layout.setSpacing(5)
        self.path_label = self.create_info_label('Path:')
        self.cost_label = self.create_info_label('Distance:')
        self.algo_info_label = self.create_info_label('Algorithm:')
        self.info_label = self.create_info_label('Info:')
        info_labels_layout.addWidget(self.path_label)
        info_labels_layout.addWidget(self.cost_label)
        info_labels_layout.addWidget(self.algo_info_label)
        info_labels_layout.addWidget(self.info_label)
        controls_grid.addLayout(info_labels_layout, 0, 2, 4, 2)

        left_panel_layout.addLayout(controls_grid)
        left_panel_layout.addStretch(1)

        action_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton('Find Path')
        self.start_button.setObjectName('findButton')
        self.reset_button = QPushButton('Reset')
        self.reset_button.setObjectName('resetButton')
        action_buttons_layout.addWidget(self.start_button)
        action_buttons_layout.addWidget(self.reset_button)
        left_panel_layout.addLayout(action_buttons_layout)

        image_preview_layout = QVBoxLayout()
        image_preview_layout.addWidget(QLabel("Image Preview"))
        self.image_preview_label = QLabel("Select a goal to see an image.")
        self.image_preview_label.setFixedSize(320, 240)
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setObjectName("imagePreview")
        image_preview_layout.addWidget(self.image_preview_label)
        image_preview_layout.addStretch(1)
        subtext_label = QLabel("Hover over locations to preview images.")
        subtext_label.setAlignment(Qt.AlignCenter)
        subtext_label.setObjectName("imagePreviewSubtext")
        image_preview_layout.addWidget(subtext_label)

        central_panel_layout.addLayout(left_panel_layout, 2)
        central_panel_layout.addLayout(image_preview_layout, 1)

        self.graph_widget = GraphicsGraphWidget(self.graph, get_node_positions())
        
        main_layout.addLayout(ai_layout)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(central_panel_layout)
        main_layout.addWidget(self.graph_widget, 2)

        self.connect_signals()

    def create_info_label(self, title):
        label = QLabel(f'<b>{title}</b> N/A')
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        return label
    
    def connect_signals(self):
        self.mode_button_group.buttonClicked.connect(self.update_selection_mode)
        self.start_button.clicked.connect(self.run_search)
        self.reset_button.clicked.connect(self.reset_ui)
        self.graph_widget.node_selected_on_map.connect(self.update_selection_from_map)
        self.graph_widget.node_hover_changed.connect(self.update_image_preview) 
        self.start_combo.currentTextChanged.connect(self.update_selection_from_combo)
        self.goal_combo.currentTextChanged.connect(self.update_selection_from_combo)
        self.ai_button.clicked.connect(self.run_ai_query)
        self.ai_query_input.returnPressed.connect(self.run_ai_query)

    def display_node_image(self, node_name):
        if not node_name or 'Junction' in node_name:
            self.image_preview_label.setText("Select a goal to see an image.")
            self.image_preview_label.setPixmap(QPixmap()) 
            return

        safe_node_name = node_name.replace(' & ', '_and_').replace(' ', '_')
        image_path = os.path.join('images', f"{safe_node_name}.png")
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            self.image_preview_label.setPixmap(pixmap.scaled(
                self.image_preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.image_preview_label.setText("") 
        else:
            self.image_preview_label.setPixmap(QPixmap()) 
            self.image_preview_label.setText(f"No image for\n{node_name}")
            
    def update_image_preview(self, node_name, is_hovering):
        if is_hovering:
            if 'Junction' not in node_name:
                self.display_node_image(node_name)
        else:
            self.display_node_image(self.goal_node)

    def run_ai_query(self):
        query = self.ai_query_input.text()
        if not query:
            QMessageBox.warning(self, 'Empty Query', 'Please enter a query.')
            return
        
        self.ai_button.setDisabled(True)
        self.ai_query_input.setDisabled(True)
        self.ai_query_input.setPlaceholderText("BotBrain is thinking... 🤔")

        self.ai_thread = QThread()
        self.ai_worker = AIWorker(query)
        self.ai_worker.moveToThread(self.ai_thread)
        
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.result_ready.connect(self.handle_ai_result)
        self.ai_worker.error.connect(self.handle_ai_error)
        
        self.ai_thread.finished.connect(self.ai_thread.deleteLater)
        self.ai_worker.result_ready.connect(self.ai_thread.quit)
        self.ai_worker.error.connect(self.ai_thread.quit)

        self.ai_thread.start()

    def handle_ai_result(self, source, destination):
        self.ai_button.setDisabled(False)
        self.ai_query_input.setDisabled(False)
        self.ai_query_input.setPlaceholderText(self.original_placeholder)

        if source and destination:
            self.reset_ui()
            self.start_node, self.goal_node = source, destination
            self.sync_widgets()
            self.display_node_image(self.goal_node)
            QMessageBox.information(self, 'Query Understood', f'BotBrain understood your query!\nStart: {source}\nGoal: {destination}\n\nClick "Find Path" to continue.')
        else:
            QMessageBox.critical(self, 'Query Failed', 'BotBrain could not understand the locations. Please try again.')

    def handle_ai_error(self, error_message):
        self.ai_button.setDisabled(False)
        self.ai_query_input.setDisabled(False)
        self.ai_query_input.setPlaceholderText(self.original_placeholder)
        QMessageBox.critical(self, 'AI Error', f'An error occurred while parsing the query:\n{error_message}')

    def update_selection_mode(self, button): 
        self.selection_mode = button.text()
        self.algo_combo.setDisabled(self.selection_mode == 'Automatic')
    
    def update_selection_from_map(self, node_name, is_start):
        if is_start: 
            self.start_node, self.goal_node = node_name, None
        else:
            if node_name != self.start_node: 
                self.goal_node = node_name
        self.sync_widgets()
        self.display_node_image(self.goal_node)
        
    def update_selection_from_combo(self):
        start, goal = self.start_combo.currentText() or None, self.goal_combo.currentText() or None
        if start and start == goal: 
            self.goal_combo.blockSignals(True)
            self.goal_combo.setCurrentIndex(0)
            self.goal_combo.blockSignals(False)
            goal = None
        self.start_node, self.goal_node = start, goal
        self.sync_widgets()
        self.display_node_image(self.goal_node)
        
    def sync_widgets(self):
        self.start_combo.blockSignals(True)
        self.goal_combo.blockSignals(True)
        self.start_combo.setCurrentText(self.start_node or "")
        self.goal_combo.setCurrentText(self.goal_node or "")
        self.start_combo.blockSignals(False)
        self.goal_combo.blockSignals(False)
        self.graph_widget.update_selection(self.start_node, self.goal_node)
        
    def reset_ui(self):
        self.update_info_label(self.path_label, 'Path:', 'N/A')
        self.update_info_label(self.cost_label, 'Distance:', 'N/A')
        self.update_info_label(self.algo_info_label, 'Algorithm:', 'N/A')
        self.update_info_label(self.info_label, 'Info:', 'N/A')
        self.start_node, self.goal_node = None, None
        self.display_node_image(self.goal_node)
        self.graph_widget.highlight_path([])
        self.auto_button.setChecked(True)
        self.update_selection_mode(self.auto_button)
        self.sync_widgets()
        
    def run_search(self):
        if not self.start_node or not self.goal_node: 
            QMessageBox.warning(self, 'Selection Incomplete', 'Please select a start and a goal.')
            return
        if self.selection_mode == 'Manual': 
            self.run_manual_search()
        else: 
            self.run_automatic_search()
        
    def run_manual_search(self):
        algo_name = self.algo_combo.currentText()
        path, cost = [], float('inf')
        if algo_name == 'A*':
            positions = get_node_positions()
            heuristic_for_astar = create_heuristic(positions, self.goal_node)
            path, cost = astar(self.graph, self.start_node, self.goal_node, heuristic_for_astar)
        else:
            algo_func = {'BFS': bfs, 'UCS': ucs, 'Dijkstra': dijkstra}.get(algo_name)
            if not algo_func: 
                QMessageBox.warning(self, 'Algorithm Error', 'Could not find function.')
                return
            path, cost = algo_func(self.graph, self.start_node, self.goal_node)
        self.display_results(path, cost, algo_name)
        
    def run_automatic_search(self):
        start, goal, graph = self.start_node, self.goal_node, self.graph
        positions = get_node_positions()
        heuristic_for_astar = create_heuristic(positions, goal)
        all_results = []
        path_d, cost_d = dijkstra(graph, start, goal)
        if path_d: all_results.append(("Dijkstra", path_d, cost_d))
        path_a, cost_a = astar(graph, start, goal, heuristic_for_astar)
        if path_a: all_results.append(("A*", path_a, cost_a))
        path_u, cost_u = ucs(graph, start, goal)
        if path_u: all_results.append(("UCS", path_u, cost_u))
        
        if not all_results:
            self.display_results([], float('inf'), "Automatic (No Path Found)")
            return

        best_algorithm, best_path, best_cost = min(all_results, key=lambda item: item[2])
        self.display_results(best_path, best_cost, f"{best_algorithm} (Automatic Best)")

    def update_info_label(self, label, title, value):
        label.setText(f'<b>{title}</b> {value}')

    def display_results(self, path, cost, algo_name):
        self.graph_widget.highlight_path(path)
        if path:
            self.update_info_label(self.path_label, 'Path:', ' → '.join(path))
            cost_unit = "steps" if "BFS" in algo_name else "meters"
            self.update_info_label(self.cost_label, 'Distance:', f"{int(cost)} {cost_unit}")
            self.update_info_label(self.algo_info_label, 'Algorithm:', f"{algo_name}")
            self.update_info_label(self.info_label, 'Info:', f"{ai.get_building_info(self.goal_node)}")
        else:
            self.update_info_label(self.path_label, 'Path:', 'No path found.')
            self.update_info_label(self.cost_label, 'Distance:', 'Infinity')
            self.update_info_label(self.algo_info_label, 'Algorithm:', f"Attempted with {algo_name}")
            self.update_info_label(self.info_label, 'Info:', 'N/A')

def main():
    app = QApplication(sys.argv)
    dark_stylesheet = """
        QWidget { 
            background-color: #2e2e2e; 
            color: #f0f0f0; 
            font-family: Segoe UI; 
            font-size: 14px;
        }
        QPushButton { 
            background-color: #555; 
            border: 1px solid #777; 
            padding: 8px 12px; 
            border-radius: 4px; 
            font-size: 14px; 
            min-height: 24px;
        }
        QPushButton:hover { background-color: #666; } 
        QPushButton:pressed, QPushButton:checked { 
            background-color: #007acc; 
            border: 1px solid #005c99; 
            color: white; 
        }
        QPushButton:disabled { 
            background-color: #444; 
            color: #888; 
            border-color: #555; 
        }
        QPushButton#findButton {
            background-color: #28a745;
            border-color: #1e7e34;
            font-weight: bold;
        }
        QPushButton#findButton:hover { background-color: #218838; }
        QPushButton#resetButton {
            background-color: #dc3545;
            border-color: #bd2130;
            font-weight: bold;
        }
        QPushButton#resetButton:hover { background-color: #c82333; }
        QComboBox { 
            background-color: #444; 
            border: 1px solid #666; 
            border-radius: 4px; 
            padding: 5px; 
            font-size: 14px; 
        }
        QComboBox:disabled { 
            background-color: #3a3a3a; 
            color: #888; 
        } 
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { 
            background-color: #444; 
            selection-background-color: #007acc; 
            border: 1px solid #666; 
        }
        QLabel { font-size: 14px; } 
        QLineEdit { 
            border: 1px solid #666; 
            background-color: #444; 
            border-radius: 4px; 
            padding: 8px; 
        }
        QLineEdit:disabled { 
            background-color: #3a3a3a; 
            color: #888; 
        }
        QGraphicsView { background-color: #212121; } 
        QMessageBox { background-color: #3e3e3e; }
        QLabel#imagePreview {
            border: 1px solid #666; 
            border-radius: 4px; 
            color: #aaa;
        }
        QLabel#imagePreviewSubtext {
            font-size: 12px;
            color: #888;
        }
    """
    app.setStyleSheet(dark_stylesheet)
    navigator = CampusNavigatorApp()
    navigator.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

