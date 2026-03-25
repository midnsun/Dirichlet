import sys
import subprocess
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QAbstractTableModel
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtGui import QFont

task_a = 1.0 
task_b = 2.0
task_c = 1.0 
task_d = 2.0

# ---------------------------
# TABLE MODEL (efficient)
# ---------------------------

class GridModel(QAbstractTableModel):

    def __init__(self, grid):
        super().__init__()
        self.grid = grid

    def rowCount(self, parent=None):
        return self.grid.shape[0]

    def columnCount(self, parent=None):
        return self.grid.shape[1]

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            val = self.grid[index.row(), index.column()]
            return f"{val:.3e}"


# ---------------------------
# TABLE WINDOW
# ---------------------------

class TableWindow(QMainWindow):

    def __init__(self, grid, title, mode):
        super().__init__()

        self.setWindowTitle(title)
        # table = QTableView()
        # table.setModel(GridModel(grid))

        rows, cols = grid.shape
        table = QTableWidget(rows + 1, cols + 1)
        x = np.linspace(task_a, task_b, cols)

        table.setHorizontalHeaderItem(0, QTableWidgetItem("i"))
        for i in range(cols):
            table.setHorizontalHeaderItem(i + 1, QTableWidgetItem(f"{i}"))
            # table.setItem(0, i+2, QTableWidgetItem(str(i)))          # index
            table.setItem(0, i+1, QTableWidgetItem(f"{x[i]:.3f}"))   # coordinate

        y = np.linspace(task_c, task_d, rows)

        table.setVerticalHeaderItem(0, QTableWidgetItem("j"))
        for j in range(rows):
            table.setVerticalHeaderItem(rows - j, QTableWidgetItem(f"{j}"))
            # table.setItem(rows + 1 - j, 0, QTableWidgetItem(str(j)))          # index
            table.setItem(rows - j, 0, QTableWidgetItem(f"{y[j]:.3f}"))   # coordinate

        for j in range(rows):
            for i in range(cols):
                if mode == 'f':
                    table.setItem(rows - j, i+1, QTableWidgetItem(f"{grid[j, i]:.3f}"))
                elif mode == 'e':
                    table.setItem(rows - j, i+1, QTableWidgetItem(f"{grid[j, i]:.3e}"))

        table.setItem(0,0,QTableWidgetItem("yj\txi"))
           
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setCentralWidget(table)
        self.resize(900, 600)


# ---------------------------
# 3D SURFACE WINDOW
# ---------------------------

class SurfaceWindow(QMainWindow):

    def __init__(self, grid, title, a, b, c, d, func_name):
        super().__init__()

        self.grid = grid
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.func_name = func_name

        self.setWindowTitle(title)
        self.resize(1200, 900)

        widget = QWidget()
        layout = QVBoxLayout()

        self.plotter = QtInteractor(widget)

        layout.addWidget(self.plotter)
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.draw_surface(grid)

    def draw_surface(self, grid):

        self.plotter.clear()
        # downsample if grid is too big
        max_points = 200
        step = max(1, max(grid.shape)//max_points)
        g = grid[::step, ::step].astype(np.float32)

        plate_size = max(self.b - self.a, self.d - self.c)
        z_scale = plate_size / np.max(np.abs(grid))
        g_scaled = g * z_scale

        ny, nx = g.shape
        x = np.linspace(self.a, self.b, nx, dtype=np.float32)
        y = np.linspace(self.c, self.d, ny, dtype=np.float32)

        X, Y = np.meshgrid(x, y)

        surf = pv.StructuredGrid(X, Y, g_scaled)
        zmin = float(np.min(g))
        zmax = float(np.max(g))

        self.plotter.add_mesh(surf, cmap="viridis", smooth_shading=True, show_scalar_bar=True) # show_scalar_bar=False
        self.plotter.show_bounds(
            bounds=[self.a, self.b, self.c, self.d, zmin * z_scale, zmax * z_scale],
            grid='back',
            location='outer',
            xtitle='x',
            ytitle='y',
            ztitle=f"{self.func_name} * {z_scale:.2e}",
        )
        self.plotter.reset_camera()

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except:
            pass
        event.accept()


# ---------------------------
# MAIN WINDOW
# ---------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.child_windows = []

        self.setWindowTitle("Основное окно интерфейса")
        self.resize(1400, 700)

        main_layout = QVBoxLayout()

        # -------------------
        # PARAMETERS
        # -------------------

        self.main_task_enabled = False
        self.n = QLineEdit("100")
        self.m = QLineEdit("100")

        self.task = QComboBox()
        self.task.addItems(["Тестовая задача", "Основная задача"])
        self.task.currentIndexChanged.connect(self.update_task)

        self.interp = QComboBox()
        self.interp.addItems(["Нулевое", "Интерполяция по x", "Интерполяция по y", "Среднее"])

        self.eps = QLineEdit("1e-12")
        self.eps_r = QLineEdit("1e-16")
        self.maxN = QLineEdit("25000")

        self.eps2 = QLineEdit("1e-15")
        self.eps_r2 = QLineEdit("1e-17")
        self.maxN2 = QLineEdit("100000")

        self.test_task = QLabel("u(a,y) = exp(1-y²)   u(b,y) = exp(4-y²)   y ∈ [c, d]\nu(x,c) = exp(x²-1)   u(x,d) = exp(x²-4)   x ∈ [a, b]\nf(x,y) = 4(x²+y²)exp(x²-y²)")
        self.main_task = QLabel("u(a,y) = 0             u(b,y) = 0,                          y ∈ [c, d]\nu(x,c) = sin²(πx)   u(x,d) = ch((x-1)(x-2)) - 1   x ∈ [a, b]\nf(x,y) = arctg(x/y)")

        # -------------------
        # PROBLEM BOX
        # -------------------

        problem_box = QGroupBox("Условие задачи:")

        problem_layout = QGridLayout()

        problem_layout.addWidget(QLabel(f"Вариант задачи 9. Выполнил студент группы 3823Б1ПМоп3 Загрядсков Максим"),0,0)
        problem_layout.addWidget(QLabel(f"Δu(x,y) = -f(x,y),  x ∈ (a, b),  y ∈ (c, d)"),1,0)
        problem_layout.addWidget(QLabel(f"a = {task_a}   b = {task_b}   c = {task_c}   d = {task_d}"),2,0)
        problem_layout.addWidget(QLabel(f"Используется метод сопряженных градиентов"),3,0)
        problem_layout.addWidget(QLabel("Выбор задания:"),4,0)
        problem_layout.addWidget(self.task,4,1)

        problem_layout.addWidget(self.test_task,5,0)
        problem_layout.addWidget(self.main_task,5,0)

        problem_box.setLayout(problem_layout)

        # -------------------
        # PARAMETERS BOX
        # -------------------

        params_box = QGroupBox("Параметры решения задачи:")

        params_layout = QGridLayout()

        params_layout.addWidget(QLabel("Число разбиений по x:"),0,0)
        params_layout.addWidget(self.n,0,1)

        params_layout.addWidget(QLabel("Число разбиений по y:"),0,2)
        params_layout.addWidget(self.m,0,3)

        params_layout.addWidget(QLabel("Начальное приближение:"),1,0)
        params_layout.addWidget(self.interp,1,2)

        params_layout.addWidget(QLabel("Точность метода:"),2,0)
        params_layout.addWidget(self.eps,2,1)

        params_layout.addWidget(QLabel("Точность метода по невязке:"),2,2)
        params_layout.addWidget(self.eps_r,2,3)

        params_layout.addWidget(QLabel("Ограничение шагов:"),2,4)
        params_layout.addWidget(self.maxN,2,5)

        params_layout.addWidget(QLabel("Точность метода\n(двойная сетка):"),3,0)
        params_layout.addWidget(self.eps2,3,1)

        params_layout.addWidget(QLabel("Точность метода по невязке\n(двойная сетка):"),3,2)
        params_layout.addWidget(self.eps_r2,3,3)

        params_layout.addWidget(QLabel("Ограничение шагов\n(двойная сетка):"),3,4)
        params_layout.addWidget(self.maxN2,3,5)

        params_box.setLayout(params_layout)

        # -------------------
        # STATS BOX
        # -------------------

        stats_box = QGroupBox("Справка")

        stats_layout = QVBoxLayout()

        self.stats_text = QLabel()
        self.stats_text.setWordWrap(True)
        self.stats_text.setMinimumHeight(120)

        stats_layout.addWidget(self.stats_text)

        stats_box.setLayout(stats_layout)

        middle_layout = QGridLayout()
    
        middle_layout.addWidget(problem_box,0,0)
        middle_layout.addWidget(params_box,1,0)
        middle_layout.addWidget(stats_box,0,1,2,1)
        middle_layout.setColumnStretch(0,3)
        middle_layout.setColumnStretch(1,2)

        # -------------------
        # BUTTONS
        # -------------------

        btn_box = QGroupBox("Кнопки")
        btn_layout = QGridLayout()

        self.run_button = QPushButton("Решить")

        self.btn_table_x = QPushButton("Таблица v(N)(xi,yj)")
        self.btn_table_ex_0 = QPushButton("Таблица u(xi,yj)")
        self.btn_table_diff_0 = QPushButton("Таблица u(xi,yj) - v(N)(xi,yj)")
        self.btn_table_ex_1 = QPushButton("Таблица v2(N2)(x2i,y2j)")
        self.btn_table_diff_1 = QPushButton("Таблица v(N)(xi,yj) - v2(N2)(x2i,y2j)")

        self.btn_surface_x = QPushButton("График v(N)(xi,yj)")
        self.btn_surface_ex_0 = QPushButton("График u(xi,yj)")
        self.btn_surface_diff_0 = QPushButton("График u(xi,yj) - v(N)(xi,yj)")
        self.btn_surface_ex_1 = QPushButton("График v2(N2)(x2i,y2j)")
        self.btn_surface_diff_1 = QPushButton("График v(N)(x,y) - v2(N2)(x2i,y2j)")
        self.btn_surface_x_interp = QPushButton("График v(0)(xi,yj)")
        self.btn_surface_ex_interp = QPushButton("График v2(0)(x2i,y2j)")

        btn_layout.addWidget(self.run_button,0,0)

        btn_layout.addWidget(self.btn_table_x,1,0)
        btn_layout.addWidget(self.btn_table_ex_0,1,1)
        btn_layout.addWidget(self.btn_table_diff_0,1,2)
        btn_layout.addWidget(self.btn_table_ex_1,1,1)
        btn_layout.addWidget(self.btn_table_diff_1,1,2)

        btn_layout.addWidget(self.btn_surface_x,2,0)
        btn_layout.addWidget(self.btn_surface_ex_0,2,1)
        btn_layout.addWidget(self.btn_surface_diff_0,2,2)
        btn_layout.addWidget(self.btn_surface_ex_1,2,1)
        btn_layout.addWidget(self.btn_surface_diff_1,2,2)
        btn_layout.addWidget(self.btn_surface_x_interp,2,3)
        btn_layout.addWidget(self.btn_surface_ex_interp,2,4)

        btn_box.setLayout(btn_layout)

        # -------------------
        # LAYOUTS
        # -------------------

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # main_layout.addWidget(problem_box)
        main_layout.addLayout(middle_layout,3)
        main_layout.addWidget(btn_box)

        # -------------------
        # OTHERS
        # -------------------        

        self.update_task()

        self.run_button.clicked.connect(self.run_solver)

        self.btn_table_x.clicked.connect(self.open_table_x)
        self.btn_table_ex_0.clicked.connect(self.open_table_ex)
        self.btn_table_diff_0.clicked.connect(self.open_table_diff)
        self.btn_table_ex_1.clicked.connect(self.open_table_ex)
        self.btn_table_diff_1.clicked.connect(self.open_table_diff)

        self.btn_surface_x.clicked.connect(self.open_surface_x)
        self.btn_surface_ex_0.clicked.connect(self.open_surface_ex)
        self.btn_surface_diff_0.clicked.connect(self.open_surface_diff)
        self.btn_surface_ex_1.clicked.connect(self.open_surface_ex)
        self.btn_surface_diff_1.clicked.connect(self.open_surface_diff)
        self.btn_surface_x_interp.clicked.connect(self.open_surface_x_interp)
        self.btn_surface_ex_interp.clicked.connect(self.open_surface_ex_interp)


    def update_task(self):

        task = self.task.currentIndex()

        enabled = task == 1
        self.main_task_enabled = enabled

        self.eps2.setEnabled(enabled)
        self.eps_r2.setEnabled(enabled)
        self.maxN2.setEnabled(enabled)
        self.btn_surface_ex_interp.setEnabled(enabled)
        self.main_task.setEnabled(enabled)
        self.test_task.setEnabled(not enabled)
        self.main_task.setVisible(enabled)
        self.test_task.setVisible(not enabled)

        self.btn_table_ex_1.setEnabled(enabled)
        self.btn_table_ex_0.setEnabled(not enabled)
        self.btn_table_ex_1.setVisible(enabled)
        self.btn_table_ex_0.setVisible(not enabled)

        self.btn_table_diff_1.setEnabled(enabled)
        self.btn_table_diff_0.setEnabled(not enabled)
        self.btn_table_diff_1.setVisible(enabled)
        self.btn_table_diff_0.setVisible(not enabled)

        self.btn_surface_ex_1.setEnabled(enabled)
        self.btn_surface_ex_0.setEnabled(not enabled)
        self.btn_surface_ex_1.setVisible(enabled)
        self.btn_surface_ex_0.setVisible(not enabled)

        self.btn_surface_diff_1.setEnabled(enabled)
        self.btn_surface_diff_0.setEnabled(not enabled)
        self.btn_surface_diff_1.setVisible(enabled)
        self.btn_surface_diff_0.setVisible(not enabled)

    # -------------------
    # RUN EXECUTABLE
    # -------------------

    def run_solver(self):

        task = self.task.currentIndex()

        interp = str()
        if (self.interp.currentText() == "Нулевое"):
            interp = "0"
        elif (self.interp.currentText() == "Интерполяция по x"):
            interp = "1"
        elif (self.interp.currentText() == "Интерполяция по y"):
            interp = "2"
        elif (self.interp.currentText() == "Среднее"):
            interp = "3"

        cmd = [
            "C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/executable.exe",
            self.n.text(),
            self.m.text(),
            str(task),
            interp,
            self.eps.text(),
            self.eps_r.text(),
            self.maxN.text(),
        ]

        if task == 1:
            cmd += [
                self.eps2.text(),
                self.eps_r2.text(),
                self.maxN2.text()
            ]
        elif task == 0:
            cmd += [
                "0.0",
                "0.0",
                "0"
            ]

        subprocess.run(cmd)

        self.load_results()

    # -------------------
    # LOAD RESULTS
    # -------------------

    def load_results(self):

        with open("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data.txt") as f:
            lines = [l.strip() for l in f.readlines()]

        nx = int(lines[0])
        ny = int(lines[1])

        error = lines[2]
        err_i = lines[3]
        err_j = lines[4]

        stats = self.make_stat_text(lines)

        self.stats_text.setText(stats)
        self.stats_text.setTextFormat(Qt.RichText)

        self.grid_x = np.fromfile("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data_x.bin", dtype=np.float64).reshape((ny, nx))
        self.grid_ex = np.fromfile("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data_example.bin", dtype=np.float64).reshape((ny, nx))
        self.grid_diff = np.fromfile("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data_diff.bin", dtype=np.float64).reshape((ny, nx))
        self.grid_x_interp = np.fromfile("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data_x_interp.bin", dtype=np.float64).reshape((ny, nx))
        self.grid_ex_interp = np.fromfile("C:/Users/chehp/OneDrive/Desktop/all/Numerical_methods/lab5/data/data_example_interp.bin", dtype=np.float64).reshape((ny, nx))


    def open_table_x(self):
        mode = ''
        if (self.task.currentIndex() == 0):
            mode = 'f'
        else:
            mode = 'f'
        w = TableWindow(self.grid_x, "v(N)(xi,yj)", mode)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_table_ex(self):
        func_name = ""
        mode = ''
        if (self.task.currentIndex() == 0):
            func_name = "u(xi,yj)"
            mode = 'f'
        else:
            func_name = "v2(N2)(xi,yj)"
            mode = 'f'
        w = TableWindow(self.grid_ex, func_name, mode)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_table_diff(self):
        func_name = ""
        mode = ''
        if (self.task.currentIndex() == 0):
            func_name = "u(xi,yj) - v(N)(xi,yj)"
            mode = 'e'
        else:
            func_name = "v(N)(xi,yj) - v2(N2)(x2i,y2j)"
            mode = 'e'
        w = TableWindow(self.grid_diff, func_name, mode)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()


    def open_surface_x(self):
        w = SurfaceWindow(self.grid_x, "v(N)(x,y)", task_a, task_b, task_c, task_d, "v(N)(x,y)")
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_surface_ex(self):
        func_name = ""
        if (self.task.currentIndex() == 0):
            func_name = "u(xi,yj)"
        else:
            func_name = "v2(N2)(x2i,y2j)"
        w = SurfaceWindow(self.grid_ex, func_name, task_a, task_b, task_c, task_d, func_name)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_surface_diff(self):
        func_name = ""
        if (self.task.currentIndex() == 0):
            func_name = "u(xi,yj) - v(N)(xi,yj)"
        else:
            func_name = "v(N)(xi,yj) - v2(N2)(x2i,y2j)"
        w = SurfaceWindow(self.grid_diff, func_name, task_a, task_b, task_c, task_d, func_name)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_surface_x_interp(self):
        func_name = "v(0)(xi,yj)"
        w = SurfaceWindow(self.grid_x_interp, func_name, task_a, task_b, task_c, task_d, func_name)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def open_surface_ex_interp(self):
        func_name = ""
        if (self.task.currentIndex() == 0):
            func_name = "u(x,y)"
        else:
            func_name = "v2(0)(x2i,y2j)"
        w = SurfaceWindow(self.grid_ex_interp, func_name, task_a, task_b, task_c, task_d, func_name)
        self.child_windows.append(w)
        w.destroyed.connect(lambda: self.child_windows.remove(w))
        w.show()

    def make_stat_text(self, lines):
        stats = str()
            
        if (self.main_task_enabled == False):
            stats += f"Для решения тестовой задачи использована сетка с параметрами:<br>"
            stats += f"<b>n = {self.n.text()}</b> - число разбиений по x; <b>m = {self.m.text()}</b> - число разбиений по y<br>"
            #stats += f"Использовался метод сопряженных градиентов<br>"
            stats += f"Точность метода: <b>ε<sub>метода</sub> = {float(self.eps.text()):.4e}</b><br><br>"

            stats += f"На решение затрачено: <b>N = {lines[7]}</b> шагов<br>"
            stats += f"На решение затрачено: <b>t = {float(lines[8]):.4f} с</b> времени<br>"
            stats += f"Достингуная точность метода: <b>ε<sub>N</sub> = {float(lines[5]):.4e}</b><br>"
            stats += f"Невязка решения в евклидовой норме: <b>||r<sup>(N)</sup>||<sub>2</sub> = {float(lines[6]):.4e}</b><br>"
            stats += f"Невязка на нулевом шаге в евклидовой норме: <b>||r<sup>(0)</sup>||<sub>2</sub> = {float(lines[13]):.4e}</b><br><br>"

            stats += f"Тестовая задача решена с погрешностью:<br>"
            stats += f"<b>max|v<sup>(N)</sup>(x<sub>i</sub>, y<sub>j</sub>) - u(x<sub>i</sub>, y<sub>j</sub>)| = {float(lines[2]):.4e}</b>, i ∈ [1,n-1], j ∈ [1,m-1]<br>"
            stats += f"Максимум достигнут в точке <b>x<sub>i</sub> = {float(lines[3]):.4e}</b>, <b>y<sub>j</sub> = {float(lines[4]):.4e}</b>,<br>"
            stats += f"В качестве начального приближения "
            interp = str()
            if (self.interp.currentText() == "Нулевое"):
                interp = "использовано <b>нулевое</b>"
            elif (self.interp.currentText() == "Интерполяция по x"):
                interp = "использована <b>интерполяция по x</b>"
            elif (self.interp.currentText() == "Интерполяция по y"):
                interp = "использована <b>интерполяция по y</b>"
            elif (self.interp.currentText() == "Среднее"):
                interp = "использовано <b>среднее</b>"
            stats += interp+"<br>"

        if (self.main_task_enabled == True):
            stats += f"Для решения основной задачи использована сетка с параметрами:<br>"
            stats += f"<b>n = {self.n.text()}</b> - число разбиений по x; <b>m = {self.m.text()}</b> - число разбиений по y<br>"
            #stats += f"Использовался метод сопряженных градиентов<br>"
            stats += f"Точность метода: <b>ε<sub>метода</sub> = {float(self.eps.text()):.4e}</b><br>"
            stats += f"Точность метода для двойной сетки: <b>ε2<sub>метода</sub> = {float(self.eps2.text()):.4e}</b><br><br>"

            stats += f"На решение затрачено: <b>N = {lines[7]}</b> шагов<br>"
            stats += f"На решение затрачено: <b>t = {float(lines[8]):.4f} с</b> времени<br>"
            stats += f"Достингуная точность метода: <b>ε<sub>N</sub> = {float(lines[5]):.4e}</b><br>"
            stats += f"Невязка решения в евклидовой норме: <b>||r<sup>(N)</sup>||<sub>2</sub> = {float(lines[6]):.4e}</b><br>"
            stats += f"Невязка на нулевом шаге в евклидовой норме: <b>||r<sup>(0)</sup>||<sub>2</sub> = {float(lines[13]):.4e}</b><br><br>"

            stats += f"На решение на двойной сетке затрачено: <b>N2 = {lines[11]}</b> шагов<br>"
            stats += f"На решение на двойной сетке затрачено: <b>t2 = {float(lines[12]):.4f} с</b> времени<br>"
            stats += f"Достингуная точность метода: <b>ε2<sub>N2</sub> = {float(lines[9]):.4e}</b><br>"
            stats += f"Невязка решения в евклидовой норме: <b>||r<sup>(N2)</sup>||<sub>2</sub> = {float(lines[10]):.4e}</b><br>"
            stats += f"Невязка на нулевом шаге в евклидовой норме: <b>||r<sup>(0)</sup>||<sub>2</sub> = {float(lines[14]):.4e}</b><br><br>"

            stats += f"Основная задача решена с точностью:<br>"
            stats += f"<b>max|v<sup>(N)</sup>(x<sub>i</sub>, y<sub>j</sub>) - v2<sup>(N2)</sup>(x<sub>2i</sub>, y<sub>2j</sub>)| = {float(lines[2]):.4e}</b>, i ∈ [1,n-1], j ∈ [1,m-1]<br>"
            stats += f"Достигнута в точке <b>x<sub>i</sub> = {float(lines[3]):.4e}</b>, <b>y<sub>j</sub> = {float(lines[4]):.4e}</b>,<br>" 
            stats += f"В качестве начального приближения "
            interp = str()
            if (self.interp.currentText() == "Нулевое"):
                interp = "использовано <b>нулевое</b>"
            elif (self.interp.currentText() == "Интерполяция по x"):
                interp = "использована <b>интерполяция по x</b>"
            elif (self.interp.currentText() == "Интерполяция по y"):
                interp = "использована <b>интерполяция по y</b>"
            elif (self.interp.currentText() == "Среднее"):
                interp = "использовано <b>среднее</b>"
            stats += interp+" для подсчёта на базовой и контрольной сетках<br>"

        return stats

# ---------------------------
# MAIN
# ---------------------------

app = QApplication(sys.argv)

font = QFont()
font.setPointSize(12)   # try 12–14
app.setFont(font)

window = MainWindow()
window.show()

sys.exit(app.exec())