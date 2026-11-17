import sys, time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow

class ManagedWindow(QMainWindow):
    def __init__(self, on_close_callback):
        super().__init__()
        self.on_close_callback = on_close_callback

    def closeEvent(self, event):
        self.on_close_callback()
        event.accept()

class WhatevPlot():

    def __init__(
            self, 
            title, 
            titles, 
            buffer_size=1000, 
            break_after=3, 
            fps=10, 
            winx=100, 
            winy=100
        ):

        self.app = QApplication(sys.argv)
        self.win = ManagedWindow(on_close_callback=self.close)
        self.win.setWindowTitle(title)
        self.win.resize(winx, winy)

        self.gl_widget = pg.GraphicsLayoutWidget()
        self.win.setCentralWidget(self.gl_widget)
        self.buffer_size = buffer_size
        
        self.buffer_size = buffer_size
        self.curves = []
        self.datas = []

        self.x_data = np.zeros(self.buffer_size)

        self.win.show()
        self.update_delay = 1.0 / fps
        self.last_update_time = time.time()
        self._is_closing = False 

        pn = 1
        for title in titles:
            plot = self.gl_widget.addPlot(title=title)
            curve = plot.plot(pen=pg.mkPen('g', width=1.5))
            data = np.zeros(self.buffer_size)
            self.curves.append(curve)
            self.datas.append(data)
            if (pn % break_after) == 0:
                self.gl_widget.nextRow()
            pn+=1
        self.idx = 0
        self.ptr = 0

    def update(self, values):
        self.idx = self.ptr % self.buffer_size
        self.x_data[self.idx] = self.ptr
        for value, data in zip(values, self.datas):
            data[self.idx] = value
        self.ptr += 1

        current_time = time.time()
                
        if current_time - self.last_update_time >= self.update_delay:
            
            rolled_x = np.roll(self.x_data, -self.idx - 1)

            for curve, data in zip(self.curves, self.datas):
                rolled_y = np.roll(data, -self.idx - 1)
                curve.setData(x=rolled_x, y=rolled_y)
            
            QApplication.processEvents()
            self.last_update_time = current_time

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        
        print("\nshut down\n")
        self.win.destroy() 
        sys.exit(0)