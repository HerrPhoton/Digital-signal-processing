from numpy.fft import fft, ifft, fftfreq
import numpy as np

import matplotlib.pyplot as plt

from typing import Literal, Union


class Converter:

    def __init__(self, **kwards):

        self.T = kwards.get('T')
        self.f = kwards.get('f')
        self.w = kwards.get('w')

        if self.T is not None:
            self.f = 1 / self.T
            self.w = 2 * np.pi / self.T

        elif self.f is not None:
            self.T = 1 / self.f
            self.w = 2 * np.pi * self.f

        elif self.w is not None:
            self.f = self.w / (2 * np.pi)
            self.T = 1 / self.f


class Signal_Handler:

    def find_spectrum(self, y: np.ndarray, dt: float):

        xfft = fftfreq(y.size) / dt
        yfft = fft(y)

        return xfft, yfft
    
    def from_spectrum(self, yfft: np.ndarray):
        return ifft(np.abs(yfft))
    
    def make_signal(self, 
                    func_name: Literal['cos', 'rect', 'triang', 'other'], 
                    A: float, 
                    w: Union[float, list, tuple],
                    interval: tuple, 
                    sample_rate: int = 1000,
                    func = None):
        
        self._funcs = {'cos' : self._cos, 'rect' : self._rect, 'trinag' : self._triang, 'other' : func}
        x, dt = np.linspace(interval[0], 
                            interval[1], 
                            num = sample_rate, 
                            endpoint = False, 
                            retstep = True)

        if isinstance(w, float):
            y = np.apply_along_axis(lambda z: self._funcs[func_name](z, A, w), 0, x)
        else:
            y = np.sum([
                np.apply_along_axis(lambda z: self._funcs[func_name](z, A, w[t]), 0, x)
                for t in range(len(w))],
                axis = 0)
        
        return x, y, dt
    
    def _cos(self, x: float, A: float, w: float):
        return A * np.cos(w * x)
    
    def _rect(self, x: float, A: float, w: float):
        return A * np.sign(np.sin(w * x))
    
    def _triang(self, x: float, A: float, w: float):
        return 2 * A / np.pi * np.arcsin(np.sin(w * x))


class Signal_Display:

    def __init__(self, subplots_num: int = 1, title: str = None, title_fontsize = None, fig_size: tuple = None):
        
        self.subplots_num = subplots_num
        self.title = title
        self.title_fontsize = title_fontsize
        self.fig_size = fig_size

        self.ax_desc = {ax_id : [] for ax_id in range(subplots_num)}
 
    def make_plot(self, ax_id, x: np.ndarray, y: np.ndarray, is_spectrum: bool = False, **kwards):

        interval = kwards.get('interval')   
        ids = range(len(x))

        if interval is not None:   
            ids = np.logical_and(x >= interval[0], x <= interval[1])

        if is_spectrum:
            y[ids] = 2 * np.abs(y[ids]) / len(y)

        new_plot = dict(x = x[ids], y = y[ids], is_spectrum = is_spectrum, **kwards)
        self.ax_desc[ax_id].append(new_plot)

    def plot(self):

        self.fig, self.ax = plt.subplots(self.subplots_num, 1)

        if self.subplots_num == 1:
            self.ax = [self.ax]

        for ax_id in self.ax_desc.keys():
            for plot_desc in self.ax_desc[ax_id]:

                is_spectrum = plot_desc.get('is_spectrum')
                
                color = plot_desc.get('color')
                linestyle = plot_desc.get('linestyle')
                linewidth = plot_desc.get('linewidth')
                legend  = plot_desc.get('legend')
                
                x_label = plot_desc.get('x_label')
                y_label = plot_desc.get('y_label')
                title = plot_desc.get('title')
                
                x = plot_desc.get('x')
                y = plot_desc.get('y')

                if color is None:
                    color = 'b'

                self.ax[ax_id].plot(x, 
                                    y, 
                                    color = color, 
                                    linestyle = linestyle, 
                                    label = legend,
                                    linewidth = linewidth)

                if not is_spectrum:
                    if x_label is None:
                        self.ax[ax_id].set_xlabel('Время, с')
                    else: 
                        self.ax[ax_id].set_xlabel(x_label)

                    if y_label is None:
                        self.ax[ax_id].set_ylabel('Амплитуда')
                    else: 
                        self.ax[ax_id].set_ylabel(y_label)

                else:
                    if x_label is None:
                        self.ax[ax_id].set_xlabel('Частота, Гц')
                    else: 
                        self.ax[ax_id].set_xlabel(x_label)

                    if y_label is None:
                        self.ax[ax_id].set_ylabel('Амплитуда')
                    else: 
                        self.ax[ax_id].set_ylabel(y_label)

                if legend is not None:
                    self.ax[ax_id].legend()

                self.ax[ax_id].set_title(title)
            
            self.ax[ax_id].grid()

        if self.fig_size is not None:
            self.fig.set_figwidth(self.fig_size[0])
            self.fig.set_figheight(self.fig_size[1])

        self.fig.suptitle(self.title, size = self.title_fontsize)
        self.fig.tight_layout()
        
        plt.show()