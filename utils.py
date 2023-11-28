from numpy.fft import fft, ifft, fftfreq
import numpy as np

import matplotlib.pyplot as plt

from typing import Literal


class Converter:

    def __init__(self, **kwards):

        self.T = kwards.get('T')
        self.f = kwards.get('f')
        self.w = kwards.get('w')

        if self.T is not None:

            if any([isinstance(self.T, float),
                   isinstance(self.T, int)]):
                self.T = [self.T]

            self.f = [1 / self.T[i] for i in range(len(self.T))]
            self.w = [2 * np.pi / self.T[i] for i in range(len(self.T))]

        elif self.f is not None:

            if any([isinstance(self.f, float),
                   isinstance(self.f, int)]):
                self.f = [self.f]

            self.T = [1 / self.f[i] for i in range(len(self.f))]
            self.w = [2 * np.pi * self.f[i] for i in range(len(self.f))]

        elif self.w is not None:

            if any([isinstance(self.w, float),
                   isinstance(self.w, int)]):
                self.w = [self.w]

            self.f = [self.w[i] / (2 * np.pi) for i in range(len(self.w))]
            self.T = [1 / self.f[i] for i in range(len(self.w))]


class Signal_Handler:

    def convolve(self, signal: np.ndarray, kernel: np.ndarray, mode = 'full'):
        return np.convolve(signal, kernel, mode = mode) / np.sum(kernel)

    def find_spectrum(self, y: np.ndarray, dt: float):

        xfft = fftfreq(y.size) / dt
        yfft = fft(y)

        return xfft, yfft
    
    def from_spectrum(self, yfft: np.ndarray):
        return ifft(yfft).real
    
    def make_signal(self, 
                    func_name: Literal['cos', 'rect', 'triang', 'other'], 
                    interval: tuple, 
                    sample_rate: int = 1000,
                    **kwards):
        
        self._funcs = {'cos' : self._cos, 'rect' : self._rect, 'trinag' : self._triang, 'other' : kwards.get('func')}
        x, dt = np.linspace(interval[0], 
                            interval[1], 
                            num = sample_rate, 
                            endpoint = False, 
                            retstep = True)

        if func_name != 'other':

            w = kwards.get('w')
            A = kwards.get('A')

            if isinstance(w, float):
                w = [w]

            y = np.sum([
                self._funcs[func_name](x, A, w[t])
                for t in range(len(w))],
                axis = 0)
                
        else:
            y = self._funcs[func_name](x)
        
        return x, y, dt
    
    def _cos(self, x: float, A: float, w: float):
        return A * np.cos(w * x)
    
    def _rect(self, x: float, A: float, w: float):
        return A * np.sign(np.sin(w * x))
    
    def _triang(self, x: float, A: float, w: float):
        return 2 * A / np.pi * np.arcsin(np.sin(w * x))


class Signal_Display:
 
    def __init__(self):
        self.clear()

    def make_plot(self, ax_id, x: np.ndarray, y: np.ndarray, is_spectrum: bool = False, normalize = True, **kwards):

        interval = kwards.get('interval')   
        ids = range(x.size)

        if interval is not None:  
            ids = (x >= interval[0]) & (x <= interval[1])

        if is_spectrum:
            y[ids] = np.abs(y[ids])

            if normalize:
                y[ids] = 2 * y[ids] / y.size

        new_plot = dict(x = x[ids], y = y[ids], is_spectrum = is_spectrum, **kwards)

        if ax_id not in list(self.ax_desc.keys()):
            self.ax_desc[ax_id] = []

        self.ax_desc[ax_id].append(new_plot)

    def plot(self, **kwards):
        
        self.subplots_num = np.unique(list(self.ax_desc.keys())).size
        self.fig, self.ax = plt.subplots(self.subplots_num)

        if self.subplots_num == 1:
            self.ax = [self.ax]

        for ax_id in np.sort(list(self.ax_desc.keys())):
            for plot_desc in self.ax_desc[ax_id]:

                is_spectrum = plot_desc.get('is_spectrum')
                
                color = plot_desc.get('color', 'b')
                linestyle = plot_desc.get('linestyle')
                linewidth = plot_desc.get('linewidth')
                legend  = plot_desc.get('legend')
                alpha = plot_desc.get('alpha', 1)
                
                x_label = plot_desc.get('x_label', 'Время, с' if not is_spectrum else 'Частота, Гц')
                y_label = plot_desc.get('y_label', 'Амплитуда')
                title = plot_desc.get('title')
                
                x = plot_desc.get('x')
                y = plot_desc.get('y')

                self.ax[ax_id].plot(x, 
                                    y, 
                                    color = color, 
                                    linestyle = linestyle, 
                                    label = legend,
                                    linewidth = linewidth,
                                    alpha = alpha)

                if legend is not None:
                    self.ax[ax_id].legend()

                if title is not None:
                    self.ax[ax_id].set_title(title)

                self.ax[ax_id].set_xlabel(x_label)
                self.ax[ax_id].set_ylabel(y_label)
            
            self.ax[ax_id].grid()

        self.set_figure_settings(**kwards)

        self.fig.set_figwidth(self.fig_size[0])
        self.fig.set_figheight(self.fig_size[1])

        self.fig.suptitle(self.title, size = self.title_fontsize)
        self.fig.tight_layout()
        
        plt.show()

    def set_figure_settings(self, **kwards):

        self.title = kwards.get('title')
        self.title_fontsize = kwards.get('title_fontsize', 17)
        self.fig_size = kwards.get('fig_size', (12, 2.3 + 1.8 * (self.subplots_num - 1)))
        
    def clear(self):
        self.ax_desc = {}