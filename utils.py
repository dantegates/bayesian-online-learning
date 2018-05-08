"""
see,
https://matplotlib.org/examples/animation/histogram.html
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation


TITLE_TEMPLATE = '{n}\n{pct:0.2f}\n{pred}\n{max}'


class Distribution:
    def __init__(self, ax, discrete=False):
        self.ax = ax
        self.discrete = discrete
    
    def update(self, dist):
        x, p = self._get_plot_points(dist)
        self.ax.plot(x, p, color='b')
        
    def clear(self):
        for line in self.ax.get_lines():
            line.remove()   

    def _get_plot_points(self, dist):
        start, stop = dist.interval(0.99)
        if self.discrete:
            x = range(int(start), int(stop)+1)
            p = dist.pmf(x)
        else:
            x = np.linspace(start, stop)
            p = dist.pdf(x)
        return x, p


class Histogram:
    def __init__(self, ax, arr):
        self.ax = ax
        self.init_hist(self.ax, arr)

    def init_hist(self, ax, arr):
        n, bins = np.histogram(arr, 40)

        # get the corners of the rectangles for the histogram
        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        top = bottom + n
        nrects = len(left)

        # here comes the tricky part -- we have to set up the vertex and path
        # codes arrays using moveto, lineto and closepoly

        # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
        # CLOSEPOLY; the vert for the closepoly is ignored but we still need
        # it to keep the codes aligned with the vertices
        nverts = nrects*(1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom

        barpath = path.Path(verts, codes)
        patch = patches.PathPatch(barpath, alpha=0.5)
        ax.add_patch(patch)

        ax.set_xlim(left[0], right[-1])
        ax.set_ylim(bottom.min(), top.max())
        
        self.verts = verts
        self.top = top
        self.bottom = bottom
        self.patch = patch

    def update(self, arr):
        n, bins = np.histogram(arr, 40)
        top = self.bottom + n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top
        
    def clear_lines(self):
        for line in self.ax.get_lines():
            line.remove()     
        
    def plot_line(self, x, color, label):
        self.ax.axvline(x, color=color, label=label)

def visualize_learning(observations, predictions, cum_max=None):
    fig, ax = plt.subplots()

    obs_hist = Histogram(ax, observations)

    def animate(i):
        obs_hist.update(observations[:i])
        obs_hist.clear_lines()
        obs_hist.plot_line(predictions[i], color='r', label='est. max')
        if cum_max is not None:
            obs_hist.plot_line(cum_max[i], color='b', label='max')
        plt.title(i+1, x=1.05, y=0.5)
        return [obs_hist.patch, ]

    ani = animation.FuncAnimation(fig, animate, len(observations),
                                  repeat=False, blit=True, interval=30)
    return ani
        
def visualize_online_learning(observations, predictions, model_params, true_dists,
                              confidence, last_n=None):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 6))
    ax_obs = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=4)
    ax_true = plt.subplot2grid((3, 6), (0, 4), colspan=2)
    ax_poi = plt.subplot2grid((3, 6), (1, 4), colspan=2, sharex=ax_true)
    ax_gamma = plt.subplot2grid((3, 6), (2, 4), colspan=2, sharex=ax_true)
    
    ax_obs.set_title('observations')
    ax_true.set_title('generating distribution')
    ax_poi.set_title('lambda posterior')
    ax_gamma.set_title('gamma posterior')

    true_dist = Distribution(ax_true, discrete=True)
    poi_dist = Distribution(ax_poi, discrete=True)
    gamma_dist = Distribution(ax_gamma)

    plt.tight_layout()

    if last_n is None:
        last_n = len(observations)
    obs_hist = Histogram(ax_obs, observations[:3*last_n])

    def animate(i):
        start = 0 if i < last_n else i - last_n
        stop = i
        obs = observations[start:stop+1]
        obs_hist.update(obs)
        
        true = true_dists[i]
        gamma, poisson = model_params[i]
        true_dist.clear(), poi_dist.clear(), gamma_dist.clear()
        true_dist.update(true)
        poi_dist.update(poisson)
        gamma_dist.update(gamma)

        pred = predictions[i]
        true_max_with_confidence = true.ppf(confidence)
        obs_hist.clear_lines()
        obs_hist.plot_line(pred, color='r', label='true')
        obs_hist.plot_line(true_max_with_confidence, color='b', label='prediction')

        title = TITLE_TEMPLATE.format(
            n=(i+1),
            pct=((obs < pred).sum() / len(obs)),
            pred=pred,
            max=max(obs))
        plt.title(title, x=1.1, y=0.6)
        return [obs_hist.patch, ]

    ani = animation.FuncAnimation(fig, animate, len(observations),
                                  repeat=False, blit=True, interval=100)
    return ani
