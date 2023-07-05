import uproot
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

typelist = ['ch+', 'ch-', 'nh', 'ph', 'el+', 'el-', 'mu+', 'mu-']


def make_subplot(ax, data, force_xylim=None):
    # default plotting configuration
    color_dict_ = {'ch': 'C0', 'nh': 'mediumpurple', 'ph': 'orange', 'el': 'red', 'mu': 'green'}
    color_dict = color_dict_.copy()
    color_dict.update({k + '+': color_dict_[k] for k in color_dict_})
    color_dict.update({k + '-': color_dict_[k] for k in color_dict_})
    if data.get('id') is None:
        data['id'] = ['default'] * len(data['pt'])
    if data.get('e') is None:
        for eta, phi, pt, id, d3d in zip(data['eta'], data['phi'], data['pt'], data['id'], data['d3d']):
            ptdraw = np.sqrt(pt) / 200
            alpha = 0.3
            if id in [4, 5]:
                ax.add_patch(mpl.patches.RegularPolygon((eta, phi), 3, radius=ptdraw, clip_on=True,
                             alpha=alpha, edgecolor='black', **make_color_args(id, d3d)))
            elif id in [6, 7]:
                ax.add_patch(mpl.patches.RegularPolygon((eta, phi), 3, radius=ptdraw, orientation=np.pi,
                             clip_on=True, alpha=alpha, edgecolor='black', **make_color_args(id, d3d)))
            elif id in [3]:
                ax.add_patch(mpl.patches.RegularPolygon((eta, phi), 5, radius=ptdraw,
                             clip_on=True, alpha=alpha, **make_color_args(id, d3d)))
            else:
                ax.add_patch(plt.Circle((eta, phi), ptdraw, clip_on=True, alpha=alpha, **make_color_args(id, d3d)))
    else:
        for eta, phi, pt, e, id, d3d in zip(data['eta'], data['phi'], data['pt'], data['e'], data['id'], data['d3d']):
            ax.add_patch(mpl.patches.Wedge((eta, phi), pt / 600., 90, 270,
                         clip_on=True, alpha=alpha, **make_color_args(id, d3d)))
            ax.add_patch(mpl.patches.Wedge((eta, phi), e / 600., 270, 90,
                         clip_on=True, alpha=alpha, **make_color_args(id, d3d)))
    max_ang = force_xylim if force_xylim else max(max(abs(data['eta'])), max(abs(data['phi'])))
    # make square plot centered at (0,0)
    ax.set_xlim(-max_ang, max_ang)
    ax.set_ylim(-max_ang, max_ang)
    ax.set_xlabel(r'$\Delta\eta$')
    ax.set_ylabel(r'$\Delta\phi$')
    ax.set_aspect('equal')
    return max_ang


def make_color_args(id, d3d):
    color = color_fader('#74c476', '#081d58', d3d)
    if id in [2, 3]:
        return {'edgecolor': color, 'linewidth': 1, 'fill': False}
    else:
        return {'facecolor': color}


def color_fader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    mix = min(1., mix)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def visualize(arrays, idx=0, title=None, ax=None):
    data = {}
    data['pt'] = np.hypot(arrays[idx].part_px, arrays[idx].part_py)
    data['eta'] = arrays[idx].part_deta
    data['phi'] = arrays[idx].part_dphi
    data['d3d'] = np.tanh(np.hypot(arrays[idx].part_d0val, arrays[idx].part_dzval))
    part_type = np.concatenate([
        [(arrays[idx].part_isChargedHadron) & (arrays[idx].part_charge == 1)],
        [(arrays[idx].part_isChargedHadron) & (arrays[idx].part_charge == -1)],
        [arrays[idx].part_isNeutralHadron],
        [arrays[idx].part_isPhoton],
        [(arrays[idx].part_isElectron) & (arrays[idx].part_charge == 1)],
        [(arrays[idx].part_isElectron) & (arrays[idx].part_charge == -1)],
        [(arrays[idx].part_isMuon) & (arrays[idx].part_charge == 1)],
        [(arrays[idx].part_isMuon) & (arrays[idx].part_charge == -1)],
    ], axis=0)
    data['id'] = np.argmax(part_type.T, axis=1)  # better

    assert len(data['eta'] == data['id'])
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    make_subplot(ax, data, force_xylim=0.5)
    if title:
        ax.set_title(title)
    return ax
