"""Shared styles for the three manuscript figures: fonts, line widths, and colors."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FONT_SIZE = 14
AXIS_LABEL_SIZE = 16
PANEL_TITLE_SIZE = 17
TICK_SIZE = 13
LEGEND_SIZE = 12
DPI = 350
SOLAR = '#D08A24'
WIND = '#188878'
HYBRID = '#344F79'
TEXT = '#26333E'


def setup():
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'], 'font.size': FONT_SIZE,
        'axes.labelsize': AXIS_LABEL_SIZE, 'axes.titlesize': PANEL_TITLE_SIZE,
        'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 1, 'axes.edgecolor': '#71808A', 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def panel(ax, letter):
    ax.set_title(f'({letter})', loc='left', pad=16, fontweight='bold')
    ax.grid(axis='y', color='#DCE2E6', linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)


def save(fig, path):
    for extension in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + extension), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path.name}.png / .pdf')
