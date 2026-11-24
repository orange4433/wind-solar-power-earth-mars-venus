"""Generate the three main-figure drafts from any working directory."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    'model_validation/pv_efficiency_audit.py',
    'figure_1_mars/plot_figure_1_mars.py',
    'figure_2_venus/plot_figure_2_venus_context.py',
    'figure_3_earth/plot_figure_3_earth_globes.py',
]

if __name__ == '__main__':
    for filename in SCRIPTS:
        print(f'Running: {filename}')
        runpy.run_path(str(HERE/filename), run_name='__main__')
