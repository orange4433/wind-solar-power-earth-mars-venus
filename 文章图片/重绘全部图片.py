"""Generate the three main-figure drafts from any working directory."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    '模型核查/pv_efficiency_audit.py',
    '图1 火星/绘制图1_火星.py',
    '图2 金星/绘制图2_金星_context.py',
    '图3 地球（整体）/绘制图3_地球_四球.py',
]

if __name__ == '__main__':
    for filename in SCRIPTS:
        print(f'运行：{filename}')
        runpy.run_path(str(HERE/filename), run_name='__main__')
