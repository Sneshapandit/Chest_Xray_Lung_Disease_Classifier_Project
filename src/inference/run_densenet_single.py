from pathlib import Path
import runpy

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / 'DENSENET121_SINGLEIMAGE'
    runpy.run_path(str(target), run_name='__main__')
