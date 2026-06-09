import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from config import DATA_DIR

mt = pd.read_csv(DATA_DIR / 'master_table.csv')
crp = mt[mt['task'] == 'creep']
sub = crp[(crp['test_temp'].between(700, 820)) &
          (crp['test_stress'].between(700, 900))]
print(f'T[700,820] sigma[700,900]: total={len(sub)}')
for th in [200, 300, 500, 700, 850, 1000, 1500]:
    print(f'  life>{th}: {len(sub[sub["target"] > th])}')
