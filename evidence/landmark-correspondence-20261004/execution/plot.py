from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path.cwd(); out=root/'evidence/landmark-correspondence-20261004'
a=json.loads((out/'source-audit.json').read_text())
rows=[r for r in a['cases'] if r['split']=='evaluation']
x=np.arange(len(rows));old=np.array([r['legacy_nose_error_eye_spans'] for r in rows]);new=np.array([r['error_eye_spans'][4] for r in rows])
fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
ax.vlines(x,np.minimum(old,new),np.maximum(old,new),color='#b4bbc2',lw=1.5)
ax.scatter(x,old,label='Historical MP1 assignment',color='#ad3b47',s=45,zorder=3)
ax.scatter(x,new,label='Prespecified MP4 assignment',color='#087e82',s=45,zorder=3)
ax.axhline(.05,color='#555',linestyle='--',lw=1,label='Fixed 0.05 threshold')
ax.set_xticks(x,[r['case_id'].replace('case_000','') for r in rows]);ax.set_xlabel('Evaluation case (all 20 enrolled / all 20 observed)')
ax.set_ylabel('Nose error / reference eye span');ax.set_ylim(bottom=0)
ax.set_title('Held-out nose correspondence: same native coordinates, different selected vertex',loc='left',fontsize=12)
ax.legend(loc='upper right',frameon=False,fontsize=9)
ax.spines[['right','top']].set_visible(False)
fig.savefig(out/'evaluation-nose-errors.png',dpi=170)
fig.savefig(out/'evaluation-nose-errors.svg')
plt.close(fig)
print('Created held-out error plot; no fitting or coordinate changes.')
