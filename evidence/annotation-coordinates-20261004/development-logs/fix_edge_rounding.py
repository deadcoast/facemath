from pathlib import Path
p=Path(__file__).resolve().parents[2]/'src/facemath/observation/annotation_ui.py'
s=p.read_text(encoding='utf-8')
old="const p=sourceAt(x,y);if(p[0]<-1e-7||p[1]<-1e-7||p[0]>width-1+1e-7||p[1]>height-1+1e-7)"
new="const p=sourceAt(x,y).map(v=>Math.round(v*1000)/1000);if(p[0]<0||p[1]<0||p[0]>width-1||p[1]>height-1)"
assert s.count(old)==1
p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
