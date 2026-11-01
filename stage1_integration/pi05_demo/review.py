"""Read-only browser for native episodes; listens on loopback only."""
import argparse,io,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse
import h5py
import numpy as np
from PIL import Image

HTML="""<!doctype html><meta charset="utf-8"><title>π0.5 插排示范检查</title>
<style>body{font:16px system-ui;background:#111820;color:#e9eef5;margin:32px;max-width:1100px}select,button{padding:10px;background:#233343;color:inherit;border:1px solid #567;border-radius:6px}img{width:42%;image-rendering:auto;border:1px solid #567}input{width:86%}pre{white-space:pre-wrap}small{color:#aab9c8}.views{display:flex;gap:20px}h1{font-size:24px}</style>
<h1>π0.5 · 插排示范检查</h1><p>外部 Kinect / 腕部 D405 · RGB · 只读</p>
<select id="episodes"></select> <button id="play">播放 / 暂停</button>
<p id="info"></p><div class="views"><img id="main"><img id="wrist"></div>
<p><input id="slider" type="range" min="0" value="0"><span id="index"></span></p>
<pre id="values"></pre><small>← / → 逐帧。成功/失败来自采集时的人工标记；mock 数据不能用于真实训练。</small>
<script>
let items=[],timer=null;const e=document.getElementById('episodes'),s=document.getElementById('slider');
async function show(){if(!e.value)return;let p=new URLSearchParams({episode:e.value,index:s.value});
for(let k of ['main','wrist'])document.getElementById(k).src='/frame?'+p+'&camera='+k;
let d=await(await fetch('/sample?'+p)).json();document.getElementById('index').textContent=s.value+'/'+s.max;
document.getElementById('values').textContent=JSON.stringify(d,null,2);}
function choose(){let d=items.find(x=>x.name===e.value);s.max=Math.max(0,d.frames-1);s.value=0;
document.getElementById('info').textContent=JSON.stringify(d);show();}
e.onchange=choose;s.oninput=show;document.getElementById('play').onclick=()=>{
if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{if(+s.value>=+s.max){clearInterval(timer);timer=null;return}s.value=+s.value+1;show()},1000/15)};
document.onkeydown=x=>{if(['ArrowLeft','ArrowRight'].includes(x.key)){s.value=Math.max(0,Math.min(+s.max,+s.value+(x.key==='ArrowLeft'?-1:1)));show()}};
fetch('/episodes').then(x=>x.json()).then(d=>{items=d;for(let x of d){let o=document.createElement('option');o.value=x.name;o.textContent=x.name+' · '+x.outcome+(x.mock?' · MOCK':'');e.append(o)}if(d.length)choose()});
</script>"""

def handler(root):
    root=Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def send(self,content,mime="application/json"):
            if not isinstance(content,bytes):content=content.encode()
            self.send_response(200);self.send_header("Content-Type",mime)
            self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(content)))
            self.end_headers();self.wfile.write(content)
        def do_GET(self):
            try:
                req=urlparse(self.path);query=parse_qs(req.query)
                if req.path=="/":return self.send(HTML,"text/html; charset=utf-8")
                if req.path=="/episodes":
                    entries=[]
                    for p in sorted(root.glob("*.h5")):
                        with h5py.File(p,"r") as f:
                            if not f.attrs.get("complete"):continue
                            entries.append(dict(name=p.name,frames=int(f.attrs["frames"]),
                                outcome=str(f.attrs["outcome"]),mock=bool(f.attrs["mock"]),
                                session=str(f.attrs["session"])))
                    return self.send(json.dumps(entries))
                name=query["episode"][0]
                if Path(name).name!=name or not name.endswith(".h5"):raise ValueError("invalid episode name")
                path=(root/name).resolve()
                if path.parent!=root:raise ValueError("invalid path")
                index=int(query["index"][0])
                with h5py.File(path,"r") as f:
                    if not 0<=index<int(f.attrs["frames"]):raise ValueError("frame outside episode")
                    if req.path=="/frame":
                        camera=query["camera"][0]
                        if camera not in ("main","wrist"):raise ValueError("invalid camera")
                        stream=io.BytesIO();Image.fromarray(f[f"images/{camera}"][index]).save(stream,format="JPEG")
                        return self.send(stream.getvalue(),"image/jpeg")
                    if req.path=="/sample":
                        keys=("q","q_target","gripper_width","gripper_command_width","action","critical","intervention")
                        return self.send(json.dumps({k:np.asarray(f[k][index]).tolist() for k in keys}))
                raise ValueError("unknown route")
            except (ValueError,KeyError,OSError) as exc:self.send_error(400,str(exc))
        def log_message(self,*args):pass
    return Handler
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",default="data/raw")
    p.add_argument("--port",type=int,default=8766);a=p.parse_args()
    print(f"Read-only review: http://127.0.0.1:{a.port}",flush=True)
    ThreadingHTTPServer(("127.0.0.1",a.port),handler(a.input)).serve_forever()
if __name__=="__main__":main()
