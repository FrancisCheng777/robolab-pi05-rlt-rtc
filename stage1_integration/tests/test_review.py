import json,tempfile,threading,time,unittest,urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from pi05_demo.episode import EpisodeWriter
from pi05_demo.review import handler
class ReviewTests(unittest.TestCase):
    def test_inventory_frame_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            w=EpisodeWriter(d,mock=True,task="insert",session="s",config={});t=time.monotonic()
            w.append(q=np.zeros(7),target=np.zeros(7),width=.04,command_width=.04,
                frames={k:(np.zeros((16,16,3),np.uint8),t) for k in ("main","wrist")},
                timestamp=t,state_sequence=1,command_sequence=1,ack=True)
            p=w.finish("success");server=ThreadingHTTPServer(("127.0.0.1",0),handler(d))
            th=threading.Thread(target=server.serve_forever);th.start()
            base=f"http://127.0.0.1:{server.server_port}"
            try:
                rows=json.load(urllib.request.urlopen(base+"/episodes"));self.assertEqual(len(rows),1)
                response=urllib.request.urlopen(base+f"/frame?episode={p.name}&index=0&camera=main")
                self.assertEqual(response.headers["Content-Type"],"image/jpeg")
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+"/sample?episode=../x.h5&index=0")
            finally:server.shutdown();th.join();server.server_close()
