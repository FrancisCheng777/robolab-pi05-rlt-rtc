"""Bounded, asynchronous storage of human observation/action pairs."""
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from .rpc import encode_observation,json_value

class FrameWriter:
    def __init__(self,path):
        self.path=Path(path);self.queue=queue.Queue(64);self.error=None;self.count=0
        self.digest=hashlib.sha256();self.closed=False
        self.stream=self.path.open('x')
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def check(self):
        if self.error:raise RuntimeError('correction recording failed: '+self.error)
    def submit(self,obs,audit,leader_q):
        self.check()
        if self.closed:raise RuntimeError('writer already closed')
        self.queue.put_nowait((obs,audit,leader_q))
    def _run(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:break
                obs,audit,leader=item
                row=dict(format='gello-correction-frame-v1',source='human_gello',
                         observation=encode_observation(obs),audit=audit,leader_q=leader)
                line=json.dumps(row,default=json_value,allow_nan=False,separators=(',',':'))+'\n'
                self.stream.write(line);self.digest.update(line.encode());self.count+=1
            except BaseException as exc:self.error=f'{type(exc).__name__}: {exc}'
            finally:self.queue.task_done()
        try:self.stream.flush();os.fsync(self.stream.fileno())
        except BaseException as exc:self.error=f'{type(exc).__name__}: {exc}'
        finally:self.stream.close()
    def close(self):
        if not self.closed:
            self.closed=True;self.queue.put(None);self.thread.join(timeout=20)
            if self.thread.is_alive():raise RuntimeError('correction storage did not drain')
        self.check()

class GelloInput:
    def __init__(self,cfg):
        import sys
        support=Path(__file__).resolve().parents[2]/'runtime_support'
        if support.is_dir():sys.path.insert(0,str(support))
        from pi05_rlt_rtc.adapters.robolab import activate
        activate()
        from pi05_demo.collect import Gello
        from pi05_demo.teleop_reference import j7_offset_from_config
        from pi05_demo.gripper_teleop import unowned
        config=json.loads(cfg.asset(cfg.raw['session']['collection_config'],'collection config').read_text())
        unowned(config['gello_port'])
        self.reader=Gello(config['gello_port'],j7_offset_rad=j7_offset_from_config(config),
                          robolab=True,max_sample_age_s=.05)
        try:
            deadline=time.monotonic()+3
            while True:
                try:self.read();break
                except RuntimeError:
                    if time.monotonic()>=deadline:raise
                    time.sleep(.02)
        except BaseException:
            self.close();raise
    def read(self):
        _,width=self.reader.read()
        return self.reader.last_unclipped_target.copy(),width
    def close(self):self.reader.agent._robot._driver.close()
