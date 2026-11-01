import json,socket,time,uuid
from .contract import CONTRACT_ID
class Bridge:
    def __init__(self,path,*,mock):
        self.path=path;self.owner=uuid.uuid4().hex;self.seq=0;self.claimed=False
        st=self.state()
        if bool(st["mock"])!=mock:raise ValueError("mock/real bridge mismatch")
        if st["calibration"]["action_contract_id"]!=CONTRACT_ID:raise ValueError("bridge action contract mismatch")
        if st.get("timestamp_clock")!="host_monotonic_seconds" or any(
                key not in st for key in ("state_bridge_receive_s","gripper_bridge_read_s")):
            raise ValueError("bridge lacks explicit host timestamps; rebuild the demo bridge")
        if st.get("claimed"):raise ValueError("bridge already claimed")
        self.session=st["session"]
    def request(self,payload):
        data=json.dumps({"protocol":1,**payload},allow_nan=False).encode()+b"\n"
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
            s.settimeout(.2);s.connect(self.path);s.sendall(data);buf=b""
            while b"\n" not in buf:
                chunk=s.recv(16384)
                if not chunk:raise RuntimeError("bridge closed")
                buf+=chunk
                if len(buf)>131072:raise RuntimeError("oversized bridge reply")
        reply=json.loads(buf.split(b"\n",1)[0])
        if not reply.get("ok"):raise RuntimeError(reply.get("error","bridge rejected"))
        return reply
    def state(self):
        st=self.request({"op":"state"})
        if not st.get("ready") or st.get("fault") or st.get("stopped"):
            raise RuntimeError(f"bridge unavailable: {st.get('fault')}")
        if not 0<=st["state_age_s"]<=.05 or not 0<=st["gripper_age_s"]<=.5:
            raise RuntimeError("stale robot or gripper")
        return st
    def claim(self):
        self.request({"op":"claim","session":self.session,"controller_id":self.owner})
        self.claimed=True
    def command(self,st,target,width):
        self.seq+=1
        sent=time.monotonic()
        reply=self.request({"op":"command","session":self.session,"controller_id":self.owner,
            "sequence":self.seq,"state_sequence":st["state_sequence"],
            "q_target":target.tolist(),"gripper_width":float(width)})
        acknowledged=time.monotonic()
        if reply.get("sequence")!=self.seq:raise RuntimeError("bridge command ACK sequence mismatch")
        reply["command_send_s"]=sent;reply["command_ack_s"]=acknowledged
        return reply
    def close(self):
        if self.claimed:
            self.request({"op":"stop","session":self.session,"controller_id":self.owner})
            self.claimed=False
