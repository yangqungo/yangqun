from __future__ import annotations
import ast
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import logging
import logging.handlers
import os
import re
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
import random
import requests

S = threading.Event()

def ih(l: logging.LoggerAdapter | None = None) -> None:
    def h(signum, frame):
        if not S.is_set():
            S.set()
            if l:
                l.warning("收到信号 %s，开始优雅退出…", signum)
    signal.signal(signal.SIGINT, h)
    try:
        signal.signal(signal.SIGTERM, h)
    except Exception:
        pass

FMT = "%(asctime)s %(levelname).1s [%(app)s/%(acc)s] %(message)s"
DFMT = "%Y-%m-%d %H:%M:%S"
LG: Dict[str, logging.LoggerAdapter] = {}
DD = Path(os.getenv("LOG_DIR") or ("/ql/log/xf" if Path("/ql/log").is_dir() else "./logs"))
DD = DD if DD.is_absolute() else Path.cwd() / DD
DD.mkdir(parents=True, exist_ok=True)

def sn(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

class F(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        r.app = getattr(r, "app", "xf")
        r.acc = getattr(r, "acc", "-")
        return super().format(r)

def tl(s: Optional[str]) -> int:
    try:
        return getattr(logging, (s or "INFO").upper())
    except Exception:
        return logging.INFO

def gl(app: str = "xf", acc: Optional[str] = None, *, log_dir: Optional[str | Path] = None, to_stdout: bool = True, per_name_file: bool = True, level: Optional[str] = None) -> logging.LoggerAdapter:
    lvl = tl(level or os.getenv("LOG_LEVEL"))
    bd = Path(log_dir) if log_dir else DD
    bd = bd if bd.is_absolute() else Path.cwd() / bd
    bd.mkdir(parents=True, exist_ok=True)
    k = f"{app}:{acc}:{bd}:{int(to_stdout)}:{int(per_name_file)}:{lvl}"
    if k in LG:
        return LG[k]
    lg = logging.getLogger(f"{app}.{acc or 'root'}")
    lg.setLevel(lvl)
    lg.propagate = False
    f = F(FMT, DFMT)
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(f)
        lg.addHandler(sh)
    if per_name_file:
        fp = (bd / (sn(acc or app) + ".log")).as_posix()
        fh = logging.handlers.TimedRotatingFileHandler(fp, when=os.getenv("LOG_ROTATE", "midnight"), backupCount=int(os.getenv("LOG_BACKUP_DAYS", "7")), encoding="utf-8")
        fh.setFormatter(f)
        lg.addHandler(fh)
        logging.LoggerAdapter(lg, {"app": app, "acc": acc or "-"}).info("File logging → %s", fp)
    ad = logging.LoggerAdapter(lg, {"app": app, "acc": acc or "-"})
    LG[k] = ad
    return ad

def b64d(d: str) -> str:
    if not isinstance(d, str):
        raise ValueError("Expected str for base64 decode")
    d = d.replace("-", "+").replace("_", "/")
    m = len(d) % 4
    if m:
        d += "=" * (4 - m)
    return base64.b64decode(d).decode("utf-8")

def rxf() -> list[tuple[str, dict]]:
    try:
        bd = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        bd = os.getcwd()
    rs = []
    for fn in os.listdir(bd):
        lf = fn.lower()
        if not (lf.startswith("xf_") and lf.endswith((".text", ".txt", ".json"))):
            continue
        p = os.path.join(bd, fn)
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                c = f.read().strip()
            if not c:
                raise ValueError("文件为空")
            try:
                d = json.loads(c)
            except json.JSONDecodeError:
                d = ast.literal_eval(c)
            if not isinstance(d, dict):
                raise ValueError("内容不是字典类型")
            rs.append((os.path.splitext(fn)[0], d))
        except Exception as e:
            print(f"❌ 文件解析失败: {fn} -> {e}")
    return rs

def etu() -> tuple[str, str]:
    try:
        dr = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        dr = os.getcwd()
    tk: Optional[str] = None
    u: Optional[str] = None
    if not os.path.isdir(dr):
        raise FileNotFoundError(f"目录不存在: {dr}")
    for fn in os.listdir(dr):
        low = fn.lower()
        if low.startswith("token_xf") and low.endswith((".text", ".txt", ".json")):
            if tk is None:
                with open(os.path.join(dr, fn), "r", encoding="utf-8") as f:
                    c = f.read().strip()
                if not c:
                    raise ValueError(f"文件 '{fn}' 为空")
                tk = c
        elif low.startswith("url_xf") and low.endswith((".text", ".txt", ".json")):
            if u is None:
                with open(os.path.join(dr, fn), "r", encoding="utf-8") as f:
                    c = f.read().strip()
                if not c:
                    raise ValueError(f"文件 '{fn}' 为空")
                u = c
    if tk is None:
        raise ValueError("未找到 token_xf 文件")
    if u is None:
        raise ValueError("未找到 url_xf 文件")
    return tk, u

def ehi(d: dict) -> dict:
    c = (d.get("accountInfoV2") or {}).get("coinAccount", {}).get("amount")
    ca = (d.get("accountInfoV2") or {}).get("cashAccount", {}).get("amountDisplay")
    tsec = {
        "daily": ((d.get("dailyTaskInfo") or {}).get("tasks") or []),
        "special": ((d.get("specialTaskInfo") or {}).get("tasks") or []),
        "watchTube": ((d.get("watchTubeTaskInfo") or {}).get("tasks") or []),
    }
    rt: Dict[str, list] = {}
    for snm, rts in tsec.items():
        so = []
        for t in (rts or []):
            ti = t.get("title")
            if not ti:
                tl = t.get("titles")
                if isinstance(tl, list) and tl and isinstance(tl[0], dict):
                    ti = tl[0].get("text")
            esid = (t.get("extParam") or {}).get("taskShowId")
            lus = []
            st = [(t, "")]
            while st:
                o, p = st.pop()
                if isinstance(o, dict):
                    for k, v in o.items():
                        np = f"{p}.{k}" if p else k
                        if k == "linkUrl":
                            li: Dict[str, Any] = {"path": np, "value": v}
                            if isinstance(v, str) and v.startswith("eyJ"):
                                try:
                                    db = base64.b64decode(v + "=" * (-len(v) % 4))
                                    ds = db.decode("utf-8")
                                    try:
                                        dj = json.loads(ds)
                                    except json.JSONDecodeError:
                                        dj = ds
                                    li["decoded"] = dj
                                except (binascii.Error, UnicodeDecodeError):
                                    pass
                            lus.append(li)
                        if isinstance(v, (dict, list)):
                            st.append((v, np))
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        np = f"{p}[{i}]"
                        if isinstance(v, (dict, list)):
                            st.append((v, np))
            so.append({"id": t.get("id"), "taskStatus": t.get("taskStatus"), "taskToken": t.get("taskToken"), "title": ti, "extParam_taskShowId": esid, "linkUrls": lus})
        rt[snm] = so
    return {"coin_amount": c, "cash_amount_display": ca, "tasks": rt}

def elp(d: dict) -> Optional[dict]:
    try:
        l = d["nextStage"]["popupInfo"]["buttonInfo"]["linkUrl"]
        de = json.loads(b64d(l))
        return {"posId": de["posId"], "box_task_id": de["businessId"], "taskToken": de["extParams"]}
    except Exception:
        return None

class HomeRes(TypedDict, total=False):
    coin: int
    taskId: str
    extParams: dict
    raw: dict

class InspireRes(TypedDict, total=False):
    inspireId: str
    creativeId: str
    subPageId: str
    raw: dict

class RewardResult(TypedDict, total=False):
    next_params: Optional[dict]
    toast: Optional[str]
    raw: dict

class ProcessRes(TypedDict, total=False):
    status: str
    raw: dict

@dataclass
class Ctx:
    a: dict
    n: int = 1
    tid: int = 6005
    tids: list[int] = field(default_factory=lambda: [6005, 6014])
    tidx: int = 0
    r: int = 0
    td: int = 0
    h: Optional[HomeRes] = None
    i: Optional[InspireRes] = None
    rw: Optional[RewardResult] = None
    p: Optional[ProcessRes] = None
    bx: dict = field(default_factory=lambda: {"box_task_id": 6017, "llsid": 1, "creativeId": "a", "idempotentId": "", "posId": "", "taskToken": "", "task_step": 1, "task_status": 1, "continuousTimes": 0})

@dataclass
class C:
    bu: str
    tk: str
    nm: str = ""
    ad: dict = field(default_factory=dict)
    s: requests.Session = field(default_factory=requests.Session, init=False)
    ds: requests.Session = field(default_factory=requests.Session, init=False)
    to: tuple[float, float] = (5.0, 20.0)
    mr: int = 3
    bb: float = 0.5
    bm: float = 8.0
    se: threading.Event = field(default_factory=threading.Event, init=False)
    px: Optional[Dict[str, str]] = field(default=None, init=False, repr=False)
    lg: logging.LoggerAdapter = field(init=False, repr=False)
    ipu: tuple[str, ...] = field(default=("https://api.ipify.org", "https://ifconfig.me/ip", "https://httpbin.org/ip"), init=False, repr=False)
    def __post_init__(self) -> None:
        self.s.trust_env = False
        self.ds.trust_env = False
        self.s.proxies.clear()
        self.px = self.ep()
        self.ap(self.px)
        self.lg = gl(app="xf", acc=self.nm or "-")
        if self.px:
            self.lg.info("启用代理=%s", self.px)
    def ar(self, path: str, payload: dict, headers: Optional[dict] = None) -> dict:
        if self.se.is_set():
            raise KeyboardInterrupt("stopped")
        u = f"{self.bu}{path}"
        h = {"X-Token": self.tk, "Content-Type": "application/json"}
        if headers:
            h.update(headers)
        r = self.s.post(u, json=payload, headers=h, timeout=self.to, proxies={"http": None, "https": None})
        r.raise_for_status()
        try:
            d = r.json()
        except ValueError as exc:
            raise TypeError("Gateway response is not valid JSON") from exc
        if isinstance(d, str):
            return json.loads(d)
        if isinstance(d, dict):
            return d
        raise TypeError(f"Unsupported type: {type(d)}")
    def dr(self, url: str, payload: Any, headers: Optional[dict] = None) -> dict:
        for a in range(1, self.mr + 1):
            if self.se.is_set():
                raise KeyboardInterrupt("stopped")
            try:
                resp = self.ds.post(url, data=payload, headers=headers, timeout=self.to)
                resp.raise_for_status()
                try:
                    d = resp.json()
                except ValueError as exc:
                    raise TypeError("Direct response is not valid JSON") from exc
                if isinstance(d, str):
                    return json.loads(d)
                if isinstance(d, dict):
                    return d
                raise TypeError(f"Unsupported response type: {type(d)}")
            except Exception:
                if a >= self.mr or self.wb(a):
                    raise
    request = dr
    def wb(self, a: int) -> bool:
        s = min(self.bm, self.bb * (2 ** (a - 1)) + random.uniform(0, 0.3))
        return self.se.wait(s)
    def ep(self) -> Optional[Dict[str, str]]:
        cfg = self.ad or {}
        if not isinstance(cfg, dict):
            return None
        p = cfg.get("proxies")
        if isinstance(p, dict) and ("http" in p or "https" in p):
            return {k: str(v) for k, v in p.items() if k in ("http", "https")}
        th = {k: cfg.get(k) for k in ("http", "https") if isinstance(cfg.get(k), (str,))}
        if th:
            return {k: str(v) for k, v in th.items()}
        stc = cfg.get("socks") or cfg.get("socks5") or cfg.get("socks_proxy") or cfg.get("proxy")
        if not isinstance(stc, dict):
            stc = cfg if any(k in cfg for k in ("host", "hostname", "server")) else {}
        host = stc.get("host") or stc.get("hostname") or stc.get("server")
        port = stc.get("port")
        scheme = (stc.get("scheme") or stc.get("type") or "socks5h").lower()
        user = stc.get("username") or stc.get("user")
        pwd = stc.get("password") or stc.get("pass") or stc.get("pwd")
        if host and port:
            au = None
            if user or pwd:
                au = f"{'' if user is None else str(user)}:{'' if pwd is None else str(pwd)}@"
            ap = f"{au}" if au else ""
            u = f"{scheme}://{ap}{host}:{port}"
            return {"http": u, "https": u}
        return None
    def ap(self, prx: Optional[Dict[str, str]]) -> None:
        self.ds.proxies.clear()
        if prx:
            self.ds.proxies.update(prx)
    def rp(self, ad: Optional[dict] = None) -> Optional[Dict[str, str]]:
        if ad is not None:
            self.ad = ad
        self.px = self.ep()
        self.ap(self.px)
        return self.px
    def ei(self, *, direct: bool) -> Optional[str]:
        if self.se.is_set():
            return None
        s = self.ds if direct else self.s
        prx = None if direct else {"http": None, "https": None}
        for u in self.ipu:
            if self.se.is_set():
                return None
            try:
                r = s.get(u, timeout=(2.0, 5.0), proxies=prx)
                r.raise_for_status()
                t = r.text.strip()
                if "httpbin.org" in u:
                    try:
                        t = (r.json().get("origin") or "").strip()
                    except Exception:
                        pass
                c = t.split(",")[0].strip()
                def _ip_like(x: str) -> bool:
                    if ":" in x and len(x) >= 3:
                        return True
                    ps = x.split(".")
                    if len(ps) == 4:
                        ok = True
                        for part in ps:
                            if not part.isdigit():
                                ok = False
                                break
                            iv = int(part)
                            if iv < 0 or iv > 255:
                                ok = False
                                break
                        if ok:
                            return True
                    return False
                if _ip_like(c):
                    return c
            except Exception:
                if self.se.wait(0.05):
                    return None
                continue
        return None

def bhp(ctx: Ctx) -> dict:
    return {"account": ctx.a}

def dm(msg: str, *, base_url: str, token: str, stop_event: threading.Event) -> str:
    p = {"operation": "decrypt", "data": msg}
    h = {"Content-Type": "application/json", "X-Token": token}
    u = base_url + "/process"
    for a in range(1, 4):
        if stop_event.is_set():
            raise KeyboardInterrupt("stopped")
        try:
            r = requests.post(u, headers=h, data=json.dumps(p), timeout=(5.0, 20.0))
            r.raise_for_status()
            return r.json()["result"]
        except Exception:
            if a >= 3 or stop_event.wait(0.3 * a):
                raise

def phr(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> HomeRes:
    if d.get("data"):
        de = dm(d["data"], base_url=base_url, token=token, stop_event=stop_event)
        return ehi(json.loads(de))
    return {}

def bip(ctx: Ctx) -> dict:
    return {"account": ctx.a, "task_dict": {}}

def ef(x: Any) -> dict[str, Any]:
    tfs = ["creativeId", "llsid"]
    rs: dict[str, Any] = {f: None for f in tfs}
    def w(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in tfs and rs[k] is None:
                    rs[k] = v
                elif isinstance(v, (dict, list)):
                    w(v)
        elif isinstance(o, list):
            for it in o:
                w(it)
    w(x)
    return rs

def pir(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> InspireRes:
    de = dm(d.get("data"), base_url=base_url, token=token, stop_event=stop_event) if d.get("data") else "{}"
    return ef(json.loads(de))

def btrp(ctx: Ctx) -> dict:
    assert ctx.i is not None, "inspire 结果缺失"
    ct = ecti(ctx)
    return {"account": ctx.a, "task_dict": ctx.i, "task_id": ct}

def ecti(ctx: Ctx) -> int:
    if not ctx.tids:
        raise ValueError("task_ids 为空，无法构造任务奖励请求")
    if ctx.tid not in ctx.tids:
        ctx.tid = ctx.tids[ctx.tidx % len(ctx.tids)]
    return ctx.tid

def fsis(d: dict, status: int = 13) -> list[str]:
    rs = []
    for st in d.get("popupInfo", {}).get("stages", []):
        if st.get("status") == status:
            rs.append(st.get("stageIndex"))
    return rs

def bbp(ctx: Ctx) -> dict:
    if ctx.bx["task_step"] == 3:
        return {"account": ctx.a, "task_dict": ctx.i, "box": ctx.bx}
    return {"account": ctx.a, "task_dict": {}, "box": ctx.bx}

def hbr(ctx: Ctx, d: dict, *, base_url: str, token: str, stop_event: threading.Event, logger: logging.LoggerAdapter | None = None) -> dict:
    raw = d.get("data")
    if not raw:
        return {}
    de = json.loads(dm(raw, base_url=base_url, token=token, stop_event=stop_event))
    if ctx.bx["task_step"] == 1:
        ctx.bx["box_task_id"] = de.get("id")
        ctx.bx["taskToken"] = de.get("taskToken")
        ids = fsis(de)
        if ids:
            ctx.bx["idempotentId"] = ids[0]
        else:
            ctx.bx["idempotentId"] = ""
            if logger:
                logger.warning("未找到可领取的宝箱")
            ctx.bx["task_status"] = 0
        ctx.bx["task_step"] = 2
    elif ctx.bx["task_step"] == 2:
        lp = elp(de)
        if lp:
            ctx.bx.update(lp)
        ctx.bx["task_step"] = 3
    elif ctx.bx["task_step"] == 3:
        def dg(dic, ks, default=None):
            for k in ks:
                if isinstance(dic, dict):
                    dic = dic.get(k)
                else:
                    return default
            return dic if dic is not None else default
        l = dg(de, ["popUp", "data", "buttonInfo", "linkUrl"])
        if l:
            try:
                dc = json.loads(b64d(l))
                ctx.bx.update({"posId": dc.get("posId", ""), "box_task_id": dc.get("businessId", ctx.bx.get("box_task_id")), "taskToken": dc.get("extParams", ctx.bx.get("taskToken"))})
                if ctx.bx.get("box_task_id") == 6031:
                    ctx.bx["continuousTimes"] += 1
            except Exception:
                ctx.bx["task_status"] = 0
        else:
            ctx.bx["task_status"] = 0
    return de

def prr(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> RewardResult:
    raw = d.get("data")
    if not raw:
        return {"raw": d}
    return json.loads(dm(raw, base_url=base_url, token=token, stop_event=stop_event))

def rf(c: C, ctx: Ctx, l: logging.LoggerAdapter) -> Ctx:
    if c.se.is_set():
        return ctx
    try:
        gh = c.ar("/andriod_home", bhp(ctx))
        b = b64d(gh["body"])
        hj = c.dr(url=gh["url"], payload=b, headers=gh["headers"])
        ctx.h = phr(hj, base_url=c.bu, token=c.tk, stop_event=c.se)
        l.info("主页 ok: coin=%s 现金=%s", ctx.h.get("coin_amount"), ctx.h.get("cash_amount_display"))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        l.error("HOME 流程失败: %s", e, exc_info=True)
        return ctx
    try:
        if c.se.is_set():
            return ctx
        bp = bbp(ctx)
        gb = c.ar("/andriod_xf_box", payload=bp)
        b = b64d(gb["body"])
        bj = c.dr(url=gb["url"], payload=b, headers=gb["headers"])
        ctx.rw = hbr(ctx, bj, base_url=c.bu, token=c.tk, stop_event=c.se, logger=l)
        l.info("宝箱状态: %s\n", ctx.rw)
        c.se.wait(1)
        if ctx.bx.get("task_status") == 1 and not c.se.is_set():
            bop = bbp(ctx)
            gbo = c.ar("/andriod_xf_box", payload=bop)
            b = b64d(gbo["body"])
            boj = c.dr(url=gbo["url"], payload=b, headers=gbo["headers"])
            ctx.rw = hbr(ctx, boj, base_url=c.bu, token=c.tk, stop_event=c.se, logger=l)
            l.info("宝箱奖励 %s \n", ctx.rw)
            for _ in range(71):
                if ctx.bx.get("task_status") != 1 or c.se.is_set():
                    break
                gi = c.ar("/xf_ad_inspire", bip(ctx))
                b = b64d(gi["body"])
                ij = c.dr(url=gi["url"], payload=b, headers=gi["headers"])
                ctx.i = pir(ij, base_url=c.bu, token=c.tk, stop_event=c.se)
                l.info("广告30秒…")
                if c.se.wait(30):
                    break
                gbr = c.ar("/andriod_xf_box", bbp(ctx))
                b = b64d(gbr["body"])
                brj = c.dr(url=gbr["url"], payload=b, headers=gbr["headers"])
                ctx.rw = hbr(ctx, brj, base_url=c.bu, token=c.tk, stop_event=c.se, logger=l)
                ctx.td += 1
                l.info("宝箱视频奖励 %s 累计=%d \n", ctx.rw, ctx.td)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        l.error("BOX 流程失败: %s", e, exc_info=True)
    for _ in range(99):
        if c.se.is_set():
            break
        try:
            gi = c.ar("/xf_ad_inspire", bip(ctx))
            b = b64d(gi["body"])
            ij = c.dr(url=gi["url"], payload=b, headers=gi["headers"])
            ctx.i = pir(ij, base_url=c.bu, token=c.tk, stop_event=c.se)
            ct = ecti(ctx)
            l.info("请求广告成功, 等待30秒\n")
            if c.se.wait(30):
                break
            gr = c.ar("/andriod_xf_reward", btrp(ctx))
            b = b64d(gr["body"])
            rj = c.dr(url=gr["url"], payload=b, headers=gr["headers"])
            ctx.rw = prr(rj, base_url=c.bu, token=c.tk, stop_event=c.se)
            av: Optional[str | float] = None
            sf = None
            if isinstance(ctx.rw, dict) and ctx.rw:
                ctx.td += 1
                l.info("%d 广告奖励结果: %s (累计=%d)\n", ct, ctx.rw, ctx.td)
                sf = ctx.rw.get("stageFinished")
                av = ctx.rw.get("amount")
            try:
                an = float(av) if av is not None else None
            except Exception:
                an = None
            if (an is not None and an < 60) or (sf is False):
                pt = ctx.tid
                ctx.tidx += 1
                ctx.r += 1
                if ctx.tidx < len(ctx.tids):
                    ctx.tid = ctx.tids[ctx.tidx]
                    l.info("奖励%s 低/未完成，切换任务ID: %s → %s (rotations=%d)", f" {an:.2f}" if an is not None else "", pt, ctx.tid, ctx.r)
                    continue
                l.info("所有任务ID已轮转完毕 (rotations=%d)，结束循环。", ctx.r)
                break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            l.error("激励广告循环异常: %s", e, exc_info=True)
            continue
    return ctx

def rfa(en: tuple[str, dict], *, base_url: str, token: str, stop_event: threading.Event) -> tuple[str, bool, Optional[str]]:
    n, ad = en
    l = gl(app="xf", acc=n)
    try:
        c = C(bu=base_url, tk=token, nm=n, ad=ad)
        c.se = stop_event
        ctx = Ctx(a=ad)
        rf(c, ctx, l)
        return n, True, None
    except KeyboardInterrupt:
        l.info("被中断：%s", n)
        return n, False, "interrupted"
    except Exception as e:
        l.error("failed: %s", e, exc_info=True)
        return n, False, str(e)

def main() -> None:
    sl = gl(app="xf", acc="SYSTEM")
    ih(sl)
    try:
        tk, bu = etu()
        u = f"{bu.rstrip('/')}/token_remaining"
        ps = {"token": tk, "collection": "tokens"}
        r = requests.get(u, params=ps, timeout=(5, 20))
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        sl.error("读取 token/url 失败: %s", e)
        return
    print("####获取token与服务器地址######")
    print(tk)
    print(bu)
    print("token剩余次数:", d.get("remaining"))
    print("##############################")
    print("####开始格式化账号######")
    print("####读取账号######")
    acs = rxf()
    if not acs:
        gl(app="xf", acc="i").error("no accounts found")
        return
    mw = min(len(acs), int(os.getenv("XF_WORKERS", "4")))
    sl.info("开始 %d 个账号, 使用 %d 个线程", len(acs), mw)
    ex = ThreadPoolExecutor(max_workers=mw, thread_name_prefix="xf")
    fs = {ex.submit(rfa, en, base_url=bu, token=tk, stop_event=S): en[0] for en in acs}
    rs: list[tuple[str, bool, Optional[str]]] = []
    try:
        while fs and not S.is_set():
            dn, pd = wait(fs.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
            for f in dn:
                try:
                    rs.append(f.result())
                except KeyboardInterrupt:
                    S.set()
                except Exception as e:
                    nm = fs[f]
                    rs.append((nm, False, str(e)))
                finally:
                    fs.pop(f, None)
    except KeyboardInterrupt:
        S.set()
        sl.warning("收到 Ctrl+C，准备停止所有任务…")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    ok = sum(1 for _, ok, _ in rs if ok)
    sl.info("本轮结束, 共计 %d/%d 个账号 (被中断的任务可能未计入)", ok, len(acs))

if __name__ == "__main__":
    main()
