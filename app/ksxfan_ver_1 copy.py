from __future__ import annotations
import ast, base64, binascii, json, logging, logging.handlers, os, re, signal, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

def _j(*s): return "".join(s)

S = threading.Event()

def i1(l: logging.LoggerAdapter | None = None) -> None:
    def h(sig, frm):
        if not S.is_set():
            S.set()
            if l: l.warning("收到信号 %s，开始优雅退出…", sig)
    signal.signal(signal.SIGINT, h)
    try: signal.signal(signal.SIGTERM, h)
    except Exception: pass

_f1 = "%(asctime)s %(levelname).1s [%(app)s/%(acc)s] %(message)s"
_d1 = "%Y-%m-%d %H:%M:%S"
_M: Dict[str, logging.LoggerAdapter] = {}
_D = Path(os.getenv("LOG_DIR") or ("/ql/log/xf" if Path("/ql/log").is_dir() else "./logs"))
_D = _D if _D.is_absolute() else Path.cwd() / _D
_D.mkdir(parents=True, exist_ok=True)

def n1(s: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

class Fm(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        r.app = getattr(r, "app", "xf")
        r.acc = getattr(r, "acc", "-")
        return super().format(r)

def lv(s: Optional[str]) -> int:
    try: return getattr(logging, (s or "INFO").upper())
    except Exception: return logging.INFO

def L(app: str = "xf", acc: Optional[str] = None, *, log_dir: Optional[str | Path] = None, to_stdout: bool = True, per_name_file: bool = True, level: Optional[str] = None) -> logging.LoggerAdapter:
    lvl = lv(level or os.getenv("LOG_LEVEL"))
    b = Path(log_dir) if log_dir else _D
    b = b if b.is_absolute() else Path.cwd() / b
    b.mkdir(parents=True, exist_ok=True)
    k = f"{app}:{acc}:{b}:{int(to_stdout)}:{int(per_name_file)}:{lvl}"
    if k in _M: return _M[k]
    lg = logging.getLogger(f"{app}.{acc or 'root'}"); lg.setLevel(lvl); lg.propagate = False
    f = Fm(_f1, _d1)
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); lg.addHandler(sh)
    if per_name_file:
        p = (b / (n1(acc or app) + ".log")).as_posix()
        fh = logging.handlers.TimedRotatingFileHandler(p, when=os.getenv("LOG_ROTATE", "midnight"), backupCount=int(os.getenv("LOG_BACKUP_DAYS", "7")), encoding="utf-8")
        fh.setFormatter(f); lg.addHandler(fh)
        logging.LoggerAdapter(lg, {"app": app, "acc": acc or "-"}).info("File logging → %s", p)
    ad = logging.LoggerAdapter(lg, {"app": app, "acc": acc or "-"})
    _M[k] = ad
    return ad

def d1(s: str) -> str:
    if not isinstance(s, str): raise ValueError("Expected str for base64 decode")
    s = s.replace("-", "+").replace("_", "/")
    m = len(s) % 4
    if m: s += "=" * (4 - m)
    return base64.b64decode(s).decode("utf-8")

def R() -> list[tuple[str, dict]]:
    try: b = os.path.dirname(os.path.abspath(__file__))
    except NameError: b = os.getcwd()
    r = []
    for fn in os.listdir(b):
        lo = fn.lower()
        if not (lo.startswith("xf_") and lo.endswith((".text", ".txt", ".json"))): continue
        fp = os.path.join(b, fn)
        try:
            with open(fp, "r", encoding="utf-8-sig") as f: ct = f.read().strip()
            if not ct: raise ValueError("文件为空")
            try: d = json.loads(ct)
            except json.JSONDecodeError: d = ast.literal_eval(ct)
            if not isinstance(d, dict): raise ValueError("内容不是字典类型")
            r.append((os.path.splitext(fn)[0], d))
        except Exception as e:
            print(f"❌ 文件解析失败: {fn} -> {e}")
    return r

def E() -> tuple[str, str]:
    try: dr = os.path.dirname(os.path.abspath(__file__))
    except NameError: dr = os.getcwd()
    t: Optional[str] = None; u: Optional[str] = None
    if not os.path.isdir(dr): raise FileNotFoundError(f"目录不存在: {dr}")
    for fn in os.listdir(dr):
        lo = fn.lower()
        if lo.startswith("token_xf") and lo.endswith((".text", ".txt", ".json")) and t is None:
            with open(os.path.join(dr, fn), "r", encoding="utf-8") as f:
                c = f.read().strip()
            if not c: raise ValueError(f"文件 '{fn}' 为空")
            t = c
        elif lo.startswith("url_xf") and lo.endswith((".text", ".txt", ".json")) and u is None:
            with open(os.path.join(dr, fn), "r", encoding="utf-8") as f:
                c = f.read().strip()
            if not c: raise ValueError(f"文件 '{fn}' 为空")
            u = c
    if t is None: raise ValueError("未找到 token_xf 文件")
    if u is None: raise ValueError("未找到 url_xf 文件")
    return t, u

def H(d: dict) -> dict:
    c = (d.get("accountInfoV2") or {}).get("coinAccount", {}).get("amount")
    ca = (d.get("accountInfoV2") or {}).get("cashAccount", {}).get("amountDisplay")
    ts = {"daily": ((d.get("dailyTaskInfo") or {}).get("tasks") or []), "special": ((d.get("specialTaskInfo") or {}).get("tasks") or []), "watchTube": ((d.get("watchTubeTaskInfo") or {}).get("tasks") or [])}
    o: Dict[str, list] = {}
    for sn, raw in ts.items():
        so = []
        for t in (raw or []):
            tl = t.get("title")
            if not tl:
                ls = t.get("titles")
                if isinstance(ls, list) and ls and isinstance(ls[0], dict): tl = ls[0].get("text")
            sid = (t.get("extParam") or {}).get("taskShowId")
            lus = []; st = [(t, "")]
            while st:
                obj, p = st.pop()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        np = f"{p}.{k}" if p else k
                        if k == "linkUrl":
                            info: Dict[str, Any] = {"path": np, "value": v}
                            if isinstance(v, str) and v.startswith("eyJ"):
                                try:
                                    db = base64.b64decode(v + "=" * (-len(v) % 4))
                                    ds = db.decode("utf-8")
                                    try: dj = json.loads(ds)
                                    except json.JSONDecodeError: dj = ds
                                    info["decoded"] = dj
                                except (binascii.Error, UnicodeDecodeError): pass
                            lus.append(info)
                        if isinstance(v, (dict, list)): st.append((v, np))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        np = f"{p}[{i}]"
                        if isinstance(v, (dict, list)): st.append((v, np))
            so.append({"id": t.get("id"), "taskStatus": t.get("taskStatus"), "taskToken": t.get("taskToken"), "title": tl, "extParam_taskShowId": sid, "linkUrls": lus})
        o[sn] = so
    return {"coin_amount": c, "cash_amount_display": ca, "tasks": o}

def P(d: dict) -> Optional[dict]:
    try:
        lk = d["nextStage"]["popupInfo"]["buttonInfo"]["linkUrl"]
        dc = json.loads(d1(lk))
        return {"posId": dc["posId"], "box_task_id": dc["businessId"], "taskToken": dc["extParams"]}
    except Exception:
        return None

class HRes(TypedDict, total=False):
    coin: int; taskId: str; extParams: dict; raw: dict

class IRes(TypedDict, total=False):
    inspireId: str; creativeId: str; subPageId: str; raw: dict

class RRes(TypedDict, total=False):
    next_params: Optional[dict]; toast: Optional[str]; raw: dict

class PRes(TypedDict, total=False):
    status: str; raw: dict

@dataclass
class C:
    account: dict
    nums: int = 1
    task_id: int = 6005
    task_ids: list[int] = field(default_factory=lambda: [6005, 6018, 6013, 6014, 6015, 6017, 6020, 6027, 6037, 6038, 6040, 6041])
    task_id_idx: int = 0
    rotations: int = 0
    tasks_done: int = 0
    home: Optional[HRes] = None
    inspire: Optional[IRes] = None
    reward: Optional[RRes] = None
    process: Optional[PRes] = None
    box: dict = field(default_factory=lambda: {"box_task_id": 6017, "llsid": 1, "creativeId": "a", "idempotentId": "", "posId": "", "taskToken": "", "task_step": 1, "task_status": 1})

@dataclass
class A:
    base_url: str
    token: str
    name: str
    account_data: dict
    stop_event: threading.Event
    session: requests.Session = field(default_factory=requests.Session)
    timeout: tuple[float, float] = (5.0, 20.0)
    max_retries: int = 3

    def w(self, a: int) -> bool:
        return self.stop_event.wait(0.5 * a)

    # 兼容 payload/headers 关键字
    def g(self, pth: str, pl: dict = None, hd: Optional[dict] = None, **kw) -> dict:
        if pl is None:
            pl = kw.get("payload")
        if hd is None:
            hd = kw.get("headers")
        u = f"{self.base_url}{pth}"
        h = {"X-Token": self.token, "Content-Type": "application/json"}
        if hd:
            h.update(hd)
        for at in range(1, self.max_retries + 1):
            if self.stop_event.is_set():
                raise KeyboardInterrupt("stopped")
            try:
                r = self.session.post(u, json=pl, headers=h, timeout=self.timeout)
                r.raise_for_status()
                try:
                    d = r.json()
                except ValueError:
                    raise TypeError("Gateway response is not valid JSON")
                if isinstance(d, str):
                    return json.loads(d)
                if isinstance(d, dict):
                    return d
                raise TypeError(f"Unsupported response type: {type(d)}")
            except Exception:
                if at >= self.max_retries or self.w(at):
                    raise

    # 兼容 url/payload/headers 关键字
    def d(self, u: str = None, pl: Any = None, hd: Optional[dict] = None, **kw) -> dict:
        if u is None:
            u = kw.get("url")
        if pl is None:
            # 兼容 direct_request 的 data 语义
            pl = kw.get("payload") if "payload" in kw else kw.get("data")
        if hd is None:
            hd = kw.get("headers")
        if u is None:
            raise TypeError("Direct request needs a url")
        for at in range(1, self.max_retries + 1):
            if self.stop_event.is_set():
                raise KeyboardInterrupt("stopped")
            try:
                r = self.session.post(u, data=pl, headers=hd, timeout=self.timeout)
                r.raise_for_status()
                try:
                    d = r.json()
                except ValueError:
                    raise TypeError("Direct response is not valid JSON")
                if isinstance(d, str):
                    return json.loads(d)
                if isinstance(d, dict):
                    return d
                raise TypeError(f"Unsupported response type: {type(d)}")
            except Exception:
                if at >= self.max_retries or self.w(at):
                    raise

def B0(ctx: C) -> dict: return {"account": ctx.account}

def D(msg: str, *, base_url: str, token: str, stop_event: threading.Event) -> str:
    pl = {"operation": "decrypt", "data": msg}
    hd = {"Content-Type": "application/json", _j("X","-","Token"): token}
    u = base_url + _j("/", "proc", "ess")
    for at in range(1, 4):
        if stop_event.is_set(): raise KeyboardInterrupt("stopped")
        try:
            r = requests.post(u, headers=hd, data=json.dumps(pl), timeout=(5.0, 20.0)); r.raise_for_status()
            return r.json()["result"]
        except Exception:
            if at >= 3 or stop_event.wait(0.3 * at): raise

def PH(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> HRes:
    if d.get("data"):
        dc = D(d["data"], base_url=base_url, token=token, stop_event=stop_event)
        return H(json.loads(dc))
    return {}

def B1(ctx: C) -> dict: return {"account": ctx.account, "task_dict": {}}

def X(d: Any) -> dict[str, Any]:
    t = ["creativeId", "llsid"]; r: dict[str, Any] = {f: None for f in t}
    def w(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in t and r[k] is None: r[k] = v
                elif isinstance(v, (dict, list)): w(v)
        elif isinstance(o, list):
            for it in o: w(it)
    w(d); return r

def PI(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> IRes:
    dc = D(d.get("data"), base_url=base_url, token=token, stop_event=stop_event) if d.get("data") else "{}"
    return X(json.loads(dc))

def U(ctx: C) -> int:
    if not ctx.task_ids: raise ValueError("task_ids 为空，无法构造任务奖励请求")
    if ctx.task_id not in ctx.task_ids: ctx.task_id = ctx.task_ids[ctx.task_id_idx % len(ctx.task_ids)]
    return ctx.task_id

def B2(ctx: C) -> dict:
    assert ctx.inspire is not None, "inspire 结果缺失"
    cid = U(ctx)
    return {"account": ctx.account, "task_dict": ctx.inspire, "task_id": cid}

def F(d: dict, st: int = 13) -> list[str]:
    r = []
    for g in d.get("popupInfo", {}).get("stages", []):
        if g.get("status") == st: r.append(g.get("stageIndex"))
    return r

def B3(ctx: C) -> dict:
    if ctx.box["task_step"] == 3: return {"account": ctx.account, "task_dict": ctx.inspire, "box": ctx.box}
    return {"account": ctx.account, "task_dict": {}, "box": ctx.box}

def HB(ctx: C, d: dict, *, base_url: str, token: str, stop_event: threading.Event, logger: logging.LoggerAdapter | None = None) -> dict:
    rw = d.get("data")
    if not rw: return {}
    dc = json.loads(D(rw, base_url=base_url, token=token, stop_event=stop_event))
    if ctx.box["task_step"] == 1:
        ctx.box["box_task_id"] = dc.get("id")
        ctx.box["taskToken"] = dc.get("taskToken")
        ids = F(dc)
        if ids: ctx.box["idempotentId"] = ids[0]
        else:
            ctx.box["idempotentId"] = ""
            if logger: logger.warning("未找到可领取的宝箱")
            ctx.box["task_status"] = 0
        ctx.box["task_step"] = 2
    elif ctx.box["task_step"] == 2:
        lp = P(dc)
        if lp: ctx.box.update(lp)
        ctx.box["task_step"] = 3
    elif ctx.box["task_step"] == 3:
        def g(dic, ks, dft=None):
            for k in ks:
                if isinstance(dic, dict): dic = dic.get(k)
                else: return dft
            return dic if dic is not None else dft
        lk = g(dc, ["popUp", "data", "buttonInfo", "linkUrl"])
        if lk:
            try:
                dd = json.loads(d1(lk))
                ctx.box.update({"posId": dd.get("posId", ""), "box_task_id": dd.get("businessId", ctx.box.get("box_task_id")), "taskToken": dd.get("extParams", ctx.box.get("taskToken"))})
            except Exception:
                ctx.box["task_status"] = 0
        else:
            ctx.box["task_status"] = 0
    return dc

def PR(d: dict, *, base_url: str, token: str, stop_event: threading.Event) -> RRes:
    rw = d.get("data")
    if not rw: return {"raw": d}
    return json.loads(D(rw, base_url=base_url, token=token, stop_event=stop_event))

def RUN(cl: A, ctx: C, lg: logging.LoggerAdapter) -> C:
    if cl.stop_event.is_set(): return ctx
    st = 0
    while True:
        if st == 0:
            try:
                gw = cl.g(_j("/", "h","o","m","e"), B0(ctx))
                bd = d1(gw["body"])
                hj = cl.d(url=gw["url"], payload=bd, headers=gw["headers"])
                ctx.home = PH(hj, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event)
                lg.info("主页 ok: coin=%s 现金=%s", ctx.home.get("coin_amount"), ctx.home.get("cash_amount_display"))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                lg.error("HOME 流程失败: %s", e, exc_info=True)
                return ctx
            st = 1; continue
        if st == 1:
            try:
                if cl.stop_event.is_set(): return ctx
                pl = B3(ctx)
                gb = cl.g(_j("/", "apple","_","xf","_","box"), payload=pl)
                bd = d1(gb["body"])
                bj = cl.d(url=gb["url"], payload=bd, headers=gb["headers"])
                ctx.reward = HB(ctx, bj, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event, logger=lg)
                if ctx.box.get("task_status") == 1 and not cl.stop_event.is_set():
                    pl2 = B3(ctx)
                    gbo = cl.g(_j("/", "apple","_","xf","_","box"), payload=pl2)
                    bd = d1(gbo["body"])
                    bo = cl.d(url=gbo["url"], payload=bd, headers=gbo["headers"])
                    ctx.reward = HB(ctx, bo, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event, logger=lg)
                    for _ in range(5):
                        if ctx.box.get("task_status") != 1 or cl.stop_event.is_set(): break
                        gi = cl.g(_j("/", "apple","_","xf","_","ad","_","inspire"), B1(ctx))
                        bd = d1(gi["body"])
                        ij = cl.d(url=gi["url"], payload=bd, headers=gi["headers"])
                        ctx.inspire = PI(ij, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event)
                        lg.info("广告25秒…")
                        if cl.stop_event.wait(25): break
                        gr = cl.g(_j("/", "apple","_","xf","_","box"), B3(ctx))
                        bd = d1(gr["body"])
                        br = cl.d(url=gr["url"], payload=bd, headers=gr["headers"])
                        ctx.reward = HB(ctx, br, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event, logger=lg)
                        ctx.tasks_done += 1
                        lg.info("宝箱奖励 %s 累计=%d \n", ctx.reward, ctx.tasks_done)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                lg.error("BOX 流程失败: %s", e, exc_info=True)
            st = 2; continue
        if st == 2:
            for _ in range(6):
                if cl.stop_event.is_set(): break
                try:
                    gi = cl.g(_j("/", "apple","_","xf","_","ad","_","inspire"), B1(ctx))
                    bd = d1(gi["body"])
                    ij = cl.d(url=gi["url"], payload=bd, headers=gi["headers"])
                    ctx.inspire = PI(ij, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event)
                    cid = U(ctx)
                    lg.info("请求广告成功, 等待30秒\n")
                    if cl.stop_event.wait(30): break
                    gw = cl.g(_j("/", "apple","_","xf","_","reward"), B2(ctx))
                    bd = d1(gw["body"])
                    rj = cl.d(url=gw["url"], payload=bd, headers=gw["headers"])
                    ctx.reward = PR(rj, base_url=cl.base_url, token=cl.token, stop_event=cl.stop_event)
                    av: Optional[str | float] = None; sf = None
                    if isinstance(ctx.reward, dict) and ctx.reward:
                        ctx.tasks_done += 1
                        lg.info("%d 奖励结果: %s (累计=%d)\n", cid, ctx.reward, ctx.tasks_done)
                        sf = ctx.reward.get("stageFinished"); av = ctx.reward.get("amount")
                    try: an = float(av) if av is not None else None
                    except Exception: an = None
                    if (an is not None and an < 60) or (sf is False):
                        pv = ctx.task_id; ctx.task_id_idx += 1; ctx.rotations += 1
                        if ctx.task_id_idx < len(ctx.task_ids):
                            ctx.task_id = ctx.task_ids[ctx.task_id_idx]
                            lg.info("奖励%s 低/未完成，切换任务ID: %s → %s (rotations=%d)", f" {an:.2f}" if an is not None else "", pv, ctx.task_id, ctx.rotations)
                            continue
                        lg.info("所有任务ID已轮转完毕 (rotations=%d)，结束循环。", ctx.rotations)
                        break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    lg.error("激励广告循环异常: %s", e, exc_info=True)
                    continue
            st = 3; continue
        break
    return ctx

def RFA(ent: tuple[str, dict], *, base_url: str, token: str, stop_event: threading.Event) -> tuple[str, bool, Optional[str]]:
    nm, ad = ent
    lg = L(app="xf", acc=nm)
    try:
        cl = A(base_url=base_url, token=token, name=nm, account_data=ad, stop_event=stop_event)
        ctx = C(account=ad)
        RUN(cl, ctx, lg)
        return nm, True, None
    except KeyboardInterrupt:
        lg.info("被中断：%s", nm)
        return nm, False, "interrupted"
    except Exception as e:
        lg.error("failed: %s", e, exc_info=True)
        return nm, False, str(e)

def main() -> None:
    sl = L(app="xf", acc="SYSTEM")
    i1(sl)
    try:
        tk, bu = E()
    except Exception as e:
        sl.error("读取 token/url 失败: %s", e); return
    print("####获取token与服务器地址######")
    print(tk); print(bu)
    print("##############################")
    print("####开始格式化账号######")
    print("####读取账号######")
    ac = R()
    if not ac:
        L(app="xf", acc="i").error("no accounts found"); return
    mw = min(len(ac), int(os.getenv("XF_WORKERS", "4")))
    sl.info("开始 %d 个账号, 使用 %d 个线程", len(ac), mw)
    ex = ThreadPoolExecutor(max_workers=mw, thread_name_prefix="xf")
    fut = {ex.submit(RFA, e, base_url=bu, token=tk, stop_event=S): e[0] for e in ac}
    rs: list[tuple[str, bool, Optional[str]]] = []
    try:
        while fut and not S.is_set():
            dn, pd = wait(fut.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
            for ft in dn:
                try: rs.append(ft.result())
                except KeyboardInterrupt: S.set()
                except Exception as e: rs.append((fut[ft], False, str(e)))
                finally: fut.pop(ft, None)
    except KeyboardInterrupt:
        S.set(); sl.warning("收到 Ctrl+C，准备停止所有任务…")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    ok = sum(1 for _, k, _ in rs if k)
    sl.info("本轮结束, 共计 %d/%d 个账号 (被中断的任务可能未计入)", ok, len(ac))

if __name__ == "__main__":
    main()
