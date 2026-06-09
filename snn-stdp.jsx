import { useState, useEffect, useRef, useCallback } from "react";

// ── LIF constants ─────────────────────────────────────────────────
const TAU_M    = 20;    // ms
const V_REST   = -70;   // mV
const V_THRESH = -55;   // mV
const V_RESET  = -75;   // mV
const R_M      = 1;
const DT       = 0.5;   // ms
const REFRAC_T = 3;     // ms

// ── STDP constants ────────────────────────────────────────────────
const TAU_LTP  = 20;    // ms - LTP trace decay
const TAU_LTD  = 20;    // ms - LTD trace decay
const W_MIN    = 0;
const W_MAX    = 15;
const FLASH_DUR = 120;  // ms edge flash duration

// ── Network topology ──────────────────────────────────────────────
const NODES = [
  { id:0, x:0.13, y:0.22, label:"IN·A", type:"input"  },
  { id:1, x:0.13, y:0.50, label:"IN·B", type:"input"  },
  { id:2, x:0.13, y:0.78, label:"IN·C", type:"input"  },
  { id:3, x:0.38, y:0.18, label:"H·1",  type:"hidden" },
  { id:4, x:0.38, y:0.50, label:"H·2",  type:"hidden" },
  { id:5, x:0.38, y:0.82, label:"H·3",  type:"hidden" },
  { id:6, x:0.63, y:0.30, label:"H·4",  type:"hidden" },
  { id:7, x:0.63, y:0.70, label:"H·5",  type:"hidden" },
  { id:8, x:0.88, y:0.50, label:"OUT",  type:"output" },
];

const EDGE_DEFS = [
  {f:0,t:3,w0:9 },{f:0,t:4,w0:5 },
  {f:1,t:3,w0:4 },{f:1,t:4,w0:10},{f:1,t:5,w0:6 },
  {f:2,t:4,w0:7 },{f:2,t:5,w0:9 },
  {f:3,t:6,w0:8 },{f:3,t:7,w0:4 },
  {f:4,t:6,w0:6 },{f:4,t:7,w0:8 },
  {f:5,t:6,w0:3 },{f:5,t:7,w0:7 },
  {f:6,t:8,w0:9 },{f:7,t:8,w0:8 },
];

function freshState() {
  return {
    neurons: NODES.map(n => ({
      ...n,
      v:      V_REST,
      refrac: 0,
      spiked: false,
      spikes: [],
      vBuf:   new Array(300).fill(V_REST),
      xPre:   0,    // pre-synaptic STDP trace
      xPost:  0,    // post-synaptic STDP trace
    })),
    weights:   new Float32Array(EDGE_DEFS.map(e => e.w0)),  // mutable weights
    edgeFlash: new Array(EDGE_DEFS.length).fill(null),      // {type:'ltp'|'ltd', age:0}
    pulses:    [],
    time:      0,
    ltpCount:  0,
    ltdCount:  0,
  };
}

export default function SNNStdp() {
  const netRef    = useRef(null);
  const rasterRef = useRef(null);
  const traceRef  = useRef(null);
  const wtRef     = useRef(null);
  const simState  = useRef(freshState());
  const animRef   = useRef(null);

  const [running,    setRunning]    = useState(false);
  const [selNeuron,  setSelNeuron]  = useState(8);
  const [learning,   setLearning]   = useState(true);
  const [eventLog,   setEventLog]   = useState([]);
  const [params, setParams] = useState({
    inputRate:    25,
    inputCurrent: 14,
    noise:        2,
    aPlus:        0.008,   // LTP rate
    aMinus:       0.009,   // LTD rate
  });
  const paramsRef    = useRef(params);
  const learningRef  = useRef(learning);
  const selNeuronRef = useRef(selNeuron);
  useEffect(() => { paramsRef.current   = params;    }, [params]);
  useEffect(() => { learningRef.current = learning;  }, [learning]);
  useEffect(() => { selNeuronRef.current = selNeuron;}, [selNeuron]);

  // ── Simulation step ───────────────────────────────────────────
  const step = useCallback(() => {
    const s = simState.current;
    const { neurons, weights, edgeFlash, pulses } = s;
    const p = paramsRef.current;
    const learn = learningRef.current;
    const I_syn = new Float32Array(neurons.length);

    // Decay STDP traces
    const decayPre  = Math.exp(-DT / TAU_LTP);
    const decayPost = Math.exp(-DT / TAU_LTD);
    for (let i = 0; i < neurons.length; i++) {
      neurons[i].xPre  *= decayPre;
      neurons[i].xPost *= decayPost;
    }

    // Advance edge flashes
    for (let e = 0; e < edgeFlash.length; e++) {
      if (edgeFlash[e]) {
        edgeFlash[e].age += DT;
        if (edgeFlash[e].age > FLASH_DUR) edgeFlash[e] = null;
      }
    }

    // Advance synaptic pulses
    const alive = [];
    for (const pulse of pulses) {
      pulse.t += DT;
      if (pulse.t >= pulse.dur) {
        I_syn[pulse.to] += weights[pulse.edgeIdx] * 4;
      } else {
        alive.push(pulse);
      }
    }
    s.pulses = alive;

    // Poisson input to input neurons
    const lambda = (p.inputRate / 1000) * DT;
    for (let i = 0; i < 3; i++) {
      if (Math.random() < lambda) I_syn[i] += p.inputCurrent;
    }

    // LIF update + STDP
    const now = s.time;
    const newEvents = [];

    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      n.spiked = false;

      if (n.refrac > 0) {
        n.refrac -= DT;
        n.v = V_RESET;
      } else {
        const noise = (Math.random() - 0.5) * p.noise;
        n.v += (DT / TAU_M) * (-(n.v - V_REST) + R_M * (I_syn[i] + noise));
        n.v = Math.max(V_RESET - 5, Math.min(n.v, V_THRESH + 2));

        if (n.v >= V_THRESH) {
          n.spiked = true;
          n.v      = V_RESET;
          n.refrac = REFRAC_T;
          n.spikes.push(now);
          if (n.spikes.length > 150) n.spikes.shift();

          // Emit pulses downstream
          for (let e = 0; e < EDGE_DEFS.length; e++) {
            if (EDGE_DEFS[e].f === i) {
              s.pulses.push({ from: i, to: EDGE_DEFS[e].t, edgeIdx: e, t: 0, dur: 6 });
            }
          }

          if (learn) {
            // PRE fires → LTD: depress edges where this neuron is POST
            // (post fires — look at pre trace xPre of upstream neurons)
            // POST fires → LTP: potentiate edges where this neuron is POST
            for (let e = 0; e < EDGE_DEFS.length; e++) {
              const { f: pre, t: post } = EDGE_DEFS[e];

              if (post === i) {
                // i just fired (post). Apply LTP: Δw = A+ * xPre[pre]
                const dw = p.aPlus * neurons[pre].xPre;
                if (dw > 0.0001) {
                  weights[e] = Math.min(W_MAX, weights[e] + dw);
                  edgeFlash[e] = { type: 'ltp', age: 0 };
                  s.ltpCount++;
                  if (Math.random() < 0.05) newEvents.push({ t: now.toFixed(0), e, type:'LTP', dw: dw.toFixed(4), w: weights[e].toFixed(2) });
                }
              }

              if (pre === i) {
                // i just fired (pre). Apply LTD: Δw = -A- * xPost[post]
                const dw = p.aMinus * neurons[post].xPost;
                if (dw > 0.0001) {
                  weights[e] = Math.max(W_MIN, weights[e] - dw);
                  edgeFlash[e] = { type: 'ltd', age: 0 };
                  s.ltdCount++;
                  if (Math.random() < 0.05) newEvents.push({ t: now.toFixed(0), e, type:'LTD', dw: (-dw).toFixed(4), w: weights[e].toFixed(2) });
                }
              }
            }
          }

          // Update STDP traces for this neuron
          n.xPre  += 1;
          n.xPost += 1;
        }
      }

      n.vBuf.push(n.v);
      if (n.vBuf.length > 300) n.vBuf.shift();
    }

    s.time += DT;

    if (newEvents.length > 0) {
      setEventLog(prev => [...newEvents, ...prev].slice(0, 20));
    }
  }, []);

  // ── Draw network ──────────────────────────────────────────────
  const drawNet = useCallback((ctx, W, H) => {
    const { neurons, weights, edgeFlash, pulses } = simState.current;

    ctx.fillStyle = "#050810";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "rgba(0,80,120,0.07)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

    // Edges
    for (let e = 0; e < EDGE_DEFS.length; e++) {
      const { f, t } = EDGE_DEFS[e];
      const A = NODES[f], B = NODES[t];
      const ax = A.x*W, ay = A.y*H, bx = B.x*W, by = B.y*H;
      const w = weights[e];
      const norm = w / W_MAX;
      const flash = edgeFlash[e];

      let r = 0, g = 120, b = 200;
      let alpha = 0.1 + norm * 0.35;

      if (flash) {
        const fadeAlpha = Math.max(0, 1 - flash.age / FLASH_DUR);
        if (flash.type === 'ltp') { r=0;   g=255; b=120; alpha = 0.2 + fadeAlpha*0.7; }
        else                      { r=255; g=60;  b=0;   alpha = 0.2 + fadeAlpha*0.7; }
      }

      ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.lineWidth   = 0.5 + norm * 4;
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();

      // Weight label on edge midpoint
      const mx = (ax+bx)/2, my = (ay+by)/2;
      ctx.fillStyle = flash
        ? (flash.type==='ltp' ? `rgba(0,255,120,${Math.max(0.4,1-flash.age/FLASH_DUR)})` : `rgba(255,80,0,${Math.max(0.4,1-flash.age/FLASH_DUR)})`)
        : "rgba(60,110,160,0.55)";
      ctx.font = "6px 'Courier New'";
      ctx.textAlign = "center";
      ctx.fillText(w.toFixed(1), mx, my);
    }

    // Pulse particles
    for (const p of pulses) {
      const A = NODES[p.from], B = NODES[p.to];
      const frac = Math.min(1, p.t / p.dur);
      const px = A.x*W + (B.x*W - A.x*W)*frac;
      const py = A.y*H + (B.y*H - A.y*H)*frac;
      const g = ctx.createRadialGradient(px,py,0,px,py,8);
      g.addColorStop(0,"rgba(0,255,180,0.9)"); g.addColorStop(1,"rgba(0,255,180,0)");
      ctx.beginPath(); ctx.arc(px,py,8,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
    }

    // Neurons
    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      const cx = n.x*W, cy = n.y*H;
      const norm = Math.max(0, Math.min(1, (n.v - V_REST)/(V_THRESH - V_REST)));
      const sel = i === selNeuronRef.current;

      if (n.spiked) {
        const glow = ctx.createRadialGradient(cx,cy,0,cx,cy,36);
        glow.addColorStop(0,"rgba(0,255,150,0.45)"); glow.addColorStop(1,"rgba(0,255,150,0)");
        ctx.beginPath(); ctx.arc(cx,cy,36,0,Math.PI*2); ctx.fillStyle=glow; ctx.fill();
      }

      // STDP trace indicator ring
      const traceR = 18 + n.xPre * 6;
      ctx.strokeStyle = `rgba(0,200,255,${Math.min(0.6, n.xPre * 0.5)})`;
      ctx.lineWidth = 1; ctx.setLineDash([2,2]);
      ctx.beginPath(); ctx.arc(cx,cy,traceR,0,Math.PI*2); ctx.stroke();
      ctx.setLineDash([]);

      if (sel) {
        ctx.strokeStyle = "rgba(255,200,0,0.7)";
        ctx.lineWidth = 2; ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.arc(cx,cy,21,0,Math.PI*2); ctx.stroke();
        ctx.setLineDash([]);
      }

      const R = 13;
      const bg = ctx.createRadialGradient(cx-3,cy-3,1,cx,cy,R);
      if (n.spiked) {
        bg.addColorStop(0,"#00FFB0"); bg.addColorStop(1,"#006644");
      } else if (n.type==="input") {
        bg.addColorStop(0,`rgb(${20+norm*100},${100+norm*100},220)`);
        bg.addColorStop(1,`rgb(${10+norm*50},${40+norm*60},100)`);
      } else if (n.type==="output") {
        bg.addColorStop(0,`rgb(${200+norm*55},${80+norm*80},${20+norm*20})`);
        bg.addColorStop(1,`rgb(${80+norm*40},${30+norm*30},10)`);
      } else {
        bg.addColorStop(0,`rgb(${20+norm*60},${60+norm*160},${180+norm*40})`);
        bg.addColorStop(1,`rgb(${10+norm*30},${30+norm*80},80)`);
      }
      ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.fillStyle=bg; ctx.fill();
      ctx.strokeStyle = n.spiked ? "#00FFB0" : `rgba(0,${80+norm*175},255,0.8)`;
      ctx.lineWidth = n.spiked ? 2 : 1; ctx.setLineDash([]);
      ctx.stroke();

      ctx.fillStyle = n.spiked ? "#00FFB0" : "rgba(150,200,255,0.9)";
      ctx.font = "bold 8px 'Courier New'"; ctx.textAlign="center";
      ctx.fillText(n.label, cx, cy+R+11);
      ctx.fillStyle="rgba(80,140,200,0.65)"; ctx.font="7px 'Courier New'";
      ctx.fillText(`${n.v.toFixed(0)}mV`, cx, cy+R+21);
    }
  }, []);

  // ── Draw raster ───────────────────────────────────────────────
  const drawRaster = useCallback((ctx, W, H) => {
    const { neurons, time } = simState.current;
    ctx.fillStyle = "#060A10"; ctx.fillRect(0,0,W,H);
    const WINDOW = 500, N = neurons.length;
    neurons.forEach((n, i) => {
      const rowH = H/N, ry = i*rowH;
      ctx.strokeStyle="rgba(0,80,120,0.2)"; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(0,ry); ctx.lineTo(W,ry); ctx.stroke();
      ctx.fillStyle = n.type==="input"?"rgba(80,160,255,0.7)":n.type==="output"?"rgba(255,140,60,0.7)":"rgba(0,200,180,0.6)";
      ctx.font="7px 'Courier New'"; ctx.textAlign="left";
      ctx.fillText(n.label, 3, ry+rowH*0.65);
      n.spikes.forEach(st => {
        const age = time-st;
        if (age<WINDOW) {
          const x=W*(1-age/WINDOW), alpha=Math.max(0,1-age/WINDOW);
          ctx.fillStyle=`rgba(0,255,150,${alpha})`;
          ctx.fillRect(x-1, ry+rowH*0.15, 2, rowH*0.7);
        }
      });
    });
    ctx.strokeStyle="rgba(255,100,0,0.25)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(W-1,0); ctx.lineTo(W-1,H); ctx.stroke();
    ctx.fillStyle="rgba(60,100,140,0.7)"; ctx.font="7px 'Courier New'";
    ctx.textAlign="right"; ctx.fillText(`−${WINDOW}ms`,W-2,H-3);
    ctx.textAlign="left";  ctx.fillText("now",2,H-3);
  }, []);

  // ── Draw voltage trace ─────────────────────────────────────────
  const drawTrace = useCallback((ctx, W, H) => {
    const { neurons } = simState.current;
    const n = neurons[selNeuronRef.current];
    if (!n) return;
    ctx.fillStyle="#04080E"; ctx.fillRect(0,0,W,H);
    const normT=(V_THRESH-V_REST)/20, yT=H-normT*H;
    ctx.strokeStyle="rgba(255,80,0,0.3)"; ctx.setLineDash([4,4]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,yT); ctx.lineTo(W,yT); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle="rgba(255,80,0,0.5)"; ctx.font="7px 'Courier New'"; ctx.textAlign="left";
    ctx.fillText(`θ ${V_THRESH}mV`,3,yT-3);
    const buf=n.vBuf, V_SPAN=30, sw=W/buf.length;
    ctx.beginPath();
    for (let i=0; i<buf.length; i++) {
      const x=i*sw, norm=(buf[i]-(V_REST-5))/V_SPAN, y=H-norm*H;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    const grad=ctx.createLinearGradient(0,0,W,0);
    grad.addColorStop(0,"rgba(0,180,255,0.1)"); grad.addColorStop(1,"rgba(0,255,180,0.9)");
    ctx.strokeStyle=grad; ctx.lineWidth=1.5; ctx.stroke();
    const lastX=(buf.length-1)*sw, lastNorm=(buf[buf.length-1]-(V_REST-5))/V_SPAN;
    ctx.lineTo(lastX,H); ctx.lineTo(0,H); ctx.closePath();
    const fillG=ctx.createLinearGradient(0,0,0,H);
    fillG.addColorStop(0,"rgba(0,255,180,0.15)"); fillG.addColorStop(1,"rgba(0,255,180,0)");
    ctx.fillStyle=fillG; ctx.fill();
    ctx.fillStyle="rgba(0,220,160,0.8)"; ctx.font="7px 'Courier New'"; ctx.textAlign="left";
    ctx.fillText(`Vm — ${n.label}  ${n.v.toFixed(1)}mV  |  xPre: ${n.xPre.toFixed(3)}  xPost: ${n.xPost.toFixed(3)}`,4,10);
  }, []);

  // ── Draw weight matrix ────────────────────────────────────────
  const drawWeightMatrix = useCallback((ctx, W, H) => {
    const { weights, edgeFlash } = simState.current;
    ctx.fillStyle="#04080E"; ctx.fillRect(0,0,W,H);
    const cols=5, rows=Math.ceil(EDGE_DEFS.length/cols);
    const cw=W/cols, rh=H/rows;
    for (let e=0; e<EDGE_DEFS.length; e++) {
      const col=e%cols, row=Math.floor(e/cols);
      const x=col*cw, y=row*rh;
      const w=weights[e], norm=w/W_MAX;
      const flash=edgeFlash[e];
      let bg = `rgba(0,${Math.floor(40+norm*160)},${Math.floor(80+norm*120)},0.5)`;
      if (flash) {
        const f=Math.max(0,1-flash.age/FLASH_DUR);
        bg = flash.type==='ltp'
          ? `rgba(0,${Math.floor(150+f*105)},${Math.floor(80*f)},0.7)`
          : `rgba(${Math.floor(150+f*105)},${Math.floor(40*f)},0,0.7)`;
      }
      ctx.fillStyle=bg;
      ctx.fillRect(x+1,y+1,cw-2,rh-2);
      ctx.fillStyle=flash
        ? (flash.type==='ltp'?"rgba(0,255,120,0.95)":"rgba(255,100,0,0.95)")
        : "rgba(120,180,220,0.8)";
      ctx.font="7px 'Courier New'"; ctx.textAlign="center";
      ctx.fillText(`${NODES[EDGE_DEFS[e].f].label}→${NODES[EDGE_DEFS[e].t].label}`, x+cw/2, y+rh*0.38);
      ctx.fillStyle=flash
        ? (flash.type==='ltp'?"#00FF80":"#FF6020")
        : "rgba(200,230,255,0.9)";
      ctx.font="bold 8px 'Courier New'";
      ctx.fillText(w.toFixed(2), x+cw/2, y+rh*0.75);
    }
  }, []);

  // ── Main render ───────────────────────────────────────────────
  const renderAll = useCallback(() => {
    const net=netRef.current, raster=rasterRef.current, trace=traceRef.current, wt=wtRef.current;
    if (net)    drawNet(net.getContext("2d"),       net.width,    net.height);
    if (raster) drawRaster(raster.getContext("2d"), raster.width, raster.height);
    if (trace)  drawTrace(trace.getContext("2d"),   trace.width,  trace.height);
    if (wt)     drawWeightMatrix(wt.getContext("2d"), wt.width,   wt.height);
  }, [drawNet, drawRaster, drawTrace, drawWeightMatrix]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    let last=0;
    const loop = ts => {
      if (ts-last>16) { for (let i=0;i<10;i++) step(); renderAll(); last=ts; }
      animRef.current=requestAnimationFrame(loop);
    };
    animRef.current=requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, step, renderAll]);

  useEffect(() => { renderAll(); }, [renderAll]);

  const reset = () => { simState.current=freshState(); setRunning(false); setEventLog([]); setTimeout(renderAll,50); };
  const handleNetClick = e => {
    const canvas=netRef.current, rect=canvas.getBoundingClientRect();
    const mx=(e.clientX-rect.left)/rect.width, my=(e.clientY-rect.top)/rect.height;
    let closest=8, dist=9999;
    NODES.forEach(n => { const d=Math.hypot(n.x-mx,n.y-my); if(d<dist){dist=d;closest=n.id;} });
    if (dist<0.08) setSelNeuron(closest);
  };

  const mono="'Courier New', monospace";
  const s = simState.current;

  return (
    <div style={{ background:"#03060B", minHeight:"100vh", color:"#7FC8FF", fontFamily:mono, padding:"10px", boxSizing:"border-box", display:"flex", flexDirection:"column", gap:"8px" }}>

      {/* Header */}
      <div style={{ display:"flex", alignItems:"baseline", gap:"12px", borderBottom:"1px solid rgba(0,100,160,0.3)", paddingBottom:"7px" }}>
        <span style={{ color:"#00FFB0", fontSize:"13px", letterSpacing:"0.2em", fontWeight:"bold" }}>NEURO·SIM v2.0</span>
        <span style={{ color:"rgba(0,150,200,0.6)", fontSize:"9px", letterSpacing:"0.12em" }}>LIF + SPIKE-TIMING DEPENDENT PLASTICITY</span>
        <div style={{ marginLeft:"auto", display:"flex", gap:"8px", alignItems:"center" }}>
          <span style={{ fontSize:"8px", color: s.ltpCount>0?"rgba(0,255,120,0.7)":"rgba(60,100,140,0.5)" }}>LTP: {s.ltpCount}</span>
          <span style={{ fontSize:"8px", color: s.ltdCount>0?"rgba(255,100,0,0.7)":"rgba(60,100,140,0.5)" }}>LTD: {s.ltdCount}</span>
          <span style={{ fontSize:"8px", color:"rgba(60,100,140,0.5)" }}>t={s.time.toFixed(0)}ms</span>
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 230px", gap:"8px" }}>

        {/* Left column */}
        <div style={{ display:"flex", flexDirection:"column", gap:"8px" }}>

          {/* Network */}
          <div>
            <div style={{ color:"rgba(0,120,180,0.5)", fontSize:"8px", letterSpacing:"0.15em", marginBottom:"4px" }}>
              NETWORK · edge width/colour ∝ weight · dashed ring = xPre trace · click to probe
            </div>
            <canvas ref={netRef} width={680} height={300} onClick={handleNetClick}
              style={{ width:"100%", display:"block", cursor:"crosshair", borderRadius:"3px", border:"1px solid rgba(0,80,120,0.3)" }} />
          </div>

          {/* Voltage trace */}
          <div>
            <div style={{ color:"rgba(0,120,180,0.5)", fontSize:"8px", letterSpacing:"0.15em", marginBottom:"4px" }}>MEMBRANE POTENTIAL · {NODES[selNeuron]?.label}</div>
            <canvas ref={traceRef} width={680} height={80}
              style={{ width:"100%", display:"block", borderRadius:"3px", border:"1px solid rgba(0,80,120,0.3)" }} />
          </div>

          {/* Weight matrix */}
          <div>
            <div style={{ color:"rgba(0,120,180,0.5)", fontSize:"8px", letterSpacing:"0.15em", marginBottom:"4px" }}>
              SYNAPTIC WEIGHT MATRIX · <span style={{color:"rgba(0,255,120,0.6)"}}>■ LTP</span> · <span style={{color:"rgba(255,100,0,0.6)"}}>■ LTD</span>
            </div>
            <canvas ref={wtRef} width={680} height={90}
              style={{ width:"100%", display:"block", borderRadius:"3px", border:"1px solid rgba(0,80,120,0.3)" }} />
          </div>

          {/* Event log */}
          <div style={{ background:"rgba(0,15,30,0.8)", border:"1px solid rgba(0,80,120,0.3)", borderRadius:"3px", padding:"6px", height:"70px", overflowY:"auto" }}>
            <div style={{ color:"rgba(0,120,180,0.5)", fontSize:"7px", letterSpacing:"0.15em", marginBottom:"4px" }}>STDP EVENT LOG (sampled)</div>
            {eventLog.length===0
              ? <div style={{ color:"rgba(60,100,140,0.5)", fontSize:"7px" }}>no events yet — run simulation with learning enabled</div>
              : eventLog.map((ev,i) => (
                <div key={i} style={{ fontSize:"7px", color: ev.type==='LTP'?"rgba(0,255,120,0.8)":"rgba(255,100,0,0.8)", lineHeight:"1.6" }}>
                  t={ev.t}ms · {ev.type} · {NODES[EDGE_DEFS[ev.e].f].label}→{NODES[EDGE_DEFS[ev.e].t].label} · Δw={ev.dw} → w={ev.w}
                </div>
              ))
            }
          </div>
        </div>

        {/* Right column */}
        <div style={{ display:"flex", flexDirection:"column", gap:"8px" }}>

          {/* Raster */}
          <div>
            <div style={{ color:"rgba(0,120,180,0.5)", fontSize:"8px", letterSpacing:"0.15em", marginBottom:"4px" }}>SPIKE RASTER</div>
            <canvas ref={rasterRef} width={230} height={200}
              style={{ width:"100%", display:"block", borderRadius:"3px", border:"1px solid rgba(0,80,120,0.3)" }} />
          </div>

          {/* Controls */}
          <div style={{ background:"rgba(0,20,40,0.8)", border:"1px solid rgba(0,80,120,0.3)", borderRadius:"3px", padding:"10px", display:"flex", flexDirection:"column", gap:"9px" }}>

            {/* Learning toggle */}
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <span style={{ fontSize:"8px", color:"rgba(0,120,180,0.5)", letterSpacing:"0.1em" }}>STDP LEARNING</span>
              <button onClick={() => setLearning(l => !l)} style={{
                padding:"3px 10px", fontFamily:mono, fontSize:"8px", letterSpacing:"0.1em",
                background: learning ? "rgba(0,255,150,0.15)" : "rgba(60,60,80,0.3)",
                border: `1px solid ${learning ? "rgba(0,255,150,0.6)" : "rgba(80,80,100,0.4)"}`,
                color: learning ? "#00FFB0" : "rgba(80,120,160,0.7)",
                borderRadius:"2px", cursor:"pointer",
              }}>{learning ? "ON" : "OFF"}</button>
            </div>

            {[
              { key:"inputRate",    label:"INPUT RATE",  unit:"Hz",  min:5,  max:80, step:1    },
              { key:"inputCurrent", label:"EXT CURRENT", unit:"pA",  min:5,  max:30, step:1    },
              { key:"noise",        label:"NOISE",       unit:"σ",   min:0,  max:8,  step:0.5  },
              { key:"aPlus",        label:"A⁺ (LTP)",   unit:"",    min:0.001, max:0.03, step:0.001 },
              { key:"aMinus",       label:"A⁻ (LTD)",   unit:"",    min:0.001, max:0.03, step:0.001 },
            ].map(({ key, label, unit, min, max, step }) => (
              <div key={key}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:"7px", color:"rgba(80,160,220,0.7)", marginBottom:"3px" }}>
                  <span>{label}</span>
                  <span style={{ color: key==="aPlus"?"rgba(0,255,120,0.8)": key==="aMinus"?"rgba(255,100,0,0.8)":"#00FFB0" }}>
                    {params[key]}{unit}
                  </span>
                </div>
                <input type="range" min={min} max={max} step={step} value={params[key]}
                  onChange={e => setParams(prev => ({ ...prev, [key]: +e.target.value }))}
                  style={{ width:"100%", accentColor: key==="aPlus"?"#00FF80": key==="aMinus"?"#FF6020":"#00D4AA", cursor:"pointer" }} />
              </div>
            ))}

            {/* Probe selector */}
            <div>
              <div style={{ fontSize:"7px", color:"rgba(80,160,220,0.7)", marginBottom:"4px" }}>PROBE NEURON</div>
              <div style={{ display:"flex", flexWrap:"wrap", gap:"3px" }}>
                {NODES.map(n => (
                  <button key={n.id} onClick={() => setSelNeuron(n.id)} style={{
                    padding:"2px 5px", fontSize:"7px", fontFamily:mono,
                    background: selNeuron===n.id ? "rgba(0,200,150,0.2)" : "rgba(0,40,80,0.5)",
                    border: `1px solid ${selNeuron===n.id ? "#00FFB0" : "rgba(0,80,120,0.4)"}`,
                    color: selNeuron===n.id ? "#00FFB0" : "rgba(80,160,220,0.7)",
                    borderRadius:"2px", cursor:"pointer",
                  }}>{n.label}</button>
                ))}
              </div>
            </div>

            {/* Run / Reset */}
            <div style={{ display:"flex", gap:"8px" }}>
              <button onClick={() => setRunning(r => !r)} style={{
                flex:1, padding:"7px 0", fontFamily:mono, fontSize:"9px", letterSpacing:"0.1em",
                background: running ? "rgba(255,60,0,0.15)" : "rgba(0,200,150,0.15)",
                border: `1px solid ${running ? "rgba(255,60,0,0.5)" : "rgba(0,255,150,0.5)"}`,
                color: running ? "#FF6030" : "#00FFB0",
                borderRadius:"3px", cursor:"pointer",
              }}>{running ? "■ HALT" : "▶ RUN"}</button>
              <button onClick={reset} style={{
                padding:"7px 10px", fontFamily:mono, fontSize:"9px",
                background:"rgba(0,40,80,0.5)", border:"1px solid rgba(0,80,120,0.4)",
                color:"rgba(80,160,220,0.7)", borderRadius:"3px", cursor:"pointer",
              }}>RST</button>
            </div>

            {/* Legend */}
            <div style={{ fontSize:"7px", color:"rgba(60,100,140,0.7)", lineHeight:"1.9", borderTop:"1px solid rgba(0,60,100,0.3)", paddingTop:"7px" }}>
              <div><span style={{color:"rgba(0,255,120,0.8)"}}>■</span> LTP — pre→post, weight ↑</div>
              <div><span style={{color:"rgba(255,100,0,0.8)"}}>■</span> LTD — post→pre, weight ↓</div>
              <div><span style={{color:"rgba(0,200,255,0.6)"}}>◌</span> dashed ring = xPre trace</div>
              <div>Edge width ∝ current weight</div>
              <div>A⁺ &gt; A⁻ → net potentiation</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
