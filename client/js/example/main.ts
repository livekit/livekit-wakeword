import '@fontsource-variable/public-sans';
import '@fontsource/commit-mono';
import { WakeWordListener, WakeWordModel } from '@livekit/wakeword';

const THRESHOLD = 0.5;
const WINDOW_MS = 8_000; // history on the chart
const COLORS = ['#1FD5F9', '#DC85FF', '#23DE6B', '#FFB752'];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const button = $<HTMLButtonElement>('start');
const status = $('status');
const title = $('title');
const stage = $('stage');
const canvas = $<HTMLCanvasElement>('chart');
const chips = $('models');
const fileInput = $<HTMLInputElement>('file');
const log = $('log');

interface Track {
  color: string;
  chip: HTMLElement;
  score: HTMLOutputElement;
}
const tracks = new Map<string, Track>();
const steps: { t: number; scores: Record<string, number> }[] = [];
const levels: { t: number; v: number }[] = [];
const hits: { t: number; name: string }[] = [];

interface Session {
  listener: WakeWordListener;
  stream: MediaStream;
  context: AudioContext;
  analyser: AnalyserNode;
}
let session: Session | null = null;

function setStatus(text: string, error = false) {
  status.textContent = text;
  status.classList.toggle('error', error);
}

/** "Say “hey livekit” or “my phrase”" from the loaded models. */
function renderTitle() {
  title.replaceChildren();
  const names = [...tracks.keys()];
  if (!names.length) {
    title.append('Load a model');
    return;
  }
  title.append('Say ');
  names.forEach((name, i) => {
    if (i) title.append(' or ');
    const em = document.createElement('em');
    em.textContent = `“${name.replace(/_/g, ' ')}”`;
    title.append(em);
  });
}

function addChip(name: string) {
  const color = COLORS[[...tracks.keys()].length % COLORS.length] as string;
  const chip = document.createElement('span');
  chip.className = 'chip';
  chip.style.setProperty('--c', color);
  const score = document.createElement('output');
  score.dataset.model = name;
  score.textContent = '0.00';
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.ariaLabel = `Remove ${name}`;
  remove.innerHTML =
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" stroke="currentColor" stroke-linecap="round"/></svg>';
  remove.onclick = async () => {
    await model.unloadModel(name);
    tracks.delete(name);
    chip.remove();
    renderTitle();
  };
  chip.append(document.createElement('i'), name, score, remove);
  chips.insertBefore(chip, chips.lastElementChild);
  tracks.set(name, { color, chip, score });
}

async function addModel(source: string | ArrayBuffer, name: string) {
  await model.loadModel(source, name);
  if (!tracks.has(name)) addChip(name);
  renderTitle();
}

async function addFiles(files: Iterable<File>) {
  for (const file of files) {
    try {
      await addModel(await file.arrayBuffer(), file.name.replace(/\.onnx$/i, ''));
      if (!session) setStatus('');
    } catch (e) {
      setStatus((e as Error).message, true);
    }
  }
}

const model = await WakeWordModel.create();
try {
  await addModel('hey_livekit.onnx', 'hey_livekit'); // next to the page
  setStatus('');
} catch (e) {
  setStatus((e as Error).message, true);
}
button.disabled = false;

function onScores(scores: Record<string, number>) {
  steps.push({ t: performance.now(), scores });
  for (const [name, s] of Object.entries(scores)) {
    const track = tracks.get(name);
    if (!track) continue;
    track.score.textContent = s.toFixed(2);
    track.chip.classList.toggle('hot', s >= THRESHOLD);
  }
}

function onDetection(name: string, confidence: number) {
  hits.push({ t: performance.now(), name });
  const color = tracks.get(name)?.color ?? COLORS[0];
  const li = document.createElement('li');
  li.style.setProperty('--c', color as string);
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString([], { hour12: false });
  const label = document.createElement('span');
  label.textContent = name;
  const score = document.createElement('output');
  score.textContent = confidence.toFixed(2);
  li.append(document.createElement('i'), label, time, score);
  log.prepend(li);
  while (log.children.length > 8) log.lastElementChild?.remove();
  for (const el of [stage, title]) {
    el.classList.add('hit');
    setTimeout(() => el.classList.remove('hit'), 700);
  }
}

button.onclick = async () => {
  if (session) {
    await session.listener.stop(); // ends the detections() loop below
    return;
  }
  button.disabled = true;
  setStatus('starting');
  let current: Session | null = null;
  try {
    const context = new AudioContext(); // in the click handler, so it may start
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const listener = new WakeWordListener(model, { threshold: THRESHOLD, debounce: 2, onScores });
    current = { listener, stream, context, analyser };
    session = current;
    await listener.start({ source: stream, audioContext: context });
    button.classList.add('on');
    button.ariaLabel = 'Stop listening';
    button.disabled = false;
    setStatus('listening');
    for await (const d of listener.detections()) onDetection(d.name, d.confidence);
    setStatus('stopped');
  } catch (e) {
    console.error(e);
    setStatus((e as Error).message, true);
  } finally {
    if (current) {
      for (const track of current.stream.getTracks()) track.stop();
      await current.context.close();
    }
    session = null;
    button.classList.remove('on');
    button.ariaLabel = 'Start listening';
    button.style.setProperty('--level', '0');
    button.disabled = false;
  }
};

fileInput.onchange = () => {
  if (fileInput.files) void addFiles([...fileInput.files]);
  fileInput.value = '';
};
let depth = 0; // dragenter/dragleave fire for every child
addEventListener('dragenter', (e) => {
  e.preventDefault();
  if (depth++ === 0) document.body.classList.add('dragging');
});
addEventListener('dragleave', () => {
  if (--depth === 0) document.body.classList.remove('dragging');
});
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  depth = 0;
  document.body.classList.remove('dragging');
  const files = [...(e.dataTransfer?.files ?? [])].filter((f) => /\.onnx$/i.test(f.name));
  void addFiles(files);
});

// the chart: one trace per model over the last WINDOW_MS, the threshold, detections and mic level
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
const wave = new Float32Array(1024);
let level = 0;

function draw(now: number) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const top = 20;
  const bottom = h - 12;
  const x = (t: number) => w - ((now - t) / WINDOW_MS) * w;
  const y = (s: number) => bottom - Math.min(1, Math.max(0, s)) * (bottom - top);
  const old = now - WINDOW_MS - 500;
  while (steps.length && (steps[0] as { t: number }).t < old) steps.shift();
  while (levels.length && (levels[0] as { t: number }).t < old) levels.shift();
  while (hits.length && (hits[0] as { t: number }).t < old) hits.shift();

  if (session) {
    session.analyser.getFloatTimeDomainData(wave);
    let sum = 0;
    for (const v of wave) sum += v * v;
    level += (Math.min(1, Math.sqrt(sum / wave.length) * 5) - level) * 0.3;
    levels.push({ t: now, v: level });
    button.style.setProperty('--level', level.toFixed(3));
  }

  // mic level: a faint band along the bottom
  ctx.fillStyle = 'rgba(255,255,255,0.05)';
  for (const { t, v } of levels) {
    const bh = v * (bottom - top) * 0.4;
    ctx.fillRect(x(t), bottom - bh, 2, bh);
  }

  // threshold
  ctx.setLineDash([2, 6]);
  ctx.strokeStyle = '#30302F';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, y(THRESHOLD));
  ctx.lineTo(w, y(THRESHOLD));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = '10px "Commit Mono", ui-monospace, monospace';
  ctx.fillStyle = '#666666';
  ctx.fillText(THRESHOLD.toFixed(1), 10, y(THRESHOLD) - 6);

  // detections
  for (const { t, name } of hits) {
    const color = tracks.get(name)?.color ?? (COLORS[0] as string);
    const fade = Math.max(0, 1 - (now - t) / WINDOW_MS);
    ctx.globalAlpha = 0.25 + 0.5 * fade;
    ctx.fillStyle = color;
    ctx.fillRect(x(t), top - 8, 1, bottom - top + 8);
    ctx.beginPath();
    ctx.arc(x(t), top - 8, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // one trace per model, the highest on top; flat at 0 until scores arrive
  const latest = (name: string) => steps.at(-1)?.scores[name] ?? 0;
  const order = [...tracks].sort(([a], [b]) => latest(a) - latest(b));
  for (const [name, { color }] of order) {
    const pts = steps
      .filter((p) => name in p.scores)
      .map((p) => [x(p.t), y(p.scores[name] as number)] as const);
    if (pts.length < 2) pts.splice(0, pts.length, [0, bottom], [w, bottom]);
    const line = new Path2D();
    smooth(line, pts);
    const [lx, ly] = pts.at(-1) as readonly [number, number];
    const area = new Path2D();
    area.moveTo((pts[0] as readonly [number, number])[0], bottom);
    smooth(area, pts, true);
    area.lineTo(lx, bottom);
    const gradient = ctx.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, `${color}55`);
    gradient.addColorStop(1, `${color}00`);
    ctx.fillStyle = gradient;
    ctx.fill(area);
    const hot = latest(name) >= THRESHOLD;
    ctx.shadowColor = color;
    ctx.shadowBlur = hot ? 18 : 8;
    ctx.strokeStyle = color;
    ctx.globalAlpha = session ? 1 : 0.5;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke(line);
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    if (session) {
      ctx.beginPath();
      ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
  requestAnimationFrame(draw);
}

/**
 * A smooth curve through every point of `pts` (Catmull-Rom, so peaks keep their true height;
 * control points are kept within the chart); `join` continues the current path.
 */
function smooth(path: Path2D, pts: readonly (readonly [number, number])[], join = false) {
  const at = (i: number) =>
    pts[Math.max(0, Math.min(pts.length - 1, i))] as readonly [number, number];
  const [x0, y0] = at(0);
  if (join) path.lineTo(x0, y0);
  else path.moveTo(x0, y0);
  const lo = Math.min(...pts.map((p) => p[1]));
  const hi = Math.max(...pts.map((p) => p[1]));
  const clamp = (v: number) => Math.max(lo, Math.min(hi, v));
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = at(i - 1);
    const [bx, by] = at(i);
    const [cx, cy] = at(i + 1);
    const [dx, dy] = at(i + 2);
    path.bezierCurveTo(
      bx + (cx - ax) / 6,
      clamp(by + (cy - ay) / 6),
      cx - (dx - bx) / 6,
      clamp(cy - (dy - by) / 6),
      cx,
      cy,
    );
  }
}
requestAnimationFrame(draw);
