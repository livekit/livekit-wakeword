import '@fontsource-variable/public-sans';
import '@fontsource/commit-mono';
import { WakeWordListener, WakeWordModel } from '@livekit/wakeword';

const THRESHOLD = 0.5;
const WINDOW_MS = 6_000; // history on the chart
const STEP_MS = 80; // the models score 80 ms steps
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
  count: HTMLElement;
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
  const count = document.createElement('b'); // detections so far
  chip.append(document.createElement('i'), name, score, count, remove);
  chips.insertBefore(chip, chips.lastElementChild);
  tracks.set(name, { color, chip, score, count });
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
  // one step per 80 ms of audio, but they can arrive in bursts: space a burst out backwards
  steps.push({ t: performance.now(), scores });
  for (let i = steps.length - 2; i >= 0; i--) {
    const [a, b] = [steps[i], steps[i + 1]] as [(typeof steps)[number], (typeof steps)[number]];
    if (a.t <= b.t - STEP_MS) break;
    a.t = b.t - STEP_MS;
  }
  for (const [name, s] of Object.entries(scores)) {
    const track = tracks.get(name);
    if (!track) continue;
    track.score.textContent = s.toFixed(2);
    track.chip.classList.toggle('hot', s >= THRESHOLD);
  }
}

function onDetection(name: string, confidence: number) {
  hits.push({ t: performance.now(), name });
  const track = tracks.get(name);
  if (track) track.count.textContent = String(Number(track.count.textContent) + 1);
  const li = document.createElement('li'); // for screen readers (and the browser test)
  li.textContent = `${name} ${confidence.toFixed(2)}`;
  log.prepend(li);
  while (log.children.length > 8) log.lastElementChild?.remove();
  for (const el of [stage, title]) {
    el.classList.add('hit');
    setTimeout(() => el.classList.remove('hit'), 700);
  }
}

/** Listens until paused. Starts on load; the browser may wait for a first tap to play audio. */
let starting = false;
async function listen() {
  starting = true;
  button.disabled = true;
  setStatus('starting');
  let current: Session | null = null;
  try {
    const context = new AudioContext();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    if (context.state !== 'running') {
      const resume = () => void context.resume();
      setStatus('tap to listen');
      button.disabled = false; // a tap on it counts too
      addEventListener('pointerdown', resume, { once: true });
      addEventListener('keydown', resume, { once: true });
      await context.resume(); // resolves once the page may play audio
    }
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const listener = new WakeWordListener(model, { threshold: THRESHOLD, debounce: 2, onScores });
    current = { listener, stream, context, analyser };
    session = current;
    starting = false;
    await listener.start({ source: stream, audioContext: context });
    button.classList.add('on');
    button.ariaLabel = 'Pause';
    button.disabled = false;
    setStatus('listening');
    for await (const d of listener.detections()) onDetection(d.name, d.confidence);
    setStatus('paused');
  } catch (e) {
    console.error(e);
    setStatus((e as Error).message, true);
  } finally {
    if (current) {
      for (const track of current.stream.getTracks()) track.stop();
      await current.context.close();
    }
    session = null;
    starting = false;
    button.classList.remove('on');
    button.ariaLabel = 'Listen';
    button.style.setProperty('--level', '0');
    button.disabled = false;
  }
}

button.onclick = () => {
  if (starting) return; // listen() is waiting for this tap
  if (session)
    void session.listener.stop(); // ends the detections() loop in listen()
  else void listen();
};
void listen();

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

// the display: a dot matrix (as in LiveKit's product art), one column per moment over the last
// WINDOW_MS; a column lights up to the highest score, in that model's color
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
const PITCH = 10; // dot spacing, CSS px
const DOT = 3;
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
  const cols = Math.max(1, Math.floor(w / PITCH));
  const rows = Math.max(2, Math.floor(h / PITCH));
  const x0 = (w - (cols - 1) * PITCH - DOT) / 2;
  const y0 = (h - (rows - 1) * PITCH - DOT) / 2;
  const span = WINDOW_MS / cols; // ms per column
  const last = Math.floor(now / span);
  const old = now - WINDOW_MS - span;
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

  // per column: the highest score (and its model), the mic level, a detection
  const col = (t: number) => cols - 1 - (last - Math.floor(t / span));
  const score = new Float32Array(cols);
  const owner: (string | undefined)[] = new Array(cols);
  const mic = new Float32Array(cols);
  const hit: (string | undefined)[] = new Array(cols);
  // each step holds until the next, so a score covers its whole 80 ms, not one column
  let i = 0;
  for (let c = 0; c < cols; c++) {
    const end = (last - (cols - 1 - c) + 1) * span; // this column's end time
    while (i < steps.length && (steps[i] as { t: number }).t < end) {
      const { t, scores } = steps[i] as (typeof steps)[number];
      i++;
      if (col(t) !== c) continue; // older than the window
      for (const [name, s] of Object.entries(scores)) {
        if (tracks.has(name) && s >= score[c]) {
          score[c] = s;
          owner[c] = name;
        }
      }
    }
    // no step in this column: hold the previous one (up to the latest step)
    const newest = steps.at(-1)?.t ?? 0;
    if (!owner[c] && c > 0 && owner[c - 1] && end - span <= newest) {
      score[c] = score[c - 1] as number;
      owner[c] = owner[c - 1];
    }
  }
  // a peak fades over the next columns (like a level meter), so a 160 ms spike stays readable
  const newestCol = col(steps.at(-1)?.t ?? 0);
  for (let c = 1; c <= newestCol; c++) {
    const tail = (score[c - 1] as number) * 0.8;
    if (owner[c - 1] && tail > (score[c] as number)) {
      score[c] = tail;
      owner[c] = owner[c - 1];
    }
  }
  for (const { t, v } of levels) {
    const c = col(t);
    if (c >= 0) mic[c] = Math.max(mic[c] as number, v);
  }
  for (const { t, name } of hits) {
    const c = col(t);
    if (c >= 0) hit[c] = name;
  }
  if (!session) {
    for (let c = 0; c < cols; c++) mic[c] = 0.06 + 0.04 * Math.sin(c * 0.18 - now / 600); // idle hum
  }

  // batch the dots by style: one fill per style
  const paths = new Map<string, Path2D>();
  const add = (style: string, c: number, r: number) => {
    let p = paths.get(style);
    if (!p) {
      p = new Path2D();
      paths.set(style, p);
    }
    p.rect(x0 + c * PITCH, y0 + (rows - 1 - r) * PITCH, DOT, DOT);
  };
  const threshold = Math.round(THRESHOLD * rows - 0.5);
  for (let c = 0; c < cols; c++) {
    const s = score[c] as number;
    const color = owner[c] ? tracks.get(owner[c] as string)?.color : undefined;
    const marked = hit[c] ? tracks.get(hit[c] as string)?.color : undefined;
    for (let r = 0; r < rows; r++) {
      const v = (r + 0.5) / rows;
      if (color && v <= s) add(s >= THRESHOLD ? `hot ${color}` : `lit ${color}`, c, r);
      else if (marked) add(`mark ${marked}`, c, r);
      else if (v <= (mic[c] as number) * 0.6) add('mic', c, r);
      else if (r === threshold) add('threshold', c, r);
      else add('off', c, r);
    }
  }
  for (const [style, path] of paths) {
    const [kind, color = ''] = style.split(' ');
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
    if (kind === 'off') ctx.fillStyle = 'rgba(255,255,255,0.07)';
    else if (kind === 'mic') ctx.fillStyle = 'rgba(255,255,255,0.22)';
    else if (kind === 'threshold') ctx.fillStyle = '#035C6D';
    else {
      ctx.fillStyle = color;
      ctx.globalAlpha = kind === 'hot' ? 1 : kind === 'lit' ? 0.6 : 0.3;
      if (kind === 'hot') {
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
      }
    }
    ctx.fill(path);
  }
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
