import { WakeWordListener, WakeWordModel } from '@livekit/wakeword';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const button = $<HTMLButtonElement>('start');
const status = $('status');

let listener: WakeWordListener | null = null;

button.onclick = async () => {
  if (listener?.isRunning) {
    await listener.stop();
    return;
  }
  button.disabled = true;
  status.textContent = 'loading model…';
  try {
    const model = await WakeWordModel.create(['/hey_livekit.onnx']);
    listener = new WakeWordListener(model, {
      threshold: 0.5,
      debounce: 2,
      onScores: ({ hey_livekit: score }) => {
        $('score').textContent = `score ${score.toFixed(2)}`;
        $('bar').style.width = `${score * 100}%`;
      },
    });
    await listener.start(); // in the click handler, so the AudioContext may start
    status.textContent = 'listening';
    button.textContent = 'Stop';
    button.disabled = false;
    for await (const d of listener.detections()) {
      const li = document.createElement('li');
      li.innerHTML = `<span>${d.name}</span><span>${d.confidence.toFixed(2)}</span>`;
      $('log').prepend(li);
      document.title = `detected ${d.name}`;
    }
    status.textContent = 'stopped';
  } catch (e) {
    console.error(e);
    status.textContent = `error: ${(e as Error).message}`;
  } finally {
    button.disabled = false;
    button.textContent = 'Start listening';
  }
};
