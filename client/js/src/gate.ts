// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

export interface Detection {
  /** The model that fired. */
  name: string;
  /** Its score, at or above the threshold. */
  confidence: number;
  /** When it fired, in milliseconds (`performance.now()` unless a clock was passed). */
  timestamp: number;
}

/**
 * Turns scores into detections: a model fires when its score reaches `threshold`, at most once
 * per `debounce` seconds. The same rule as the Python and Swift listeners.
 */
export class DetectionGate {
  private readonly last = new Map<string, number>();

  constructor(
    readonly threshold = 0.5,
    readonly debounce = 2.0,
  ) {}

  /** The first model in `scores` that fires at time `now` (ms), if any. */
  check(scores: Record<string, number>, now: number): Detection | null {
    for (const [name, score] of Object.entries(scores)) {
      const last = this.last.get(name);
      if (score >= this.threshold && (last === undefined || now - last >= this.debounce * 1000)) {
        this.last.set(name, now);
        return { name, confidence: score, timestamp: now };
      }
    }
    return null;
  }

  reset(): void {
    this.last.clear();
  }
}
