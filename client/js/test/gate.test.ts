// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import { expect, it } from 'vitest';
import { DetectionGate } from '../src/gate.js';

it('fires at the threshold, once per debounce window per wake word', () => {
  const gate = new DetectionGate(0.5, 2);
  expect(gate.check({ a: 0.49 }, 0)).toBeNull();
  expect(gate.check({ a: 0.5 }, 0)).toEqual({ name: 'a', confidence: 0.5, timestamp: 0 });
  expect(gate.check({ a: 0.9 }, 1999)).toBeNull();
  expect(gate.check({ b: 0.9 }, 1999)?.name).toBe('b'); // each wake word has its own window
  expect(gate.check({ a: 0.9 }, 2000)?.name).toBe('a');
  gate.reset();
  expect(gate.check({ a: 0.9 }, 2001)?.name).toBe('a');
});

it('reports the first model that fires', () => {
  expect(new DetectionGate().check({ a: 0.1, b: 0.7, c: 0.9 }, 0)?.name).toBe('b');
});
