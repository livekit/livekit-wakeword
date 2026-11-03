// Copyright 2026 LiveKit, Inc. Licensed under the Apache License, Version 2.0.
//
// TensorFlow Lite Micro runtime for livekit-wakeword micro models. Inputs `mel` (1, 8, 40) and
// states `s0..sK`; signature output `output_0` is the score, `output_{i+1}` the new `s<i>`.
#ifndef LKWW_TFLM_H
#define LKWW_TFLM_H

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lkww_micro.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/mul.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#ifdef LKWW_PROFILE  // per-op timings (MicroProfiler, ~50 KB): define it to profile a board
#include "tensorflow/lite/micro/micro_profiler.h"
#endif
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Newer TFLite Micro moved the kernel internals (kernels/internal -> micro/kernels/internal) into
// tflite::micro; esp-tflite-micro still has the old layout. Look names up in both.
namespace lkww_tfl {
using namespace ::tflite;
#if __has_include("tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h")
using namespace ::tflite::micro;
#endif
}  // namespace lkww_tfl

// Tensor positions of CONV_2D (fixed by the schema).
namespace lkww_tfl_io {
constexpr int kInput = 0, kWeights = 1, kBias = 2, kOutput = 0;
}  // namespace lkww_tfl_io

// TFLite's fixed-point requantization (the default double rounding: gemmlowp's
// SaturatingRoundingDoublingHighMul then RoundingDivideByPOT), inline. TFLite Micro compiles its
// own copies out of line with asserts, which costs more than the arithmetic in elementwise ops.
namespace lkww_fx {
inline int32_t HighMul(int32_t a, int32_t b) {
  if (a == b && a == INT32_MIN) return INT32_MAX;
  const int64_t ab = static_cast<int64_t>(a) * b;
  const int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return static_cast<int32_t>((ab + nudge) / (int64_t{1} << 31));
}
inline int32_t DivPot(int32_t x, int e) {
  const int32_t mask = static_cast<int32_t>((int64_t{1} << e) - 1);
  const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> e) + ((x & mask) > threshold ? 1 : 0);
}
// MultiplyByQuantizedMultiplier(int32_t, ...)
inline int32_t Mul32(int32_t x, int32_t m, int shift) {
  return DivPot(HighMul(x * (1 << (shift > 0 ? shift : 0)), m), shift > 0 ? 0 : -shift);
}
// MultiplyByQuantizedMultiplierSmallerThanOneExp
inline int32_t MulSmall(int32_t x, int32_t m, int left_shift) {
  return DivPot(HighMul(x, m), -left_shift);
}
}  // namespace lkww_fx

// CONV_2D for the convolutions esp-nn is slow on. The streaming model runs many convolutions on
// one or a few output pixels (esp-nn spends more on per-call setup than on MACs there) and
// grouped convolutions with one input channel per group (no library has a fast path). Those run
// here: each output pixel's receptive field is gathered once into an int16 patch (input + offset;
// im2col, scratch in the arena), so every output channel is one contiguous dot product over
// fh * fw * fdepth values, the layout TFLite stores the weights in. Products sum in 32 bits, as in
// the reference kernel, so results are bit-identical. Everything else goes to the stock kernel.
namespace lkww_conv {
constexpr int kChunk = 256;

inline const TFLMRegistration& Base() {
  static const TFLMRegistration r = lkww_tfl::Register_CONV_2D();
  return r;
}

// The stock kernel keeps its state in node->user_data; ours wraps it.
struct Data {
  void* base;
  int patch_index;  // arena scratch for the patch, -1 for grouped convolutions (no patch)
  bool ours;        // runs here (see above)
};

// convolutions with at most this many output pixels run here
constexpr int kSmallPixels = 4;

// Swaps the stock kernel's state into node->user_data for the duration of a call.
struct BaseData {
  TfLiteNode* node;
  void* saved;
  explicit BaseData(TfLiteNode* n) : node(n), saved(n->user_data) {
    node->user_data = static_cast<Data*>(saved)->base;
  }
  ~BaseData() { node->user_data = saved; }
};

inline void* Init(TfLiteContext* c, const char* buf, size_t n) {
  auto* d = static_cast<Data*>(c->AllocatePersistentBuffer(c, sizeof(Data)));
  if (d == nullptr) return nullptr;
  d->base = Base().init(c, buf, n);
  d->patch_index = -1;
  d->ours = false;
  return d;
}

inline TfLiteStatus Prepare(TfLiteContext* c, TfLiteNode* node) {
  {
    BaseData base(node);
    TF_LITE_ENSURE_OK(c, Base().prepare(c, node));
  }
  auto* d = static_cast<Data*>(node->user_data);
  lkww_tfl::MicroContext* mc = lkww_tfl::GetMicroContext(c);
  TfLiteTensor* in = mc->AllocateTempInputTensor(node, lkww_tfl_io::kInput);
  TfLiteTensor* w = mc->AllocateTempInputTensor(node, lkww_tfl_io::kWeights);
  TfLiteTensor* out = mc->AllocateTempOutputTensor(node, lkww_tfl_io::kOutput);
  TF_LITE_ENSURE(c, in != nullptr && w != nullptr && out != nullptr);
  TfLiteTensor* b = mc->AllocateTempInputTensor(node, lkww_tfl_io::kBias);
  const int fh = w->dims->data[1], fw = w->dims->data[2], fdepth = w->dims->data[3];
  const bool grouped_1 = fdepth == 1 && in->dims->data[3] > 1;
  d->ours = in->type == kTfLiteInt8 && w->type == kTfLiteInt8 &&
            (b == nullptr || b->type == kTfLiteInt32) &&
            (out->dims->data[1] * out->dims->data[2] <= kSmallPixels || grouped_1);
  if (d->ours && !grouped_1) {
    TF_LITE_ENSURE_OK(c, c->RequestScratchBufferInArena(
                             c, fh * fw * fdepth * sizeof(int16_t), &d->patch_index));
  }
  mc->DeallocateTempTfLiteTensor(in);
  mc->DeallocateTempTfLiteTensor(w);
  mc->DeallocateTempTfLiteTensor(out);
  if (b != nullptr) mc->DeallocateTempTfLiteTensor(b);
  return kTfLiteOk;
}

// sum_k x[k] * w[k] for four weight rows `stride` apart; n <= kChunk. Kept out of line so the
// loop gets the whole register file (inlined, GCC spills inside it).
__attribute__((noinline)) inline void Dot4(const int16_t* x, const int8_t* w, int stride, int n,
                                           int32_t* s) {
  const int8_t *w0 = w, *w1 = w + stride, *w2 = w1 + stride, *w3 = w2 + stride;
  int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int k = 0;
  for (; k + 1 < n; k += 2) {
    const int32_t a = x[k], b = x[k + 1];
    s0 += a * w0[k] + b * w0[k + 1];
    s1 += a * w1[k] + b * w1[k + 1];
    s2 += a * w2[k] + b * w2[k + 1];
    s3 += a * w3[k] + b * w3[k + 1];
  }
  if (k < n) {
    const int32_t a = x[k];
    s0 += a * w0[k];
    s1 += a * w1[k];
    s2 += a * w2[k];
    s3 += a * w3[k];
  }
  s[0] = s0;
  s[1] = s1;
  s[2] = s2;
  s[3] = s3;
}

__attribute__((noinline)) inline int32_t Dot1(const int16_t* x, const int8_t* w, int n) {
  int32_t s = 0;
  for (int k = 0; k < n; ++k) s += (int32_t)x[k] * w[k];
  return s;
}

inline int8_t Requantize(int32_t acc, const lkww_tfl::OpDataConv& d, int ch, int32_t out_off) {
  int32_t v = lkww_fx::Mul32(acc, d.per_channel_output_multiplier[ch],
                             static_cast<int>(d.per_channel_output_shift[ch])) + out_off;
  v = v < d.output_activation_min ? d.output_activation_min : v;
  return static_cast<int8_t>(v > d.output_activation_max ? d.output_activation_max : v);
}

// The input offset is added while gathering, so padding is 0 exactly as the reference skips it.
inline void ConvS8(const TfLiteConvParams& p, const lkww_tfl::OpDataConv& d,
                   const lkww_tfl::RuntimeShape& is, const int8_t* in,
                   const lkww_tfl::RuntimeShape& fs, const int8_t* filt, const int32_t* bias,
                   const lkww_tfl::RuntimeShape& os, int8_t* out, int16_t* patch) {
  const int32_t in_off = -d.input_zero_point, out_off = d.output_zero_point;
  const int batches = is.Dims(0), ih = is.Dims(1), iw = is.Dims(2), idepth = is.Dims(3);
  const int fh = fs.Dims(1), fw = fs.Dims(2), fdepth = fs.Dims(3);
  const int oh = os.Dims(1), ow = os.Dims(2), odepth = os.Dims(3);
  const int groups = idepth / fdepth, per_group = odepth / groups;
  const int K = fh * fw * fdepth;  // weights per output channel
  if (fdepth == 1 && groups > 1) {  // one input channel per group: depthwise in disguise
    const int taps = fh * fw;
    for (int bb = 0; bb < batches; ++bb) {
      for (int oy = 0; oy < oh; ++oy) {
        const int y0 = oy * p.stride_height - d.padding.height;
        for (int ox = 0; ox < ow; ++ox) {
          const int x0 = ox * p.stride_width - d.padding.width;
          int8_t* o = out + ((bb * oh + oy) * ow + ox) * odepth;
          if (taps == 1 && y0 >= 0 && y0 < ih && x0 >= 0 && x0 < iw) {  // 1x1: per-channel affine
            const int8_t* xi = in + ((bb * ih + y0) * iw + x0) * idepth;
            for (int ic = 0, oc = 0; ic < idepth; ++ic) {
              const int32_t v = static_cast<int32_t>(xi[ic]) + in_off;
              for (int m = 0; m < per_group; ++m, ++oc) {
                o[oc] = Requantize(v * filt[oc] + (bias ? bias[oc] : 0), d, oc, out_off);
              }
            }
            continue;
          }
          for (int ic = 0, oc = 0; ic < idepth; ++ic) {
            for (int m = 0; m < per_group; ++m, ++oc) {
              const int8_t* wp = filt + oc * taps;
              int32_t sum = 0;
              for (int ky = 0; ky < fh; ++ky) {
                const int y = y0 + ky * p.dilation_height_factor;
                if (y < 0 || y >= ih) continue;
                const int8_t* row = in + (bb * ih + y) * iw * idepth + ic;
                for (int kx = 0; kx < fw; ++kx) {
                  const int xx = x0 + kx * p.dilation_width_factor;
                  if (xx < 0 || xx >= iw) continue;
                  sum += (static_cast<int32_t>(row[xx * idepth]) + in_off) * wp[ky * fw + kx];
                }
              }
              o[oc] = Requantize(sum + (bias ? bias[oc] : 0), d, oc, out_off);
            }
          }
        }
      }
    }
    return;
  }
  for (int b = 0; b < batches; ++b) {
    for (int oy = 0; oy < oh; ++oy) {
      const int y0 = oy * p.stride_height - d.padding.height;
      for (int ox = 0; ox < ow; ++ox) {
        const int x0 = ox * p.stride_width - d.padding.width;
        int8_t* o = out + ((b * oh + oy) * ow + ox) * odepth;
        for (int g = 0; g < groups; ++g) {
          int16_t* q = patch;
          for (int ky = 0; ky < fh; ++ky) {
            const int y = y0 + ky * p.dilation_height_factor;
            for (int kx = 0; kx < fw; ++kx, q += fdepth) {
              const int xx = x0 + kx * p.dilation_width_factor;
              if (y < 0 || y >= ih || xx < 0 || xx >= iw) {
                memset(q, 0, fdepth * sizeof(int16_t));  // zero padding
              } else {
                const int8_t* src = in + ((b * ih + y) * iw + xx) * idepth + g * fdepth;
                for (int ch = 0; ch < fdepth; ++ch) q[ch] = static_cast<int16_t>(src[ch] + in_off);
              }
            }
          }
          const int oc_end = (g + 1) * per_group;
          for (int oc = g * per_group; oc < oc_end;) {
            const int8_t* w = filt + oc * K;
            if (oc + 4 <= oc_end) {
              int32_t acc[4] = {0, 0, 0, 0};
              int32_t s[4];
              for (int k0 = 0; k0 < K; k0 += kChunk) {
                Dot4(patch + k0, w + k0, K, K - k0 < kChunk ? K - k0 : kChunk, s);
                for (int j = 0; j < 4; ++j) acc[j] += s[j];
              }
              for (int j = 0; j < 4; ++j, ++oc) {
                o[oc] = Requantize(acc[j] + (bias ? bias[oc] : 0), d, oc, out_off);
              }
            } else {
              int32_t acc = 0;
              for (int k0 = 0; k0 < K; k0 += kChunk) {
                acc += Dot1(patch + k0, w + k0, K - k0 < kChunk ? K - k0 : kChunk);
              }
              o[oc] = Requantize(acc + (bias ? bias[oc] : 0), d, oc, out_off);
              ++oc;
            }
          }
        }
      }
    }
  }
}

inline TfLiteStatus Eval(TfLiteContext* c, TfLiteNode* node) {
  const auto* data = static_cast<const Data*>(node->user_data);
  if (!data->ours) {
    BaseData base(node);
    return Base().invoke(c, node);
  }
  const TfLiteEvalTensor* in = tflite::micro::GetEvalInput(c, node, lkww_tfl_io::kInput);
  const TfLiteEvalTensor* w = tflite::micro::GetEvalInput(c, node, lkww_tfl_io::kWeights);
  const TfLiteEvalTensor* b = tflite::micro::GetEvalInput(c, node, lkww_tfl_io::kBias);
  TfLiteEvalTensor* out = tflite::micro::GetEvalOutput(c, node, lkww_tfl_io::kOutput);
  const auto& p = *static_cast<const TfLiteConvParams*>(node->builtin_data);
  // OpDataConv is the first member of the stock kernel's state (esp-nn and reference alike)
  const auto& d = *static_cast<const lkww_tfl::OpDataConv*>(data->base);
  int16_t* patch = data->patch_index < 0
      ? nullptr
      : static_cast<int16_t*>(c->GetScratchBuffer(c, data->patch_index));
  ConvS8(p, d, tflite::micro::GetTensorShape(in), tflite::micro::GetTensorData<int8_t>(in),
         tflite::micro::GetTensorShape(w), tflite::micro::GetTensorData<int8_t>(w),
         tflite::micro::GetOptionalTensorData<int32_t>(b), tflite::micro::GetTensorShape(out),
         tflite::micro::GetTensorData<int8_t>(out), patch);
  return kTfLiteOk;
}

inline void Free(TfLiteContext* c, void* buf) {
  if (buf != nullptr && Base().free != nullptr) Base().free(c, static_cast<Data*>(buf)->base);
}

inline void Reset(TfLiteContext* c, void* buf) {
  if (buf != nullptr && Base().reset != nullptr) Base().reset(c, static_cast<Data*>(buf)->base);
}

inline TFLMRegistration Registration() {
  TFLMRegistration r = Base();
  r.init = Init;
  r.free = Free;
  r.prepare = Prepare;
  r.invoke = Eval;
  r.reset = Reset;
  return r;
}
}  // namespace lkww_conv

// ADD, MUL and MEAN with fast int8 paths. The stock kernels compute a 6-D index for every
// element of a broadcast (BC-ResNet adds a (C, 1, T) tensor across frequency in every block) and
// for every element a MEAN reduces; here the shapes are merged into a few strided loops. The
// per-element arithmetic is the reference one, so results are bit-identical.
namespace lkww_ew {
constexpr int kMaxDims = 6;

// Output dims with each input's element stride (0 where it is broadcast), contiguous dims merged.
struct Walk {
  int n = 0;
  int size[kMaxDims];
  int sa[kMaxDims], sb[kMaxDims];
};

inline int Dim(const TfLiteIntArray* d, int i, int nd) {
  const int k = i - (nd - d->size);
  return k < 0 ? 1 : d->data[k];
}

inline bool MakeWalk(const TfLiteIntArray* a, const TfLiteIntArray* b, const TfLiteIntArray* o,
                     Walk* w) {
  const int nd = o->size;
  if (nd > kMaxDims || a->size > nd || b->size > nd) return false;
  int size[kMaxDims], sa[kMaxDims], sb[kMaxDims];
  int ca = 1, cb = 1;
  for (int i = nd - 1; i >= 0; --i) {
    const int od = o->data[i], ad = Dim(a, i, nd), bd = Dim(b, i, nd);
    if ((ad != od && ad != 1) || (bd != od && bd != 1)) return false;
    size[i] = od;
    sa[i] = ad == od ? ca : 0;
    sb[i] = bd == od ? cb : 0;
    ca *= ad;
    cb *= bd;
  }
  w->n = 0;
  for (int i = 0; i < nd; ++i) {  // outer to inner; merge into the previous dim when contiguous
    if (size[i] == 1) continue;
    const int m = w->n - 1;
    if (m >= 0 && w->sa[m] == sa[i] * size[i] && w->sb[m] == sb[i] * size[i]) {
      w->size[m] *= size[i];
      w->sa[m] = sa[i];
      w->sb[m] = sb[i];
    } else {
      w->size[w->n] = size[i];
      w->sa[w->n] = sa[i];
      w->sb[w->n] = sb[i];
      w->n++;
    }
  }
  if (w->n == 0) {
    w->n = 1;
    w->size[0] = 1;
    w->sa[0] = w->sb[0] = 0;
  }
  return true;
}

// f(a_index, b_index) for every output element, in order.
template <typename F>
inline void Run(const Walk& w, F f) {
  int idx[kMaxDims] = {};
  const int in = w.n - 1;
  int oa = 0, ob = 0;
  for (;;) {
    for (int i = 0, ia = oa, ib = ob; i < w.size[in]; ++i, ia += w.sa[in], ib += w.sb[in]) f(ia, ib);
    int d = in - 1;
    for (; d >= 0; --d) {  // odometer over the outer dims
      oa += w.sa[d];
      ob += w.sb[d];
      if (++idx[d] < w.size[d]) break;
      oa -= w.sa[d] * w.size[d];
      ob -= w.sb[d] * w.size[d];
      idx[d] = 0;
    }
    if (d < 0) return;
  }
}

// These kernels keep no state of their own: the stock kernel's Init / Prepare run unchanged and
// only Invoke is replaced, so the walk is rebuilt per call (a few dims; no arena cost per node).
inline bool BinaryWalk(const TfLiteEvalTensor* a, const TfLiteEvalTensor* b,
                       const TfLiteEvalTensor* o, Walk* w) {
  return o->type == kTfLiteInt8 && a->type == o->type && b->type == o->type &&
         MakeWalk(a->dims, b->dims, o->dims, w);
}

// ---- ADD ----
inline const TFLMRegistration& AddBase() {
  static const TFLMRegistration r = lkww_tfl::Register_ADD();
  return r;
}

// int8: each input's scaled value depends only on its byte, so two 256-entry tables (filled per
// call, in arena scratch) leave one requantize per element instead of three.
inline void AddRunS8(const Walk& w, const lkww_tfl::OpDataAdd& p, const int8_t* a, const int8_t* b,
                     int8_t* o, int32_t* lut) {
  const int32_t lmin = p.output_activation_min, lmax = p.output_activation_max;
  const int32_t offo = p.output_offset, mo = p.output_multiplier;
  const int so = p.output_shift;
  const int32_t ls = 1 << p.left_shift;
  int32_t* la = lut + 128;  // la[v] for v in [-128, 127]
  int32_t* lb = lut + 256 + 128;
  for (int v = -128; v < 128; ++v) {
    la[v] = lkww_fx::MulSmall((p.input1_offset + v) * ls, p.input1_multiplier, p.input1_shift);
    lb[v] = lkww_fx::MulSmall((p.input2_offset + v) * ls, p.input2_multiplier, p.input2_shift);
  }
  int k = 0;
  Run(w, [&](int ia, int ib) {
    int32_t r = lkww_fx::MulSmall(la[a[ia]] + lb[b[ib]], mo, so) + offo;
    r = r < lmin ? lmin : r > lmax ? lmax : r;
    o[k++] = static_cast<int8_t>(r);
  });
}

struct AddData {
  void* base;
  int lut_index;  // 2 x 256 int32 tables, -1 when the stock kernel runs
};

inline void* AddInit(TfLiteContext* c, const char* buf, size_t n) {
  auto* d = static_cast<AddData*>(c->AllocatePersistentBuffer(c, sizeof(AddData)));
  if (d == nullptr) return nullptr;
  d->base = AddBase().init(c, buf, n);
  d->lut_index = -1;
  return d;
}

// Runs a stock-kernel callback with its own state in node->user_data.
template <typename F>
inline auto WithBase(TfLiteNode* node, F f) -> decltype(f()) {
  void* saved = node->user_data;
  node->user_data = static_cast<AddData*>(saved)->base;
  auto r = f();
  node->user_data = saved;
  return r;
}

inline TfLiteStatus AddPrepare(TfLiteContext* c, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(c, WithBase(node, [&] { return AddBase().prepare(c, node); }));
  auto* d = static_cast<AddData*>(node->user_data);
  lkww_tfl::MicroContext* mc = lkww_tfl::GetMicroContext(c);
  TfLiteTensor* o = mc->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(c, o != nullptr);
  if (o->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(c, c->RequestScratchBufferInArena(c, 512 * sizeof(int32_t), &d->lut_index));
  }
  mc->DeallocateTempTfLiteTensor(o);
  return kTfLiteOk;
}

inline void AddFree(TfLiteContext* c, void* buf) {
  if (buf != nullptr && AddBase().free != nullptr) AddBase().free(c, static_cast<AddData*>(buf)->base);
}

inline void AddReset(TfLiteContext* c, void* buf) {
  if (buf != nullptr && AddBase().reset != nullptr) AddBase().reset(c, static_cast<AddData*>(buf)->base);
}

inline TfLiteStatus AddEval(TfLiteContext* c, TfLiteNode* node) {
  const TfLiteEvalTensor* a = tflite::micro::GetEvalInput(c, node, 0);
  const TfLiteEvalTensor* b = tflite::micro::GetEvalInput(c, node, 1);
  TfLiteEvalTensor* o = tflite::micro::GetEvalOutput(c, node, 0);
  const auto* d = static_cast<const AddData*>(node->user_data);
  Walk w;
  if (d->lut_index < 0 || !BinaryWalk(a, b, o, &w)) {
    return WithBase(node, [&] { return AddBase().invoke(c, node); });
  }
  const auto& p = *static_cast<const lkww_tfl::OpDataAdd*>(d->base);
  AddRunS8(w, p, tflite::micro::GetTensorData<int8_t>(a), tflite::micro::GetTensorData<int8_t>(b),
           tflite::micro::GetTensorData<int8_t>(o),
           static_cast<int32_t*>(c->GetScratchBuffer(c, d->lut_index)));
  return kTfLiteOk;
}

inline TFLMRegistration AddRegistration() {
  TFLMRegistration r = AddBase();
  r.init = AddInit;
  r.free = AddFree;
  r.prepare = AddPrepare;
  r.invoke = AddEval;
  r.reset = AddReset;
  return r;
}

// ---- MUL ----
inline const TFLMRegistration& MulBase() {
  static const TFLMRegistration r = lkww_tfl::Register_MUL();
  return r;
}

inline void MulRun(const Walk& w, const lkww_tfl::OpDataMul& p, const int8_t* a, const int8_t* b,
                   int8_t* o) {
  const int32_t lmin = p.output_activation_min, lmax = p.output_activation_max;
  const int32_t oa = -p.input1_zero_point, ob = -p.input2_zero_point, oz = p.output_zero_point;
  const int32_t m = p.output_multiplier;
  const int sh = p.output_shift;
  int k = 0;
  Run(w, [&](int ia, int ib) {
    int32_t r = oz + lkww_fx::Mul32((oa + a[ia]) * (ob + b[ib]), m, sh);
    r = r < lmin ? lmin : r > lmax ? lmax : r;
    o[k++] = static_cast<int8_t>(r);
  });
}

inline TfLiteStatus MulEval(TfLiteContext* c, TfLiteNode* node) {
  const TfLiteEvalTensor* a = tflite::micro::GetEvalInput(c, node, 0);
  const TfLiteEvalTensor* b = tflite::micro::GetEvalInput(c, node, 1);
  TfLiteEvalTensor* o = tflite::micro::GetEvalOutput(c, node, 0);
  Walk w;
  if (!BinaryWalk(a, b, o, &w)) return MulBase().invoke(c, node);
  const auto& p = *static_cast<const lkww_tfl::OpDataMul*>(node->user_data);
  MulRun(w, p, tflite::micro::GetTensorData<int8_t>(a), tflite::micro::GetTensorData<int8_t>(b),
         tflite::micro::GetTensorData<int8_t>(o));
  return kTfLiteOk;
}

// ---- MEAN over one contiguous block of axes ----
inline const TFLMRegistration& MeanBase() {
  static const TFLMRegistration r = lkww_tfl::Register_MEAN();
  return r;
}

// outer x reduce x inner view of the input; false if the axes are not one contiguous block
inline bool MeanShape(const TfLiteEvalTensor* in, const TfLiteEvalTensor* axis, int* outer,
                      int* reduce, int* inner) {
  const int nd = in->dims->size;
  if (nd > kMaxDims || axis->type != kTfLiteInt32) return false;
  bool sel[kMaxDims] = {};
  int na = 1;
  for (int i = 0; i < axis->dims->size; ++i) na *= axis->dims->data[i];
  const int32_t* ax = tflite::micro::GetTensorData<int32_t>(axis);
  for (int i = 0; i < na; ++i) {
    const int a = ax[i] < 0 ? ax[i] + nd : ax[i];
    if (a < 0 || a >= nd) return false;
    sel[a] = true;
  }
  int first = -1, last = -1;
  for (int i = 0; i < nd; ++i) {
    if (!sel[i]) continue;
    if (first >= 0 && last != i - 1) return false;
    if (first < 0) first = i;
    last = i;
  }
  if (first < 0) return false;
  *outer = *reduce = *inner = 1;
  for (int i = 0; i < nd; ++i) (i < first ? *outer : i <= last ? *reduce : *inner) *= in->dims->data[i];
  return true;
}

inline void MeanRun(const lkww_tfl::OpDataReduce& p, int outer, int R, int inner, const int8_t* in,
                    int8_t* out, int32_t* sum) {
  // same rescale as reference_ops::QuantizedMeanOrSum (mean of R elements)
  int shift = 63 - lkww_tfl::CountLeadingZeros(static_cast<uint64_t>(R));
  shift = shift < 32 ? shift : 32;
  shift = shift < 31 + p.shift ? shift : 31 + p.shift;
  const int32_t mult = static_cast<int32_t>((static_cast<int64_t>(p.multiplier) << shift) / R);
  const int out_shift = p.shift - shift;
  for (int o = 0; o < outer; ++o) {
    for (int i = 0; i < inner; ++i) sum[i] = 0;
    const int8_t* x = in + o * R * inner;
    for (int r = 0; r < R; ++r, x += inner) {
      for (int i = 0; i < inner; ++i) sum[i] += x[i];
    }
    for (int i = 0; i < inner; ++i) {
      const int32_t shifted = static_cast<int32_t>(sum[i] - static_cast<int64_t>(p.input_zp) * R);
      int32_t v = lkww_fx::Mul32(shifted, mult, out_shift) + p.output_zp;
      v = v < -128 ? -128 : v > 127 ? 127 : v;
      out[o * inner + i] = static_cast<int8_t>(v);
    }
  }
}

inline TfLiteStatus MeanEval(TfLiteContext* c, TfLiteNode* node) {
  const TfLiteEvalTensor* in = tflite::micro::GetEvalInput(c, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(c, node, 1);
  TfLiteEvalTensor* out = tflite::micro::GetEvalOutput(c, node, 0);
  int outer, R, inner;
  if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8 || !MeanShape(in, axis, &outer, &R, &inner)) {
    return MeanBase().invoke(c, node);
  }
  const auto& p = *static_cast<const lkww_tfl::OpDataReduce*>(node->user_data);
  // the stock kernel's accumulator scratch: one int32 per output, and inner <= outputs
  auto* sum = static_cast<int32_t*>(c->GetScratchBuffer(c, p.scratch_accumulator_idx));
  MeanRun(p, outer, R, inner, tflite::micro::GetTensorData<int8_t>(in),
          tflite::micro::GetTensorData<int8_t>(out), sum);
  return kTfLiteOk;
}

template <const TFLMRegistration& (*BaseReg)()>
inline TFLMRegistration Registration(TfLiteStatus (*invoke)(TfLiteContext*, TfLiteNode*)) {
  TFLMRegistration r = BaseReg();
  r.invoke = invoke;
  return r;
}
}  // namespace lkww_ew

class LkwwDetector {
 public:
  static constexpr int kMaxStates = 32;

  // `warm`: run the 2 s silence warm-up now (false: only load, e.g. to measure the arena)
  LkwwDetector(const void* model_data, uint8_t* arena, size_t arena_bytes, bool warm = true)
      : model_(tflite::GetModel(model_data)),
#ifdef LKWW_PROFILE
        interpreter_(model_, Resolver(), arena, arena_bytes, nullptr, &prof_) {
#else
        interpreter_(model_, Resolver(), arena, arena_bytes) {
#endif
    if (interpreter_.AllocateTensors() != kTfLiteOk) return;
    ok_ = MapTensors() && PlanStateCopy();
    if (warm) Reset();
  }

  ~LkwwDetector() { free(temp_); }
  LkwwDetector(const LkwwDetector&) = delete;
  LkwwDetector& operator=(const LkwwDetector&) = delete;

  bool ok() const { return ok_; }
  size_t arena_used() { return interpreter_.arena_used_bytes(); }
  int num_states() const { return n_states_; }
  size_t state_bytes(int i) const { return state_[i]->bytes; }
  bool uses_scratch() const { return temp_ != nullptr; }  // bytes: temp_bytes()
  size_t temp_bytes() const { return temp_bytes_; }

  // Forget everything heard: zero state, then 2 s of silence (what the model was trained on).
  void Reset() {
    lkww_frontend_reset(&fe_);
    if (!ok_) return;
    for (int i = 0; i < n_states_; i++) ZeroState(state_[i]);
    static const int16_t silence[LKWW_CHUNK] = {0};
    for (int i = 0; i < kWarmupSteps; i++) Process(silence);
  }

  // Consume LKWW_CHUNK samples; returns the wake-word probability, or -1 on error.
  float Process(const int16_t* pcm) {
    if (!ok_) return -1.0f;
    lkww_frontend_process(&fe_, pcm, mel_buf_);
    Quantize(mel_buf_, mel_);
    return Step();
  }

  // Same, from 8 x 40 log-mel frames.
  float ProcessMel(const float* mel) {
    if (!ok_) return -1.0f;
    Quantize(mel, mel_);
    return Step();
  }

 private:
  static constexpr int kWarmupSteps = 25;  // 2 s

  float Step() {
#ifdef LKWW_PROFILE
    prof_.ClearEvents();
#endif
    if (interpreter_.Invoke() != kTfLiteOk) return -1.0f;
#ifdef LKWW_PROFILE
    if (++prof_n_ == 40) {  // one step after the warm-up, per op type
      MicroPrintf("PROFILE total ticks %u arena_used %u", (unsigned)prof_.GetTotalTicks(),
                  (unsigned)interpreter_.arena_used_bytes());
      prof_.LogTicksPerTagCsv();
    }
#endif
    const float score = Dequantize(score_, 0);
    for (int k = 0; k < n_copies_; k++) {
      const Copy& c = copies_[k];
      if (c.to_temp) {
        memcpy(temp_ + c.offset, state_out_[c.i]->data.raw, state_out_[c.i]->bytes);
      } else {
        CopyState(c.i, c.from_temp ? static_cast<const void*>(temp_ + c.offset)
                                   : state_out_[c.i]->data.raw);
      }
    }
    return score;
  }

  // ---- state hand-over: s<i>_out -> s<i> after every step ----
  // TFLite Micro reuses arena memory, so an output state can sit on top of another input state.
  // PlanStateCopy orders the copies so nothing is overwritten before it is read; a cycle goes
  // through one temporary buffer of the largest state (instead of a copy of all of them).
  struct Copy {
    uint8_t i;
    bool to_temp, from_temp;
    uint32_t offset;  // in temp_
  };

  bool PlanStateCopy() {
    bool done[kMaxStates] = {}, parked[kMaxStates] = {};
    uint32_t offset[kMaxStates] = {};
    int left = n_states_;
    while (left > 0) {
      int pick = -1;
      for (int i = 0; i < n_states_ && pick < 0; i++) {
        if (done[i]) continue;
        bool safe = true;  // writing s<i> must not clobber an output still to be read
        for (int j = 0; j < n_states_ && safe; j++) {
          if (!done[j] && !parked[j] && j != i && Overlap(state_[i], state_out_[j])) safe = false;
        }
        // an element-wise requantize cannot run in place over a shifted copy of itself
        if (safe && !parked[i] && !SameLayout(i) && Overlap(state_[i], state_out_[i]) &&
            state_[i]->data.raw != state_out_[i]->data.raw) {
          safe = false;
        }
        if (safe) pick = i;
      }
      if (pick < 0) {  // a cycle: park the smallest pending output in temp_
        int k = -1;
        for (int i = 0; i < n_states_; i++) {
          if (!done[i] && !parked[i] && (k < 0 || state_out_[i]->bytes < state_out_[k]->bytes)) {
            k = i;
          }
        }
        if (k < 0) return false;
        parked[k] = true;
        offset[k] = static_cast<uint32_t>(temp_bytes_);
        temp_bytes_ += (state_out_[k]->bytes + 3) & ~size_t{3};
        copies_[n_copies_++] = {static_cast<uint8_t>(k), true, false, offset[k]};
        continue;
      }
      copies_[n_copies_++] = {static_cast<uint8_t>(pick), false, parked[pick], offset[pick]};
      done[pick] = true;
      left--;
    }
    if (temp_bytes_ > 0) {
      temp_ = static_cast<uint8_t*>(malloc(temp_bytes_));
      if (temp_ == nullptr) return false;
    }
    return true;
  }

  bool SameLayout(int i) const {
    const TfLiteTensor *a = state_out_[i], *b = state_[i];
    return a->params.scale == b->params.scale && a->params.zero_point == b->params.zero_point;
  }

  // s<i>_out (its bytes at `src`) -> s<i>: a copy, or a requantize when the converter gave the
  // two different scales. Same arithmetic as the host runtime (MicroRuntime): dequantize in
  // float32, divide by the input scale, round half to even.
  void CopyState(int i, const void* src) {
    TfLiteTensor* to = state_[i];
    if (SameLayout(i)) {
      memmove(to->data.raw, src, to->bytes);
      return;
    }
    const TfLiteQuantizationParams& q = state_out_[i]->params;
    for (size_t k = 0; k < to->bytes; k++) Store(to, k, Load(static_cast<const int8_t*>(src)[k], q));
  }

  static float Load(int8_t v, const TfLiteQuantizationParams& q) {
    return (static_cast<float>(v) - q.zero_point) * q.scale;
  }

  static void Store(TfLiteTensor* t, size_t k, float x) {
    const float q = nearbyintf(x / t->params.scale) + t->params.zero_point;
    t->data.int8[k] = static_cast<int8_t>(q < -128.f ? -128.f : q > 127.f ? 127.f : q);
  }

  static void Quantize(const float* x, TfLiteTensor* t) {
    for (size_t k = 0; k < t->bytes; k++) Store(t, k, x[k]);
  }

  static float Dequantize(const TfLiteTensor* t, int k) { return Load(t->data.int8[k], t->params); }

  // Zero in the tensor's real-valued domain: the zero point.
  static void ZeroState(TfLiteTensor* t) {
    memset(t->data.raw, static_cast<int8_t>(t->params.zero_point), t->bytes);
  }
  static bool Overlap(const TfLiteTensor* a, const TfLiteTensor* b) {
    const uint8_t* a0 = static_cast<const uint8_t*>(a->data.data);
    const uint8_t* b0 = static_cast<const uint8_t*>(b->data.data);
    return a0 < b0 + b->bytes && b0 < a0 + a->bytes;
  }

  static int Position(const flatbuffers::Vector<int32_t>* list, int32_t tensor_index) {
    for (uint32_t i = 0; i < list->size(); i++) {
      if (list->Get(i) == tensor_index) return static_cast<int>(i);
    }
    return -1;
  }

  bool MapTensors() {
    const auto* sigs = model_->signature_defs();
    const auto* sub = model_->subgraphs()->Get(0);
    if (sigs == nullptr || sigs->size() == 0) return false;
    const auto* sig = sigs->Get(0);
    for (uint32_t i = 0; i < sig->inputs()->size(); i++) {
      const auto* t = sig->inputs()->Get(i);
      const char* name = t->name()->c_str();
      int pos = Position(sub->inputs(), static_cast<int32_t>(t->tensor_index()));
      if (pos < 0) return false;
      if (strcmp(name, "mel") == 0) {
        mel_ = interpreter_.input(pos);
      } else if (name[0] == 's') {
        int k = atoi(name + 1);
        if (k < 0 || k >= kMaxStates) return false;
        state_[k] = interpreter_.input(pos);
        if (k + 1 > n_states_) n_states_ = k + 1;
      }
    }
    for (uint32_t i = 0; i < sig->outputs()->size(); i++) {
      const auto* t = sig->outputs()->Get(i);
      const flatbuffers::String* name = t->name();
      int pos = Position(sub->outputs(), static_cast<int32_t>(t->tensor_index()));
      if (name == nullptr || pos < 0) return false;
      int k = -1;
      for (int j = static_cast<int>(name->size()) - 1; j >= 0; j--) {
        if (name->c_str()[j] == '_') {
          k = atoi(name->c_str() + j + 1);
          break;
        }
      }
      if (k < 0) return false;
      if (k == 0) {
        score_ = interpreter_.output(pos);
      } else if (k - 1 < kMaxStates) {
        state_out_[k - 1] = interpreter_.output(pos);
      }
    }
    // int8 everywhere (livekit-wakeword's micro export); anything else is not a model for us
    if (mel_ == nullptr || score_ == nullptr || mel_->type != kTfLiteInt8 ||
        score_->type != kTfLiteInt8 || mel_->bytes != LKWW_FRAMES * LKWW_N_MELS) {
      MicroPrintf("livekit_wakeword: not an int8 micro model (re-export it with this version)");
      return false;
    }
    for (int i = 0; i < n_states_; i++) {
      if (state_[i] == nullptr || state_out_[i] == nullptr) return false;
      if (state_[i]->type != kTfLiteInt8 || state_out_[i]->type != kTfLiteInt8 ||
          state_[i]->bytes != state_out_[i]->bytes) {
        return false;
      }
    }
    return true;
  }

  static tflite::MicroMutableOpResolver<20>& Resolver() {
    static tflite::MicroMutableOpResolver<20> r;
    static bool init = false;
    if (!init) {
#ifdef LKWW_REFERENCE_KERNELS  // stock TFLite Micro kernels, for checking ours against
      r.AddConv2D(); r.AddAdd(); r.AddMul(); r.AddMean();
#else
      r.AddConv2D(lkww_conv::Registration());
      r.AddAdd(lkww_ew::AddRegistration());
      r.AddMul(lkww_ew::Registration<lkww_ew::MulBase>(lkww_ew::MulEval));
      r.AddMean(lkww_ew::Registration<lkww_ew::MeanBase>(lkww_ew::MeanEval));
#endif
      r.AddDepthwiseConv2D(); r.AddFullyConnected();
      r.AddConcatenation(); r.AddDequantize(); r.AddLogistic(); r.AddPad(); r.AddQuantize();
      r.AddReshape(); r.AddStridedSlice(); r.AddSub(); r.AddTranspose(); r.AddRelu();
      r.AddSlice(); r.AddSplit(); r.AddRsqrt(); r.AddSquaredDifference();
      init = true;
    }
    return r;
  }

  const tflite::Model* model_;
#ifdef LKWW_PROFILE
  tflite::MicroProfiler prof_;
  int prof_n_ = 0;
#endif
  tflite::MicroInterpreter interpreter_;
  lkww_frontend_t fe_ = {};  // zero: the LKWW_MEL_* log-mel defaults
  float mel_buf_[LKWW_FRAMES * LKWW_N_MELS];  // front end output (kept off the task stack)
  TfLiteTensor* mel_ = nullptr;
  TfLiteTensor* score_ = nullptr;
  TfLiteTensor* state_[kMaxStates] = {};
  TfLiteTensor* state_out_[kMaxStates] = {};
  int n_states_ = 0;
  Copy copies_[2 * kMaxStates] = {};
  int n_copies_ = 0;
  uint8_t* temp_ = nullptr;
  size_t temp_bytes_ = 0;
  bool ok_ = false;
};

#endif  // LKWW_TFLM_H
