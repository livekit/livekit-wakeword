// host_test model.tflite mel.f32 > scores.txt   (mel.f32: N x 8 x 40 float32, row-major)
#include <cstdio>
#include <vector>

#include "lkww_tflm.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::vector<char> model;
  FILE* f = fopen(argv[1], "rb");
  for (int c; (c = fgetc(f)) != EOF;) model.push_back(static_cast<char>(c));
  fclose(f);
  static uint8_t arena[256 * 1024];
  LkwwDetector det(model.data(), arena, sizeof(arena));
  if (!det.ok()) { fprintf(stderr, "init failed\n"); return 1; }
  fprintf(stderr, "arena used %zu bytes, %d states:", det.arena_used(), det.num_states());
  size_t total = 0;
  for (int i = 0; i < det.num_states(); i++) total += det.state_bytes(i);
  fprintf(stderr, " %zu bytes of state, %zu byte copy buffer\n", total, det.temp_bytes());
  FILE* m = fopen(argv[2], "rb");
  float mel[LKWW_FRAMES * LKWW_N_MELS];
  while (fread(mel, sizeof(float), LKWW_FRAMES * LKWW_N_MELS, m) == LKWW_FRAMES * LKWW_N_MELS) {
    printf("%.6f\n", det.ProcessMel(mel));
  }
  fclose(m);
  return 0;
}
