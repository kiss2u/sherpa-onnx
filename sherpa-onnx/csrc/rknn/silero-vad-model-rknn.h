// sherpa-onnx/csrc/rknn/silero-vad-model-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_SILERO_VAD_MODEL_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_SILERO_VAD_MODEL_RKNN_H_

#include "rknn_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class SileroVadModelRknn : public VadModel {
 public:
  explicit SileroVadModelRknn(const VadModelConfig &config);

  template <typename Manager>
  SileroVadModelRknn(Manager *mgr, const VadModelConfig &config);

  ~SileroVadModelRknn() override;

  // reset the internal model states
  void Reset() override;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  bool IsSpeech(const float *samples, int32_t n) override;

  // For silero vad V4, it is WindowShift().
  int32_t WindowSize() const override;

  // 512
  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_SILERO_VAD_MODEL_RKNN_H_
