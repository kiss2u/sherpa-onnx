// sherpa-onnx/csrc/offline-tdnn-ctc-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TDNN_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TDNN_CTC_MODEL_H_
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

/** This class implements the tdnn model of the yesno recipe from icefall.
 *
 * See
 * https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn
 */
class OfflineTdnnCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineTdnnCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineTdnnCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineTdnnCtcModel() override;

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C).
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a pair containing:
   *  - log_probs: A 3-D tensor of shape (N, T', vocab_size).
   *  - log_probs_length A 1-D tensor of shape (N,). Its dtype is int64_t
   */
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value /*features_length*/) override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TDNN_CTC_MODEL_H_
