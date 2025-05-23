// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline SenseVoice model,
// i.e., non-streaming SenseVoice model
// to decode files with homophone replacer.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileSenseVoiceWithHr {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html
    // to download model files
    String model = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx";
    String tokens = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt";

    String waveFilename = "./test-hr.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineSenseVoiceModelConfig senseVoice =
        OfflineSenseVoiceModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setSenseVoice(senseVoice)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    HomophoneReplacerConfig hr =
        HomophoneReplacerConfig.builder()
            .setDictDir("./dict")
            .setLexicon("./lexicon.txt")
            .setRuleFsts("./replace.fst")
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .setHr(hr)
            .build();

    OfflineRecognizer recognizer = new OfflineRecognizer(config);
    OfflineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    recognizer.decode(stream);

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
