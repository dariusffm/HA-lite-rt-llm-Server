# Recherche — NPU-/GPU-Support in LiteRT-LM 0.17.0 auf Linux (x86_64/aarch64)

**Datum:** 2026-09-16  **Anlass:** Prüfung, ob NPU-Beschleunigung für den `litert-lm-server`-Container
auf HAOS (Linux) technisch möglich ist. Auslöser: `interfaces.py` in `litert-lm-api` 0.17.0 wirft
`RuntimeError("NPU is supported only for Intel OpenVINO on Windows. Current platform is '...'")`,
sobald `Backend.NPU()` ohne expliziten `litert_dispatch_lib_dir` instanziiert wird und `sys.platform != "win32"`.

## Zusammenfassung

Die Fehlermeldung ist korrekt, aber irreführend, wenn man sie wörtlich nimmt: **Der native LiteRT/LiteRT-LM-Stack
unterstützt Intel-NPUs (OpenVINO) inzwischen auch unter Linux** (Ubuntu 22.04/24.04, dokumentiert seit
Frühjahr/Sommer 2026). Was tatsächlich Windows-only ist, ist ausschließlich die **Auto-Detection-Bequemlichkeit
im Python-Paket `litert-lm-api`**: Das PyPI-Wheel bündelt die OpenVINO-Dispatch-Bibliothek
(`vendors/intel_openvino/dispatch/`) nur im Windows-Wheel; im Linux-Wheel (manylinux_2_27_x86_64,
geprüft für 0.17.0) fehlt dieses Verzeichnis komplett. Der Code-Pfad `litert_dispatch_lib_dir=None`
(Auto-Detect) ist daher hart auf `sys.platform == "win32"` gegated — aber der Pfad mit **explizit
gesetztem `litert_dispatch_lib_dir`** überspringt diese Prüfung vollständig und würde auf Linux
funktionieren, *wenn* man die Linux-Dispatch-Library (`LiteRtDispatch_IntelOpenvino.so`) selbst aus dem
LiteRT-LM-Quellcode baut (Bazel) — das ist in der offiziellen Doku für Linux beschrieben, aber nicht
Teil des pip-Pakets. AMD XDNA, Qualcomm Snapdragon X, Rockchip RKNPU und Hailo/AI-HAT haben **keinen**
LiteRT-LM-Support (weder nativ noch als Issue/Roadmap-Item) — dort führt kein Weg an separaten
Toolchains (RKLLM, Hailo-Runtime, ONNX Runtime EPs) vorbei. GPU (OpenCL via ML Drift) ist auf Linux
aarch64 im nativen Runtime "gesund" (laut Google-Maintainer-Antwort), aber ebenfalls nicht über die
Python-API erreichbar — nur über die CLI (`litert_lm_main --backend=gpu`).

## Befund je Frage

### 1. NPU-Support für Linux (Intel/AMD/Qualcomm/Rockchip/Hailo)

- **Intel OpenVINO NPU (Core Ultra Series 2 "Lunar Lake"/NPU4000, Series 3 "Panther Lake"/NPU5010):**
  Nativ **unterstützt auf Linux** (Ubuntu 22.04/24.04) und Windows gleichermaßen, laut offizieller
  Doku "Intel NPU (OpenVino) with LiteRT" (Google AI Edge, Stand laut Seiten-Footer 2026-06-02).
  Voraussetzung: Linux-NPU-Treiber ≥ v1.32.1 + Level-Zero-Loader ≥ v1.27.0 (ältere Treiber scheitern
  mit `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`). Die Doku beschreibt explizit das **Bauen der
  Dispatch-Library für Linux** (`LiteRtDispatch_IntelOpenvino.so`) aus dem Repo — das ist ein
  Build-Schritt, kein Pip-Install.
  Im **Python-Paket `litert-lm-api` 0.17.0** ist dieser Linux-Pfad jedoch nicht "out of the box"
  nutzbar: Das Auto-Detect in `interfaces.py::NPU.__post_init__` prüft hart `sys.platform == "win32"`
  und importiert danach das `openvino`-Pip-Paket, um `ov.Core().available_devices` abzufragen — dieser
  ganze Zweig ist auf Windows beschränkt. Der einzige dokumentierte Linux-Weg über die Python-API wäre,
  `NPU(litert_dispatch_lib_dir="/pfad/zur/selbstgebauten/.so")` **explizit** zu setzen (dieser Zweig
  umgeht die Windows-Prüfung komplett), was aber einen eigenen Bazel-Build der Dispatch-Library
  voraussetzt.
- **AMD XDNA (Ryzen AI NPU):** Keine LiteRT/LiteRT-LM-Unterstützung gefunden — weder Doku noch offene
  Roadmap-Issues. AMD-NPUs werden im Ökosystem über die **Vitis AI Execution Provider** (ONNX Runtime,
  Linux-only laut AMD-Doku) angesprochen, nicht über LiteRT.
- **Qualcomm Snapdragon X (Hexagon NPU, Windows-on-ARM):** LiteRT-LM unterstützt Qualcomm NPUs nur im
  **Android**-Kontext (SM8550/8650/8750 über QNN/AI Engine Direct); für Linux/Windows-on-ARM-Laptops
  gibt es keinen dokumentierten Pfad. Mehrere offene GitHub-Issues (`#1121`, `#1377`, `#6059`) zeigen
  zudem, dass selbst der Android-Qualcomm-NPU-Pfad in der Praxis instabil ist (QNN-Versionskonflikte,
  Registrierungsfehler, Tool-Calling-Prompts liefern "Garbage Output" ab ~580 Tokens).
- **Rockchip RKNPU (RK3588 u. a.):** Kein LiteRT-LM-Support, weder dokumentiert noch als Issue. Rockchip
  betreibt eine eigene Toolchain (`rknn-llm`/RKLLM), community-seitig gibt es `rk-llama.cpp`- und
  `rkllama`-Forks außerhalb des Google-Ökosystems.
- **Hailo (Raspberry Pi AI HAT/AI Kit):** Kein LiteRT-LM-Support gefunden. Hailo hat eine eigene
  Runtime/HailoRT; Integration erfolgt in der Praxis über separate Projekte (z. B. Frigate), nicht über
  LiteRT.

### 2. GPU-Backend auf Linux — Voraussetzungen und reale Berichte

- LiteRT nutzt für GPU den **ML-Drift-Engine** mit Backends für OpenCL, OpenGL, Metal, WebGPU — auf
  Linux relevant ist **OpenCL**. Treiber-seitig kommen damit in Frage: Intel `compute-runtime`
  (Neo/NEO, bestes Ergebnis laut Phoronix-Vergleich ggü. Mesa Rusticl auf Arc-GPUs, Faktor ~1,34×),
  Mesa **Rusticl** (`RUSTICL_ENABLE=<gallium-treiber>`, deckt Intel iris/Nvidia nouveau/AMD
  radeonsi/ARM panfrost ab, aber nicht durchgängig feature-komplett — einige OpenCL-Workloads liefen
  auf Rusticl nicht, auf Intel Compute Runtime schon) sowie NVIDIA-eigene OpenCL-ICDs. Vulkan wird von
  ML Drift nicht als primärer Linux-Pfad genannt (Fokus liegt auf OpenCL).
- **Realer Linux-Betrieb bestätigt, aber nicht über die Python-API:** GitHub-Issue
  [`google-ai-edge/LiteRT-LM#2001`](https://github.com/google-ai-edge/LiteRT-LM/issues/2001)
  (Jetson Orin NX, Ubuntu 22.04, `litert-lm-api` 0.10.1) bestätigt ausdrücklich, dass der
  **GPU-Pfad auf `linux_aarch64` funktioniert** (`litert_lm_main --backend=gpu` lief; CPU nur
  ~3 Tok/s, GPU über eigenen C++-Wrapper ~2 s/Prompt) — die Python-Bindings exponierten zum
  Issue-Zeitpunkt aber keine GPU-Backend-Auswahl. Für `x86_64`-Desktop/Server-Linux wurde kein
  spezifischer öffentlicher Benchmark-Bericht gefunden; die Architektur ist identisch, das
  begrenzende Element bleibt die Treiberverfügbarkeit (Intel/AMD/NVIDIA OpenCL-ICD im Container).
- **GPU-fähige `.litertlm`-Modelle:** Laut Doku und Beispielen sind die Standard-`.litertlm`-Exports
  (z. B. Gemma3-1B, Gemma4-E2B in den üblichen Quantisierungen) GPU-fähig — GPU ist ein
  Standard-Backend neben CPU, keine separate Modellvariante wie bei NPU/AOT nötig. Bekannte
  Einschränkung: ein offener Bug (`#2114`) zeigt, dass der Clspv-Kernel-Compiler bei bestimmten
  Adress-Space-Konvertierungen (`__global`→`__constant`) auf manchen OpenCL-Implementierungen
  (ANGLE-CL) hart scheitert — modellabhängig, nicht generell.

### 3. Alternativen im Ökosystem für NPU-Zugriff auf Linux

- **OpenVINO GenAI** (`openvinotoolkit/openvino`, aktuell 2026.3): Vollwertige, aktiv gepflegte
  NPU-Pipeline für Intel Core Ultra unter Linux — speculative decoding auf NPU, VLM-Pipelines,
  EAGLE-3, FP8-Quantisierung, MoE-Offloading. Seit 2026.0 gibt es NPU-Compiler-Integration für
  AOT- *und* On-Device-Kompilierung ohne OEM-Treiber-Abhängigkeit; seit 2026.1 ist "Compiler-in-Plugin"
  der bevorzugte Compiler-Typ. **Empfohlener First-Class-Weg für Intel-NPU auf Linux**, unabhängig von
  LiteRT-LM.
- **Intel NPU Acceleration Library** (`intel/intel-npu-acceleration-library`): **End of Life,
  Repository archiviert.** Intel verweist explizit auf OpenVINO/OpenVINO GenAI als Nachfolger — für
  neue Projekte nicht mehr relevant.
- **ONNX Runtime + OpenVINO Execution Provider:** Unterstützt CPU/GPU/NPU unter Linux, aktuell
  empfohlene OpenVINO-Version 2025.3 (Minimum 2025.0) laut EP-Doku. Reifer, gut dokumentierter Pfad.
- **ONNX Runtime + Vitis AI Execution Provider (AMD):** Linux-only, kompiliert den Modellgraph für
  Ryzen-AI-NPU/Vitis-AI-DPU — der relevante Pfad für AMD-XDNA-NPUs, da LiteRT diese nicht unterstützt.
- **llama.cpp:** Eigener, in Entwicklung befindlicher **OpenVINO-Backend** (Preview seit
  OpenVINO 2026.1, deckt Intel CPU/GPU/NPU ab). Für **Rockchip RKNPU** existieren
  Community-GGML-Backends (`rk-llama.cpp`-Forks, wissenschaftlich begleitet). Für **Hailo** wurde kein
  llama.cpp-Backend gefunden.

### 4. Braucht LiteRT-LM für NPU AOT-kompilierte Modelle?

Uneinheitlich je Vendor:

- **Google Tensor:** Nur AOT (über die CompiledModel-API), On-Device-Kompilierung ist noch nicht
  verfügbar (Beta).
- **Qualcomm AI Engine Direct, MediaTek NeuroPilot:** Beides möglich — AOT **und** On-Device-(JIT-)
  Kompilierung über die CompiledModel-API.
- **Intel OpenVINO:** Ebenfalls beides möglich (AOT **und** JIT/On-Device). Die NPU-Doku weist
  ausdrücklich darauf hin, dass der NPU-Treiber bei reinem AOT-Kompilieren (ohne Ausführung auf
  echter NPU-Hardware) übersprungen werden kann — nur zur **Ausführung** auf NPU-Hardware wird der
  Treiber zwingend gebraucht. AOT ist "highly recommended" für größere Modelle, um die
  On-Device-Initialisierungszeit zu senken, aber nicht zwingend für Intel OpenVINO.
- Es gibt **keine separaten `.litertlm`-Dateien pro NPU-Vendor** in dem Sinne, dass das Format sich
  ändert — AOT-Kompilierung erzeugt vendor-/hardware-spezifische kompilierte Artefakte, die
  eigenständig gebaut/mitgeliefert werden müssen; das ist zusätzlicher Build-Aufwand pro Zielhardware,
  kein grundsätzlich anderes Container-Format.

## Quellen

- [Intel NPU (OpenVino) with LiteRT](https://ai.google.dev/edge/litert/next/intel) — Google AI Edge, Footer-Datum 2026-06-02
- [Run LLMs using LiteRT-LM (NPU)](https://ai.google.dev/edge/litert/next/litert_lm_npu) — Google AI Edge, Footer-Datum 2026-09-02
- [NPU acceleration with LiteRT (Vendor-Übersicht)](https://ai.google.dev/edge/litert/next/npu) — Google AI Edge
- [LiteRT-LM Python API](https://ai.google.dev/edge/litert-lm/python) — Google AI Edge
- [OpenVINO™ backend for LiteRT: Optimize NPU performance on Intel® Core™ Ultra processors](https://www.intel.com/content/www/us/en/developer/articles/community/litert-unlocks-core-ultra-npu-performance-for-aipc.html) — Intel Developer Zone, Messwerte "as of May 2026"
- `litert_lm/interfaces.py`, Klasse `NPU.__post_init__` — extrahiert aus PyPI-Wheel
  `litert_lm_api-0.17.0-py3-none-manylinux_2_27_x86_64.whl` (2026-09-16 heruntergeladen und geprüft:
  kein `vendors/intel_openvino/dispatch/`-Verzeichnis im Linux-Wheel enthalten, im Gegensatz zum
  Windows-Wheel-Pfad im Quellcode)
- [Issue #2001 — GPU backend support in litert-lm-api on linux_aarch64](https://github.com/google-ai-edge/LiteRT-LM/issues/2001) — google-ai-edge/LiteRT-LM, geöffnet 2026, closed
- [Issue #2114 — GPU Engine init fails on Galaxy S26 Exynos (Clspv/MLDrift)](https://github.com/google-ai-edge/LiteRT-LM/issues/2114)
- [Issue #1121 — Unable to run LiteRT-LM on Qualcomm NPU](https://github.com/google-ai-edge/LiteRT-LM/issues/1121)
- [Issue #1377 — Unable to run LiteRT-LM on Qualcomm device](https://github.com/google-ai-edge/LiteRT-LM/issues/1377)
- [Issue #3508 — NPU prefill-chunk bug on Qualcomm/Intel NPU](https://github.com/google-ai-edge/LiteRT-LM/issues/3508)
- [litert-lm-api 0.17.0 · PyPI](https://pypi.org/project/litert-lm-api/0.17.0/) — Release 2026-09-04
- [Releases · google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/releases)
- [Intel Releases OpenVINO 2026 With Improved NPU Handling, Expanded LLM Support](https://www.phoronix.com/news/Intel-OpenVINO-2026.0-Released) — Phoronix
- [Intel Releases OpenVINO 2026.1 With Backend For Llama.cpp](https://www.phoronix.com/news/OpenVINO-2026.1-Released) — Phoronix
- [OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) — OpenVINO-Doku
- [OpenVINO Release Notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
- [GitHub - intel/intel-npu-acceleration-library](https://github.com/intel/intel-npu-acceleration-library) — Status: archiviert/EOL, verweist auf OpenVINO
- [OpenVINO™ Execution Provider for ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [Vitis AI Execution Provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html) — AMD/ONNX Runtime, Linux-only
- [Support for Intel NPU and Intel Arc GPU acceleration in llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15883) — llama.cpp Discussion #15883
- [GitHub - jorik41/rk-llama.cpp](https://github.com/jorik41/rk-llama.cpp) — Rockchip-NPU-GGML-Backend (Community)
- [Rusticl vs. Intel Compute Runtime Performance For OpenCL On Battlemage](https://www.phoronix.com/review/intel-battlemage-rusticl/4) — Phoronix
- [intel/linux-npu-driver — Releases](https://github.com/intel/linux-npu-driver/releases)

## Empfehlung für das Add-on

1. **Kein NPU-Einsatz über `litert-lm-api` in der aktuellen Form (0.17.0).** Die Fehlermeldung ist im
   Kern korrekt für den Weg, den das Add-on heute nutzt (Auto-Detect, Pip-Paket). Ein Workaround über
   manuelles Setzen von `litert_dispatch_lib_dir` würde einen eigenen Bazel-Build der
   `LiteRtDispatch_IntelOpenvino.so` für Linux im Docker-Image erfordern — das steht im Widerspruch zu
   Regel 1 der Add-on-Architektur ("bashio/Python liest nur ENV, keine Fremdformate") nicht direkt,
   aber es sprengt den aktuellen Scope (Build-Komplexität, Bazel-Toolchain im Container, nur für
   Intel-Core-Ultra-Hosts relevant — HAOS-Hardware ist überwiegend x86_64-Mini-PC oder RPi ohne NPU).
   Empfehlung: **nicht verfolgen**, bis Google die Dispatch-Library für Linux ins Pip-Wheel aufnimmt
   (kein Tracking-Issue dafür gefunden — ggf. selbst als Feature Request im LiteRT-LM-Repo einreichen).
2. **GPU (OpenCL) ist der realistischere nächste Hebel**, ist aber ebenfalls nicht über
   `litert-lm-api` erreichbar (analog zum NPU-Problem: Issue #2001 zeigt, dass der native
   `litert_lm_main`-Binary GPU auf Linux/aarch64 kann, die Python-Bindings aber nicht). Für den
   aktuellen FastAPI/uvicorn-Wrapper (der die Python-API nutzt, nicht die CLI) bringt das also aktuell
   nichts, ohne den `litert_lm_main`-Prozess direkt zu shellen oder auf eine künftige
   `litert-lm-api`-Version mit GPU-Backend-Parameter zu warten.
3. **Für AMD/Qualcomm/Rockchip/Hailo-Hosts gibt es keinen LiteRT-LM-Pfad** — falls solche Hardware in
   HA-Installationen relevant wird, wäre das ein Wechsel des Inferenz-Backends (OpenVINO GenAI für
   Intel, ONNX-Runtime-EPs für AMD/Vitis-AI, RKLLM für Rockchip, HailoRT für Hailo) und damit ein
   eigenständiges Architektur-Thema, kein Konfigurationsflag im bestehenden `litert-lm-server`.
4. **Kurzfristig bleibt CPU-Inferenz der einzige unterstützte, stabile Pfad** für dieses Add-on auf
   Linux/HAOS. Die Recherche liefert keinen Grund, den `LITERT_*`-Options-Schema jetzt um NPU/GPU-Flags
   zu erweitern — das wäre verfrühte Komplexität ohne funktionierenden Unterbau in
   `litert-lm-api` 0.17.0.
