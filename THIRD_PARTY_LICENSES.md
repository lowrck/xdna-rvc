# Third-party software

| Component | Use | License | Source |
|---|---|---|---|
| ONNX Runtime 1.30.0 (and the Ryzen AI build) | Inference runtime (downloaded at configure time, not vendored) | MIT | https://github.com/microsoft/onnxruntime |
| AMD Ryzen AI Software / VitisAI EP | NPU execution provider (installed by the user, not redistributed here) | AMD license, see Ryzen AI docs | https://ryzenai.docs.amd.com |
| AMD RyzenAI-SW `utilities/npu_check` | NPU detection logic adapted in `src/inference/npu_detect.cpp` | MIT, Copyright (c) 2023 Advanced Micro Devices, Inc. | https://github.com/amd/RyzenAI-SW |
| DirectML 1.15.4 | Windows GPU execution (downloaded at configure time) | Microsoft DirectML license | https://www.nuget.org/packages/Microsoft.AI.DirectML |
| spdlog 1.17.0 (bundles fmt) | Logging | MIT | https://github.com/gabime/spdlog |
| nlohmann/json 3.12.0 | JSON | MIT | https://github.com/nlohmann/json |
| CLI11 2.7.2 | Command line parsing | BSD-3-Clause | https://github.com/CLIUtils/CLI11 |
| miniaudio 0.11.25 | Audio device I/O and WAV codec | Public domain (Unlicense) or MIT-0 | https://github.com/mackron/miniaudio |
| Dear ImGui 1.92.9b | GUI | MIT | https://github.com/ocornut/imgui |
| GLFW 3.5.1 | GUI window/context | zlib | https://github.com/glfw/glfw |
| doctest 2.5.3 | C++ unit tests | MIT | https://github.com/doctest/doctest |
| Retrieval-based-Voice-Conversion-WebUI | Model definitions adapted in `tools/xdna_rvc_tools/rvc/` (inference subset), RMVPE network definition, realtime/SOLA and index-blending algorithms | MIT, Copyright (c) 2023 liujing04, 源文雨, Ftps | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI |

Model weights (HuBERT/ContentVec, RMVPE, user voice models) are **not** part of this
repository. Users download them separately and are responsible for complying with
their licenses.

## MIT License text (applies to the MIT components above)

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
