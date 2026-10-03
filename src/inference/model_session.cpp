#include "inference/model_session.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "inference/ort_runtime.h"
#include "util/error.h"
#include "util/log.h"

namespace xr {

namespace {

std::basic_string<ORTCHAR_T> to_ort_path(const std::filesystem::path& p) {
#if defined(_WIN32)
    return p.wstring();
#else
    return p.string();
#endif
}

std::vector<TensorInfo> describe(const Ort::Session& session, bool inputs) {
    std::vector<TensorInfo> out;
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n = inputs ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < n; ++i) {
        TensorInfo t;
        auto name = inputs ? session.GetInputNameAllocated(i, alloc) : session.GetOutputNameAllocated(i, alloc);
        t.name = name.get();
        auto type_info = inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto tinfo = type_info.GetTensorTypeAndShapeInfo();
            t.type = tinfo.GetElementType();
            t.shape = tinfo.GetShape();
            const size_t rank = tinfo.GetDimensionsCount();
            std::vector<const char*> names(rank, nullptr);
            tinfo.GetSymbolicDimensions(names.data(), rank);
            for (const char* n : names) t.dim_names.emplace_back(n ? n : "");
        }
        out.push_back(std::move(t));
    }
    return out;
}

size_t element_size(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
        default: throw std::runtime_error(std::string("unsupported tensor element type ") + element_type_name(type));
    }
}

}  // namespace

const char* element_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        default: return "other";
    }
}

bool TensorInfo::is_static() const {
    for (auto d : shape) {
        if (d < 0) return false;
    }
    return true;
}

std::string TensorInfo::to_string() const {
    std::vector<std::string> dims;
    for (auto d : shape) dims.push_back(d < 0 ? "?" : std::to_string(d));
    return fmt::format("{}: {}[{}]", name, element_type_name(type), fmt::join(dims, ", "));
}

Ort::Value make_zero_tensor(ONNXTensorElementDataType type, const std::vector<int64_t>& shape) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::Value v = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(), type);
    const size_t count = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                         [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
    auto* data = static_cast<uint8_t*>(v.GetTensorMutableRawData());
    std::fill(data, data + count * element_size(type), uint8_t{0});
    return v;
}

ModelSession::ModelSession(Ort::Session session, SessionReport report, bool profiling_active)
    : session_(std::move(session)), report_(std::move(report)), profiling_active_(profiling_active) {
    inputs_ = describe(session_, true);
    outputs_ = describe(session_, false);
    for (const auto& t : inputs_) input_name_storage_.push_back(t.name);
    for (const auto& t : outputs_) output_name_storage_.push_back(t.name);
    for (const auto& s : input_name_storage_) input_names_.push_back(s.c_str());
    for (const auto& s : output_name_storage_) output_names_.push_back(s.c_str());
}

std::filesystem::path materialize_static_model(OrtRuntime& runtime, const std::filesystem::path& src,
                                              const std::map<std::string, int64_t>& dims,
                                              const std::filesystem::path& cache_dir) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(src, ec);
    const auto mtime = std::filesystem::last_write_time(src, ec).time_since_epoch().count();
    std::string key = src.stem().string();
    for (const auto& [d, v] : dims) key += fmt::format("__{}{}", d, v);
    key += fmt::format("__{:x}", std::hash<std::string>{}(fmt::format("{}|{}|{}", src.string(), size, mtime)));
    const auto out = cache_dir / (key + ".onnx");
    if (std::filesystem::exists(out)) return out;
    std::filesystem::create_directories(cache_dir);
    const auto tmp = cache_dir / (key + ".onnx.tmp");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);  // standard ONNX ops only
    for (const auto& [d, v] : dims) so.AddFreeDimensionOverrideByName(d.c_str(), v);
    so.SetOptimizedModelFilePath(to_ort_path(tmp).c_str());
    const auto t0 = std::chrono::steady_clock::now();
    { Ort::Session s(runtime.env(), to_ort_path(src).c_str(), so); }
    std::filesystem::rename(tmp, out);
    XR_LOG_INFO("wrote static-shape model {} in {:.0f} ms", out.string(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return out;
}

std::unique_ptr<ModelSession> ModelSession::open(OrtRuntime& runtime, const SessionRequest& request,
                                             const OpenOptions& options) {
    if (!std::filesystem::exists(request.model_path)) {
        throw UserError("[" + request.stage + "] model file not found: " + request.model_path.string(),
                        "Re-import the voice model or check model.json paths.");
    }

    std::vector<BackendKind> chain;
    if (options.backend == BackendKind::Auto) {
        chain = automatic_preference();
    } else {
        chain.push_back(options.backend);
        if (options.allow_cpu_fallback && options.backend != BackendKind::CPU) chain.push_back(BackendKind::CPU);
    }

    SessionReport report;
    report.stage = request.stage;
    report.model_path = request.model_path;
    report.requested = options.backend;

    std::string first_failure;
    bool backend_failed = false;  // an available backend failed (as opposed to being unavailable)
    for (size_t attempt = 0; attempt < chain.size(); ++attempt) {
        const BackendKind kind = chain[attempt];
        auto backend = make_backend(kind);
        if (!backend) {
            const std::string why = std::string(display_name(kind)) + ": no implementation in this build";
            report.attempts.push_back(why);
            if (first_failure.empty()) first_failure = why;
            continue;
        }
        const auto avail = backend->availability(runtime);
        if (!avail.available) {
            const std::string why = std::string(display_name(kind)) + " unavailable: " + avail.reason;
            report.attempts.push_back(why);
            XR_LOG_INFO("[{}] {}", request.stage, why);
            if (first_failure.empty()) first_failure = why;
            continue;
        }

        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (runtime.global_threads() > 0) {
                so.DisablePerSessionThreads();
            } else {
                if (request.intra_op_threads > 0) so.SetIntraOpNumThreads(request.intra_op_threads);
                so.SetInterOpNumThreads(1);
            }
            so.SetLogSeverityLevel(static_cast<int>(request.session_log_level));
            so.SetLogId(request.stage.c_str());
            if (request.disallow_cpu_ep_fallback && kind != BackendKind::CPU) {
                so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            }
            const bool profiling = request.collect_evidence || request.keep_profiling;
            if (profiling) {
                std::filesystem::create_directories(request.diagnostics_dir);
                const auto prefix = request.diagnostics_dir / ("ort_profile_" + request.stage + "_" +
                                                               std::string(to_string(kind)));
                so.EnableProfiling(to_ort_path(prefix).c_str());
            }
            for (const auto& [dim, value] : request.free_dims) so.AddFreeDimensionOverrideByName(dim.c_str(), value);
            SessionRequest effective = request;
            if (kind == BackendKind::XDNA2 && !request.free_dims.empty()) {
                effective.model_path = materialize_static_model(runtime, request.model_path, request.free_dims,
                                                                request.static_model_dir);
                report.attempts.push_back("static-shape model: " + effective.model_path.string());
            }
            backend->configure(so, effective);
            backend->before_create(effective, report);

            const auto t0 = std::chrono::steady_clock::now();
            Ort::Session session(runtime.env(), to_ort_path(effective.model_path).c_str(), so);
            const auto t1 = std::chrono::steady_clock::now();

            report.create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            report.effective = kind;
            report.attempts.push_back(std::string(display_name(kind)) +
                                      fmt::format(": session created in {:.1f} ms", report.create_ms));
            // Automatic mode skipping backends that are not installed is a selection, not a
            // fallback. Anything else that did not run where it was asked to is a fallback.
            const bool fallback = attempt > 0 && (options.backend != BackendKind::Auto || backend_failed);
            if (fallback) {
                report.fell_back = true;
                report.fallback_reason = first_failure;
                XR_LOG_WARN("[{}] FALLBACK: requested {}, running on {}. Reason: {}", request.stage,
                            display_name(options.backend), display_name(kind), first_failure);
            } else if (attempt > 0) {
                XR_LOG_INFO("[{}] Automatic backend selection chose {} ({})", request.stage, display_name(kind),
                            first_failure);
            }
            backend->after_create(effective, report);

            std::unique_ptr<ModelSession> s(new ModelSession(std::move(session), std::move(report), profiling));
            s->free_dims_ = request.free_dims;
            if (request.collect_evidence) s->collect_evidence(options, request);
            XR_LOG_INFO("[{}] {}", request.stage, s->report_.summary());
            return s;
        } catch (const Ort::Exception& e) {
            const std::string why = std::string(display_name(kind)) + " session creation failed: " + e.what();
            report.attempts.push_back(why);
            XR_LOG_WARN("[{}] {}", request.stage, why);
            if (!backend_failed) first_failure = why;  // a real failure explains more than "unavailable"
            backend_failed = true;
        } catch (const UserError& e) {
            const std::string why = std::string(display_name(kind)) + ": " + e.what();
            report.attempts.push_back(why);
            XR_LOG_WARN("[{}] {}", request.stage, why);
            if (!backend_failed) first_failure = why;
            backend_failed = true;
        }
    }

    std::string hint;
    if (options.backend == BackendKind::XDNA2) {
        hint = "Retry with --backend cpu (or enable CPU fallback), run `xdna-rvc-cli providers` to check the NPU "
               "runtime, or run tools/inspect_execution.py on the model to find unsupported operators.";
    } else if (options.backend != BackendKind::CPU) {
        hint = "Retry with --backend cpu or enable CPU fallback.";
    }
    std::string attempts;
    for (const auto& a : report.attempts) attempts += "\n  - " + a;
    throw UserError("[" + request.stage + "] could not create an inference session for " +
                        request.model_path.string() + " with backend " + std::string(display_name(options.backend)) +
                        ". Attempts:" + attempts,
                    hint);
}

void ModelSession::collect_evidence(const OpenOptions& options, const SessionRequest& request) {
    try {
        std::vector<Ort::Value> probe_inputs;
        for (const auto& in : inputs_) {
            std::vector<int64_t> shape = in.shape;
            for (size_t d = 0; d < shape.size(); ++d) {
                if (shape[d] >= 0 || d >= in.dim_names.size()) continue;
                const auto it = free_dims_.find(in.dim_names[d]);
                if (it != free_dims_.end()) shape[d] = it->second;
            }
            const bool resolved = std::all_of(shape.begin(), shape.end(), [](int64_t v) { return v >= 0; });
            if (!resolved) {
                const auto it = options.probe_shapes.find(in.name);
                if (it == options.probe_shapes.end()) {
                    throw std::runtime_error("input '" + in.name + "' has dynamic shape and no probe shape was given");
                }
                shape = it->second;
            }
            probe_inputs.push_back(make_zero_tensor(in.type, shape));
        }
        (void)run(probe_inputs);
    } catch (const std::exception& e) {
        report_.evidence_error = std::string("probe run failed: ") + e.what();
    }
    if (profiling_active_ && !request.keep_profiling) {
        const std::string file = end_profiling();
        if (!file.empty()) {
            report_.profile_file = file;
            try {
                report_.provider_usage = parse_ort_profile(file);
            } catch (const std::exception& e) {
                report_.evidence_error = e.what();
            }
        }
    }
}

std::string ModelSession::end_profiling() {
    if (!profiling_active_) return {};
    profiling_active_ = false;
    Ort::AllocatorWithDefaultOptions alloc;
    auto name = session_.EndProfilingAllocated(alloc);
    return name ? std::string(name.get()) : std::string{};
}

const TensorInfo& ModelSession::input(std::string_view name) const {
    for (const auto& t : inputs_) {
        if (t.name == name) return t;
    }
    throw std::out_of_range(fmt::format("[{}] model has no input named '{}'", report_.stage, name));
}

const TensorInfo& ModelSession::output(std::string_view name) const {
    for (const auto& t : outputs_) {
        if (t.name == name) return t;
    }
    throw std::out_of_range(fmt::format("[{}] model has no output named '{}'", report_.stage, name));
}

bool ModelSession::has_input(std::string_view name) const {
    for (const auto& t : inputs_) {
        if (t.name == name) return true;
    }
    return false;
}

void ModelSession::run(std::span<const Ort::Value> in, std::span<Ort::Value> out) {
    if (in.size() != input_names_.size() || out.size() != output_names_.size()) {
        throw std::invalid_argument(fmt::format("[{}] run(): expected {} inputs/{} outputs, got {}/{}", report_.stage,
                                                input_names_.size(), output_names_.size(), in.size(), out.size()));
    }
    session_.Run(run_options_, input_names_.data(), in.data(), in.size(), output_names_.data(), out.data(),
                 out.size());
}

std::vector<Ort::Value> ModelSession::run(std::span<const Ort::Value> in) {
    if (in.size() != input_names_.size()) {
        throw std::invalid_argument(fmt::format("[{}] run(): expected {} inputs, got {}", report_.stage,
                                                input_names_.size(), in.size()));
    }
    return session_.Run(run_options_, input_names_.data(), in.data(), in.size(), output_names_.data(),
                        output_names_.size());
}

}  // namespace xr
