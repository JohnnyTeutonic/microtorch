// mtstudio — the run-spec driver (STUDIO_PLAN.md M1).
//
//   mtstudio run spec.json          execute the spec
//   mtstudio plan spec.json         print the resolved plan and exit
//
// One JSON describes the lifecycle; this driver executes it stage by
// stage and emits a JSONL event stream (stdout + <out>/events.jsonl) that
// the M2 UI will consume. v0 scope: arch presets + custom dims, corpus +
// GGUF vocab, train with early stopping + checkpoint/resume, safetensors
// export, GGUF export for llama-family models, serve-command print.
// (arXiv arch population and the finetune stage are the documented next
// increments; papers/fetch.py already emits the config this schema takes.)
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "microtorch/gguf.hpp"
#include "microtorch/llama.hpp"
#include "microtorch/safetensors.hpp"
#include "parity_model.hpp"

using namespace microtorch;
using nlohmann::json;

namespace {

struct Spec {
    std::string name = "run";
    // arch
    std::string family = "gpt2";      // gpt2 | llama
    std::string attention = "exact";  // exact | kimi | srd (gpt2 family)
    size_t d = 128, layers = 2, heads = 4, T = 128;
    // data
    std::string corpus, vocab_gguf;
    size_t vocab_cap = 4096;
    // train
    int steps = 500;
    float lr = 3e-3f, clip = 1.0f, lambda_gate = 0.05f;
    int eval_every = 50, ckpt_every = 100;
    int batch = 1;           // sequences per FORWARD (stacked rows, one graph)
    int accum = 1;           // batches accumulated per optimizer step
    bool ckpt_act = false;   // activation checkpointing per block
    unsigned seed = 7;       // model init + data-order seed (Atlas multi-seed)
    int gradmap_every = 5;   // per-layer grad-norm event cadence
    size_t es_patience = 0;  // early stopping (0 = off)
    float es_min_delta = 0.0f;
    // export/serve
    bool exp_safetensors = true, exp_gguf = false;
    bool serve = false;
    std::string out_dir = "mtstudio_out";
};

// Known presets; "custom" reads arch.custom.* instead.
const std::map<std::string, std::array<size_t, 4>> PRESETS = {
    // name -> {d, layers, heads, T}
    {"gpt2-nano", {128, 2, 4, 128}},  {"llama-tiny", {128, 2, 4, 128}},
    {"gpt2-small", {256, 4, 8, 256}}, {"kimi-tiny", {128, 2, 4, 128}},
    {"srd-tiny", {128, 2, 4, 128}},
};

Spec parse_spec(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open spec " + path);
    json j = json::parse(f, nullptr, true, /*ignore_comments=*/true);
    Spec s;
    s.name = j.value("name", s.name);

    const json arch = j.value("arch", json::object());
    if (arch.contains("preset")) {
        const std::string p = arch["preset"];
        auto it = PRESETS.find(p);
        if (it == PRESETS.end()) throw std::runtime_error("unknown preset " + p);
        s.d = it->second[0];
        s.layers = it->second[1];
        s.heads = it->second[2];
        s.T = it->second[3];
        if (p.rfind("kimi", 0) == 0) s.attention = "kimi";
        if (p.rfind("srd", 0) == 0) s.attention = "srd";
        if (p.rfind("llama", 0) == 0) s.family = "llama";
    }
    if (arch.contains("custom")) {
        const json c = arch["custom"];
        s.d = c.value("d", s.d);
        s.layers = c.value("layers", s.layers);
        s.heads = c.value("heads", s.heads);
        s.attention = c.value("attention", s.attention);
    }

    const json data = j.value("data", json::object());
    s.corpus = data.value("corpus", "");
    s.vocab_gguf = data.value("vocab", "");
    s.vocab_cap = data.value("vocab_cap", s.vocab_cap);
    s.T = data.value("T", s.T);

    const json tr = j.value("train", json::object());
    s.steps = tr.value("steps", s.steps);
    s.lr = tr.value("lr", s.lr);
    s.clip = tr.value("clip", s.clip);
    s.eval_every = tr.value("eval_every", s.eval_every);
    s.ckpt_every = tr.value("checkpoint_every", s.ckpt_every);
    s.batch = tr.value("batch", s.batch);
    s.accum = tr.value("accum", s.accum);
    s.ckpt_act = tr.value("checkpoint_activations", s.ckpt_act);
    s.seed = tr.value("seed", s.seed);
    s.gradmap_every = tr.value("gradmap_every", s.gradmap_every);
    if (tr.contains("early_stopping")) {
        s.es_patience = tr["early_stopping"].value("patience", size_t(0));
        s.es_min_delta = tr["early_stopping"].value("min_delta", 0.0f);
    }

    const json ex = j.value("export", json::object());
    for (const auto& fmt : ex.value("formats", std::vector<std::string>{"safetensors"})) {
        if (fmt == "gguf") s.exp_gguf = true;
        if (fmt == "safetensors") s.exp_safetensors = true;
    }
    s.serve = j.value("serve", json::object()).value("on_finish", false);
    s.out_dir = j.value("out_dir", s.out_dir);
    return s;
}

// ---- events: JSONL to stdout + file (the M2 UI's feed) ----
struct Events {
    std::ofstream file;
    explicit Events(const std::string& path) : file(path, std::ios::app) {}
    void emit(const json& j) {
        const std::string line = j.dump();
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        file << line << "\n";
        file.flush();
    }
};

// GGUF vocab reader + word tokenizer (the srd_parity path).
std::vector<std::string> read_gguf_vocab(const std::string& path);
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens);

parity::AttnKind attn_kind(const std::string& s) {
    if (s == "kimi") return parity::AttnKind::KIMI;
    if (s == "srd") return parity::AttnKind::SRD;
    return parity::AttnKind::EXACT;
}

// Per-module L2 gradient norms, grouped by the first dotted-path segment
// ("wte", "attn_0", "mlp_1", ...). This is the data the M2 node-graph
// glows with: fading nodes = vanishing gradients, flashing = exploding.
json grad_map(const nn::Module& m) {
    std::map<std::string, double> sq;
    for (const auto& [name, p] : m.named_parameters()) {
        if (p->grad.rows() == 0) continue;
        // Group at the first segment — except structural containers
        // ("layers.N", "h.N", "blocks.N"), which keep their index so the
        // node graph gets per-block resolution instead of one blob.
        auto cut = name.find('.');
        std::string group = cut == std::string::npos ? name : name.substr(0, cut);
        if ((group == "layers" || group == "h" || group == "blocks") && cut != std::string::npos) {
            const auto cut2 = name.find('.', cut + 1);
            group = cut2 == std::string::npos ? name : name.substr(0, cut2);
        }
        double acc = 0;
        for (size_t i = 0; i < p->grad.rows(); ++i)
            for (size_t j = 0; j < p->grad.cols(); ++j)
                acc += static_cast<double>(p->grad(i, j)) * p->grad(i, j);
        sq[group] += acc;
    }
    json out = json::object();
    for (const auto& [k, v] : sq) out[k] = std::sqrt(v);
    return out;
}

int run(const Spec& s, bool plan_only) {
    std::printf("== mtstudio: %s ==\n", s.name.c_str());
    std::printf("arch: %s d=%zu layers=%zu heads=%zu | T=%zu vocab_cap=%zu\n", s.attention.c_str(),
                s.d, s.layers, s.heads, s.T, s.vocab_cap);
    std::printf(
        "train: %d steps batch=%d accum=%d lr=%g clip=%g eval_every=%d "
        "ckpt_every=%d early_stop(patience=%zu, min_delta=%g)\n",
        s.steps, s.batch, s.accum, s.lr, s.clip, s.eval_every, s.ckpt_every, s.es_patience,
        s.es_min_delta);
    std::printf("export: %s%s | serve: %s | out: %s\n", s.exp_safetensors ? "safetensors " : "",
                s.exp_gguf ? "gguf" : "", s.serve ? "yes" : "no", s.out_dir.c_str());
    if (plan_only) return 0;
    if (s.corpus.empty() || s.vocab_gguf.empty())
        throw std::runtime_error("spec needs data.corpus and data.vocab");
    if (s.family != "llama" && s.layers != 2)
        throw std::runtime_error("gpt2 family: layers must be 2 (parity model)");

    std::system(("mkdir -p " + s.out_dir).c_str());
    Events ev(s.out_dir + "/events.jsonl");
    ev.emit({{"event", "start"}, {"name", s.name}, {"steps", s.steps}});

    // Data.
    auto tokens = read_gguf_vocab(s.vocab_gguf);
    if (s.vocab_cap > 0 && s.vocab_cap < tokens.size()) tokens.resize(s.vocab_cap);
    std::map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); ++i) vocab.emplace(tokens[i], static_cast<int>(i));
    std::ifstream cf(s.corpus);
    if (!cf) throw std::runtime_error("cannot open corpus " + s.corpus);
    std::string text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto ids = tokenize(text, vocab, 400000);
    // Hold out the tail 5% for validation (early stopping's signal).
    const size_t val_start = ids.size() - ids.size() / 20;
    ev.emit({{"event", "data"},
             {"tokens", ids.size()},
             {"vocab", tokens.size()},
             {"val_tokens", ids.size() - val_start}});

    // Model + optimizer (+ resume). Two families behind one seam: the
    // gpt2 parity model (exact/kimi/srd attention) or nn::Llama (RMSNorm/
    // RoPE/SwiGLU, HF names -> GGUF-exportable).
    std::shared_ptr<parity::ParityLM> gpt;
    std::shared_ptr<nn::Llama> llama;
    if (s.family == "llama") {
        nn::LlamaConfig lc;
        lc.vocab = tokens.size();
        lc.d = s.d;
        lc.n_layers = s.layers;
        lc.n_heads = s.heads;
        lc.d_ff = 3 * s.d;
        lc.n_ctx = s.T;
        llama = std::make_shared<nn::Llama>(lc, s.seed);
        llama->checkpoint_blocks = s.ckpt_act;
    } else {
        gpt = std::make_shared<parity::ParityLM>(attn_kind(s.attention), tokens.size(), s.d,
                                                 s.heads, s.T, s.seed);
        if (s.ckpt_act) {
            throw std::runtime_error(
                "train.checkpoint_activations requires the llama family for now");
        }
    }
    // Atlas stage-0 structural echo: the run's identity as a data point.
    const size_t n_params =
        (llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt))
            .parameter_count();
    ev.emit({{"event", "model"},
             {"family", s.family},
             {"attention", s.attention},
             {"d", s.d},
             {"layers", s.layers},
             {"heads", s.heads},
             {"T", s.T},
             {"vocab", tokens.size()},
             {"batch", s.batch},
             {"accum", s.accum},
             {"lr", s.lr},
             {"seed", s.seed},
             {"checkpoint_activations", s.ckpt_act},
             {"params", n_params}});
    nn::Module& model_ref =
        llama ? static_cast<nn::Module&>(*llama) : static_cast<nn::Module&>(*gpt);
    auto fwd = [&](const std::vector<int>& ids, size_t seq_len = 0) {
        return llama ? llama->forward(ids, seq_len) : gpt->forward(ids, seq_len);
    };
    if (s.batch > 1 && !llama && attn_kind(s.attention) != parity::AttnKind::EXACT) {
        throw std::runtime_error("train.batch > 1 requires llama family or exact attention");
    }
    model_ref.train();
    nn::AdamW opt(model_ref.parameters(), s.lr);
    const std::string ckpt = s.out_dir + "/model.safetensors";
    const std::string state_path = s.out_dir + "/state.txt";
    int start_step = 0;
    {
        std::ifstream st(state_path);
        if (st >> start_step && start_step > 0) {
            model_ref.load_state_dict(load_safetensors(ckpt));
            ev.emit({{"event", "resume"}, {"step", start_step}});
        } else
            start_step = 0;
    }

    // Data order follows the spec seed so multi-seed sweeps vary both init
    // and batch composition (offset keeps seed=7 runs distinct from the
    // old fixed-123 stream only in the documented way).
    std::mt19937 rng(123 + 1000003u * s.seed);
    const auto t_train0 = std::chrono::steady_clock::now();
    int last_step = start_step;
    // Resume determinism: each step consumed accum*batch draws.
    for (int i = 0; i < start_step * s.accum * s.batch; ++i) rng();
    auto save = [&](int step) {
        save_safetensors(ckpt, model_ref.state_dict());
        std::ofstream st(state_path);
        st << step << "\n";
    };

    // Train.
    float best_val = 1e30f;
    size_t evals_flat = 0;
    bool stopped_early = false;
    const bool is_srd = attn_kind(s.attention) == parity::AttnKind::SRD;
    for (int step = start_step + 1; step <= s.steps; ++step) {
        last_step = step;
        const size_t lim = val_start - s.T - 1;
        // Mini-batching + accumulation: each of s.accum micro-steps stacks
        // s.batch sequences into ONE forward ([batch*T, d] rows; positions
        // and the attention mask restart per sequence — receipts in
        // tests/test_batching.cpp), backward pre-scaled by 1/accum so the
        // summed gradient is the mean over all batch*accum sequences.
        opt.zero_grad();
        float task_mean = 0, gate_mean = 0;
        for (int k = 0; k < s.accum; ++k) {
            std::vector<int> x, y;
            x.reserve(s.batch * s.T);
            y.reserve(s.batch * s.T);
            for (int b = 0; b < s.batch; ++b) {
                const size_t at = rng() % lim;
                x.insert(x.end(), ids.begin() + at, ids.begin() + at + s.T);
                y.insert(y.end(), ids.begin() + at + 1, ids.begin() + at + s.T + 1);
            }
            Var logits = fwd(x, s.batch > 1 ? s.T : 0);
            Var task = ops::cross_entropy(logits, y);
            Var loss = task;
            if (is_srd) loss = ops::add(task, ops::scale(gpt->mean_gate(), s.lambda_gate));
            backward(ops::scale(loss, 1.0f / static_cast<float>(s.accum)));
            task_mean += task->data(0, 0) / static_cast<float>(s.accum);
            if (is_srd) gate_mean += gpt->mean_gate()->data(0, 0) / static_cast<float>(s.accum);
        }
        // Per-module grad norms BEFORE clipping: this is the true signal
        // the glow UI wants (clipping would mask explosions).
        json gm;
        if (s.gradmap_every > 0 && step % s.gradmap_every == 0) gm = grad_map(model_ref);
        const float total_norm = ops::clip_grad_norm(model_ref.parameters(), s.clip);
        opt.step();

        json e = {
            {"event", "step"}, {"step", step}, {"loss", task_mean}, {"grad_norm", total_norm}};
        if (is_srd) e["gate"] = gate_mean;
        if (!gm.is_null()) e["grads"] = gm;
        ev.emit(e);

        if (step % s.eval_every == 0) {
            NoGrad ng;
            model_ref.eval();
            double vl = 0;
            const int NV = 8;
            std::mt19937 vrng(999);
            for (int k = 0; k < NV; ++k) {
                const size_t va = val_start + vrng() % (ids.size() - val_start - s.T - 1);
                std::vector<int> vx(ids.begin() + va, ids.begin() + va + s.T);
                std::vector<int> vy(ids.begin() + va + 1, ids.begin() + va + s.T + 1);
                vl += ops::cross_entropy(fwd(vx), vy)->data(0, 0);
            }
            vl /= NV;
            model_ref.train();
            ev.emit({{"event", "eval"}, {"step", step}, {"val_loss", vl}});
            if (s.es_patience > 0) {
                if (vl < best_val - s.es_min_delta) {
                    best_val = static_cast<float>(vl);
                    evals_flat = 0;
                } else if (++evals_flat >= s.es_patience) {
                    ev.emit({{"event", "early_stop"}, {"step", step}, {"best_val", best_val}});
                    stopped_early = true;
                }
            } else if (vl < best_val)
                best_val = static_cast<float>(vl);
        }
        if (step % s.ckpt_every == 0 || stopped_early || step == s.steps) {
            save(step);
            if (stopped_early) break;
        }
    }

    // Export.
    if (s.exp_safetensors) {
        save_safetensors(s.out_dir + "/" + s.name + ".safetensors", model_ref.state_dict());
        ev.emit({{"event", "export"}, {"format", "safetensors"}});
    }
    if (s.exp_gguf) {
        if (llama) {
            auto sd2 = model_ref.state_dict();
            // Tied head: inject lm_head = E^T in microtorch [in, out]
            // layout; the exporter transposes it back into llama
            // [vocab, hidden] byte order under weights_in_out.
            if (!sd2.count("lm_head.weight")) {
                const Matrix& E = llama->embed_tokens->weight->data;
                Matrix ET(E.cols(), E.rows());
                for (size_t i = 0; i < E.rows(); ++i)
                    for (size_t j = 0; j < E.cols(); ++j) ET(j, i) = E(i, j);
                sd2.emplace("lm_head.weight", std::move(ET));
            }
            gguf::LlamaExportConfig gc;
            gc.name = s.name;
            gc.embedding_length = (uint32_t)s.d;
            gc.block_count = (uint32_t)s.layers;
            gc.head_count = (uint32_t)s.heads;
            gc.feed_forward_length = (uint32_t)(3 * s.d);
            gc.vocab_size = (uint32_t)tokens.size();
            gc.context_length = (uint32_t)s.T;
            gc.rms_eps = 1e-6f;
            gc.weights_in_out = true;  // microtorch Linear is [in, out]
            gc.tokens = tokens;
            const std::string gpath = s.out_dir + "/" + s.name + ".gguf";
            gguf::export_gguf_llama(gpath, sd2, gc);
            ev.emit({{"event", "export"}, {"format", "gguf"}, {"path", gpath}});
        } else {
            ev.emit({{"event", "export_skipped"},
                     {"format", "gguf"},
                     {"reason", "gpt2-family blocks are not llama-shaped"}});
        }
    }
    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_train0).count();
    ev.emit({{"event", "done"},
             {"best_val", best_val},
             {"early_stopped", stopped_early},
             {"final_step", last_step},
             {"wall_seconds", wall_s}});
    // Atlas stage-0 result row: one durable JSON per run, joining the
    // structural echo with the outcome. atlas_extract.py enriches it with
    // behavioural features computed from events.jsonl.
    {
        json result = {{"name", s.name},
                       {"family", s.family},
                       {"attention", s.attention},
                       {"d", s.d},
                       {"layers", s.layers},
                       {"heads", s.heads},
                       {"T", s.T},
                       {"batch", s.batch},
                       {"accum", s.accum},
                       {"lr", s.lr},
                       {"seed", s.seed},
                       {"checkpoint_activations", s.ckpt_act},
                       {"params", n_params},
                       {"steps_requested", s.steps},
                       {"final_step", last_step},
                       {"best_val", best_val},
                       {"early_stopped", stopped_early},
                       {"wall_seconds", wall_s},
                       {"tokens_per_second", wall_s > 0 ? (last_step - start_step) *
                                                              static_cast<double>(s.batch) *
                                                              s.accum * s.T / wall_s
                                                        : 0.0}};
        std::ofstream rf(s.out_dir + "/result.json");
        rf << result.dump(2) << "\n";
    }

    if (s.serve) {
        if (llama && s.exp_gguf) {
            std::printf(
                "serve: tinyllama %s/%s.gguf %s/%s.gguf 4 prompt "
                "\"once upon a time\" --max-tokens 40 -ngl 0 "
                "--top-k 1 --raw-prompt\n",
                s.out_dir.c_str(), s.name.c_str(), s.out_dir.c_str(), s.name.c_str());
        } else {
            std::printf(
                "serve: exported to %s/%s.safetensors (gguf serving "
                "needs family=llama + gguf export)\n",
                s.out_dir.c_str(), s.name.c_str());
        }
    }
    return 0;
}

}  // namespace

// ---- GGUF vocab + tokenizer (shared logic with srd_parity) ----
namespace {
template <typename T>
T rd(const std::vector<uint8_t>& b, size_t& p) {
    T v;
    std::memcpy(&v, b.data() + p, sizeof(T));
    p += sizeof(T);
    return v;
}
std::string rd_str(const std::vector<uint8_t>& b, size_t& p) {
    const uint64_t n = rd<uint64_t>(b, p);
    std::string s(reinterpret_cast<const char*>(b.data() + p), n);
    p += n;
    return s;
}
std::vector<std::string> read_gguf_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> b(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
    size_t p = 0;
    if (rd<uint32_t>(b, p) != 0x46554747u) throw std::runtime_error("not GGUF");
    rd<uint32_t>(b, p);
    rd<uint64_t>(b, p);
    const uint64_t n_meta = rd<uint64_t>(b, p);
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(b, p);
        const uint32_t vt = rd<uint32_t>(b, p);
        switch (vt) {
            case 4:
                rd<uint32_t>(b, p);
                break;
            case 5:
                rd<int32_t>(b, p);
                break;
            case 6:
                rd<float>(b, p);
                break;
            case 8:
                rd_str(b, p);
                break;
            case 9: {
                const uint32_t et = rd<uint32_t>(b, p);
                const uint64_t n = rd<uint64_t>(b, p);
                for (uint64_t k = 0; k < n; ++k) {
                    if (et == 8) {
                        std::string t = rd_str(b, p);
                        if (key == "tokenizer.ggml.tokens") tokens.push_back(std::move(t));
                    } else if (et == 6)
                        rd<float>(b, p);
                    else
                        throw std::runtime_error("bad array");
                }
                break;
            }
            default:
                throw std::runtime_error("bad meta");
        }
    }
    return tokens;
}
std::vector<int> tokenize(const std::string& text, const std::map<std::string, int>& vocab,
                          size_t max_tokens) {
    std::vector<int> ids;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        auto it = vocab.find(cur);
        ids.push_back(it == vocab.end() ? 0 : it->second);
        cur.clear();
    };
    for (char ch : text) {
        if (ids.size() >= max_tokens) break;
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalpha(c) || c == '\'' || std::isdigit(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else {
            flush();
            if (!std::isspace(c)) {
                std::string pch(1, static_cast<char>(c));
                auto it = vocab.find(pch);
                ids.push_back(it == vocab.end() ? 0 : it->second);
            }
        }
    }
    flush();
    return ids;
}
}  // namespace

// ---- M2 live mode: minimal HTTP server (POSIX; runs under WSL/Linux).
// GET /              -> the studio UI (index.html)
// GET /events.jsonl  -> the run dir's current event stream
// The UI polls /events.jsonl every 2s when served over http, turning the
// dashboard into a live training monitor.
#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int serve_ui(const std::string& out_dir, int port, const std::string& ui_path) {
#ifdef _WIN32
    std::fprintf(stderr,
                 "mtstudio serve: POSIX-only for now (run under "
                 "WSL); Windows needs a winsock port.\n");
    (void)out_dir;
    (void)port;
    (void)ui_path;
    return 1;
#else
    const std::string ui = slurp(ui_path);
    if (ui.empty()) throw std::runtime_error("cannot read UI at " + ui_path + " (set MTSTUDIO_UI)");
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("bind failed (port in use?)");
    if (::listen(fd, 8) < 0) throw std::runtime_error("listen failed");
    std::printf(
        "mtstudio serve: http://localhost:%d/  (events from %s, "
        "Ctrl-C to stop)\n",
        port, out_dir.c_str());

    auto respond = [](int c, const char* status, const char* ctype, const std::string& body) {
        char head[256];
        const int n = std::snprintf(head, sizeof(head),
                                    "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                                    "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                                    status, ctype, body.size());
        (void)!::write(c, head, n);
        (void)!::write(c, body.data(), body.size());
    };

    for (;;) {
        const int c = ::accept(fd, nullptr, nullptr);
        if (c < 0) continue;
        char req[1024] = {0};
        const ssize_t r = ::read(c, req, sizeof(req) - 1);
        std::string line = r > 0 ? std::string(req) : "";
        if (line.rfind("GET /events.jsonl", 0) == 0) {
            respond(c, "200 OK", "application/jsonl", slurp(out_dir + "/events.jsonl"));
        } else if (line.rfind("GET / ", 0) == 0 || line.rfind("GET /index.html", 0) == 0) {
            respond(c, "200 OK", "text/html; charset=utf-8", ui);
        } else {
            respond(c, "404 Not Found", "text/plain", "404");
        }
        ::close(c);
    }
#endif
}
}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    try {
        if ((cmd == "run" || cmd == "plan") && argc >= 3)
            return run(parse_spec(argv[2]), cmd == "plan");
        if (cmd == "serve" && argc >= 3) {
            const int port = argc > 3 ? std::atoi(argv[3]) : 8123;
            const char* ui = std::getenv("MTSTUDIO_UI");
            return serve_ui(argv[2], port, ui ? ui : "studio/index.html");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtstudio: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr,
                 "usage: mtstudio run|plan spec.json\n"
                 "       mtstudio serve <out_dir> [port]   (MTSTUDIO_UI "
                 "overrides the index.html path)\n");
    return 2;
}
