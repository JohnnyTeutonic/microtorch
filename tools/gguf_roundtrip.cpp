// Semantic end-to-end check, exporter leg.
//
//   gguf_roundtrip <in.gguf> <out.gguf> [out.safetensors]
//
// Reads a trained transformer_cpp GGUF (chat7b et al.), reconstructs the
// HF-style state_dict + hyperparameters + embedded word-level vocabulary,
// and re-exports through microtorch's export_gguf_llama (and optionally
// save_safetensors). If tinyllama.cpp then generates IDENTICAL argmax
// output from in.gguf and out.gguf, the exporter is semantics-preserving
// on real trained weights -- not just byte-plumbing on random fixtures.
//
// The reader here is the same minimal spec-based GGUF v3 parser as
// tests/test_gguf_export.cpp (kept independent of the writer on purpose).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "microtorch/gguf.hpp"
#include "microtorch/safetensors.hpp"

using namespace microtorch;

namespace {

struct RawTensor {
    std::vector<uint64_t> dims;  // file order (writer emits outer-first)
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct GGUFFile {
    std::map<std::string, uint32_t> u32s;
    std::map<std::string, float> f32s;
    std::map<std::string, std::string> strings;
    std::vector<std::string> tokens;
    std::map<std::string, RawTensor> tensors;
    uint64_t data_start = 0;
    std::vector<uint8_t> bytes;
};

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

GGUFFile parse(const std::string& path) {
    GGUFFile g;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    g.bytes.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(g.bytes.data()), static_cast<std::streamsize>(g.bytes.size()));

    size_t p = 0;
    if (rd<uint32_t>(g.bytes, p) != 0x46554747u)
        throw std::runtime_error("not a GGUF file: " + path);
    const uint32_t version = rd<uint32_t>(g.bytes, p);
    if (version != 3) throw std::runtime_error("unsupported GGUF version");
    const uint64_t n_tensors = rd<uint64_t>(g.bytes, p);
    const uint64_t n_meta = rd<uint64_t>(g.bytes, p);

    for (uint64_t i = 0; i < n_meta; ++i) {
        const std::string key = rd_str(g.bytes, p);
        const uint32_t vt = rd<uint32_t>(g.bytes, p);
        switch (vt) {
            case 4:
                g.u32s[key] = rd<uint32_t>(g.bytes, p);
                break;
            case 5:
                g.u32s[key] = static_cast<uint32_t>(rd<int32_t>(g.bytes, p));
                break;
            case 6:
                g.f32s[key] = rd<float>(g.bytes, p);
                break;
            case 8:
                g.strings[key] = rd_str(g.bytes, p);
                break;
            case 9: {
                const uint32_t et = rd<uint32_t>(g.bytes, p);
                const uint64_t n = rd<uint64_t>(g.bytes, p);
                for (uint64_t k = 0; k < n; ++k) {
                    if (et == 8) {
                        std::string tok = rd_str(g.bytes, p);
                        if (key == "tokenizer.ggml.tokens") g.tokens.push_back(std::move(tok));
                    } else if (et == 6) {
                        rd<float>(g.bytes, p);
                    } else {
                        throw std::runtime_error("unexpected array elem type");
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("unexpected metadata type " + std::to_string(vt) +
                                         " for " + key);
        }
    }

    for (uint64_t i = 0; i < n_tensors; ++i) {
        const std::string name = rd_str(g.bytes, p);
        RawTensor t;
        const uint32_t nd = rd<uint32_t>(g.bytes, p);
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(rd<uint64_t>(g.bytes, p));
        t.type = rd<uint32_t>(g.bytes, p);
        t.offset = rd<uint64_t>(g.bytes, p);
        g.tensors.emplace(name, std::move(t));
    }

    const uint64_t align = 32;
    g.data_start = (p + align - 1) & ~(align - 1);
    return g;
}

// Materialize a tensor as a Matrix. File dims are outer-first, so a 2-D
// tensor becomes Matrix(dims[0], dims[1]) -- for projections that is the
// HF/llama [out, in] layout export_gguf_llama re-emits unchanged.
Matrix to_matrix(const GGUFFile& g, const std::string& name) {
    const RawTensor& t = g.tensors.at(name);
    if (t.type != 0) throw std::runtime_error(name + ": only F32 GGUFs supported");
    const size_t rows = t.dims.size() == 2 ? t.dims[0] : 1;
    const size_t cols = t.dims.size() == 2 ? t.dims[1] : t.dims[0];
    Matrix m(rows, cols);
    std::memcpy(&m(0, 0), g.bytes.data() + g.data_start + t.offset, rows * cols * sizeof(float));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: gguf_roundtrip <in.gguf> <out.gguf> "
                     "[out.safetensors]\n");
        return 2;
    }
    const std::string in_path = argv[1], out_path = argv[2];

    GGUFFile g = parse(in_path);

    gguf::LlamaExportConfig cfg;
    const std::string arch =
        g.strings.count("general.architecture") ? g.strings.at("general.architecture") : "llama";
    cfg.architecture = arch;
    cfg.name = g.strings.count("general.name") ? g.strings.at("general.name") : "roundtrip";
    auto u32_or = [&](const std::string& k, uint32_t d) {
        return g.u32s.count(k) ? g.u32s.at(k) : d;
    };
    cfg.context_length = u32_or(arch + ".context_length", 2048);
    cfg.embedding_length = u32_or(arch + ".embedding_length", 0);
    cfg.block_count = u32_or(arch + ".block_count", 0);
    cfg.feed_forward_length = u32_or(arch + ".feed_forward_length", 0);
    cfg.head_count = u32_or(arch + ".attention.head_count", 0);
    cfg.head_count_kv = u32_or(arch + ".attention.head_count_kv", 0);
    cfg.vocab_size = u32_or(arch + ".vocab_size", 0);
    if (g.f32s.count(arch + ".attention.layer_norm_rms_epsilon"))
        cfg.rms_eps = g.f32s.at(arch + ".attention.layer_norm_rms_epsilon");
    if (g.f32s.count(arch + ".rope.freq_base"))
        cfg.rope_freq_base = g.f32s.at(arch + ".rope.freq_base");
    cfg.tokens = g.tokens;
    if (g.strings.count("tokenizer.ggml.model"))
        cfg.tokenizer_model = g.strings.at("tokenizer.ggml.model");
    cfg.bos_token_id = u32_or("tokenizer.ggml.bos_token_id", 1);
    cfg.eos_token_id = u32_or("tokenizer.ggml.eos_token_id", 1);
    cfg.unk_token_id = u32_or("tokenizer.ggml.unknown_token_id", 0);
    cfg.pad_token_id = u32_or("tokenizer.ggml.padding_token_id", 0);

    std::printf(
        "read %s: %s, d=%u layers=%u heads=%u ff=%u vocab=%u "
        "tokens=%zu tensors=%zu\n",
        in_path.c_str(), arch.c_str(), cfg.embedding_length, cfg.block_count, cfg.head_count,
        cfg.feed_forward_length, cfg.vocab_size, g.tokens.size(), g.tensors.size());

    // GGUF names -> HF-Llama names (the exporter's map, reversed).
    std::map<std::string, Matrix> sd;
    sd.emplace("embed_tokens.weight", to_matrix(g, "token_embd.weight"));
    sd.emplace("norm.weight", to_matrix(g, "output_norm.weight"));
    if (g.tensors.count("output.weight"))
        sd.emplace("lm_head.weight", to_matrix(g, "output.weight"));
    for (uint32_t l = 0; l < cfg.block_count; ++l) {
        const std::string gg = "blk." + std::to_string(l) + ".";
        const std::string hf = "layers." + std::to_string(l) + ".";
        sd.emplace(hf + "self_attn.q_proj.weight", to_matrix(g, gg + "attn_q.weight"));
        sd.emplace(hf + "self_attn.k_proj.weight", to_matrix(g, gg + "attn_k.weight"));
        sd.emplace(hf + "self_attn.v_proj.weight", to_matrix(g, gg + "attn_v.weight"));
        sd.emplace(hf + "self_attn.o_proj.weight", to_matrix(g, gg + "attn_output.weight"));
        sd.emplace(hf + "input_layernorm.weight", to_matrix(g, gg + "attn_norm.weight"));
        sd.emplace(hf + "mlp.gate_proj.weight", to_matrix(g, gg + "ffn_gate.weight"));
        sd.emplace(hf + "mlp.up_proj.weight", to_matrix(g, gg + "ffn_up.weight"));
        sd.emplace(hf + "mlp.down_proj.weight", to_matrix(g, gg + "ffn_down.weight"));
        sd.emplace(hf + "post_attention_layernorm.weight", to_matrix(g, gg + "ffn_norm.weight"));
    }

    std::vector<std::string> unmapped;
    gguf::export_gguf_llama(out_path, sd, cfg, &unmapped);
    std::printf("wrote %s (%zu state_dict tensors)\n", out_path.c_str(), sd.size());

    if (argc > 3) {
        save_safetensors(argv[3], sd);
        std::printf("wrote %s (safetensors twin)\n", argv[3]);

        // HF-style config.json sidecar: tinyllama.cpp's SafeTensors
        // directory path reads hyperparameters from it
        // (SafeTensorsLoader::load_model_config_from_json).
        const std::string st_path(argv[3]);
        const size_t slash = st_path.find_last_of("/\\");
        const std::string dir =
            slash == std::string::npos ? std::string(".") : st_path.substr(0, slash);
        std::ofstream cj(dir + "/config.json");
        cj << "{\n"
           << "  \"architectures\": [\"LlamaForCausalLM\"],\n"
           << "  \"model_type\": \"llama\",\n"
           << "  \"hidden_size\": " << cfg.embedding_length << ",\n"
           << "  \"intermediate_size\": " << cfg.feed_forward_length << ",\n"
           << "  \"num_hidden_layers\": " << cfg.block_count << ",\n"
           << "  \"num_attention_heads\": " << cfg.head_count << ",\n"
           << "  \"num_key_value_heads\": "
           << (cfg.head_count_kv ? cfg.head_count_kv : cfg.head_count) << ",\n"
           << "  \"vocab_size\": " << cfg.vocab_size << ",\n"
           << "  \"max_position_embeddings\": " << cfg.context_length << ",\n"
           << "  \"rms_norm_eps\": " << cfg.rms_eps << ",\n"
           << "  \"rope_theta\": " << cfg.rope_freq_base << ",\n"
           << "  \"bos_token_id\": " << cfg.bos_token_id << ",\n"
           << "  \"eos_token_id\": " << cfg.eos_token_id << ",\n"
           << "  \"torch_dtype\": \"float32\"\n"
           << "}\n";
        std::printf("wrote %s/config.json (HF sidecar)\n", dir.c_str());
    }
    return 0;
}
