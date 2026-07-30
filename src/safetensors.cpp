#include "microtorch/safetensors.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {

std::map<std::string, Matrix> load_safetensors(
    const std::string& path, std::map<std::string, std::string>* skipped) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);

    std::uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);      // little-endian, like x86
    if (!f || hlen == 0 || hlen > (1u << 26)) {
        throw std::runtime_error("safetensors: implausible header length");
    }
    std::string htext(hlen, '\0');
    f.read(htext.data(), static_cast<std::streamsize>(hlen));
    const auto header = nlohmann::json::parse(htext);
    const std::streamoff base = 8 + static_cast<std::streamoff>(hlen);

    std::map<std::string, Matrix> out;
    for (auto it = header.begin(); it != header.end(); ++it) {
        std::string name = it.key();
        if (name == "__metadata__") continue;
        if (name.rfind("transformer.", 0) == 0) name = name.substr(12);
        const auto& meta = it.value();
        const auto shape = meta.at("shape").get<std::vector<std::uint64_t>>();
        const auto offs = meta.at("data_offsets").get<std::vector<std::uint64_t>>();

        if (shape.size() > 2) {                     // rank-skip, see header
            if (skipped) (*skipped)[name] = "rank " + std::to_string(shape.size());
            continue;
        }
        const std::string dtype = meta.at("dtype").get<std::string>();
        if (dtype != "F32") {
            throw std::runtime_error("safetensors: " + name + " is " + dtype +
                                     "; this loader is F32-only (phase 1c)");
        }
        const size_t rows = shape.size() == 2 ? shape[0] : 1;
        const size_t cols = shape.size() == 2 ? shape[1]
                            : (shape.empty() ? 1 : shape[0]);
        Matrix m(rows, cols);
        const std::uint64_t nbytes = offs.at(1) - offs.at(0);
        if (nbytes != rows * cols * 4) {
            throw std::runtime_error("safetensors: size mismatch at " + name);
        }
        f.seekg(base + static_cast<std::streamoff>(offs.at(0)));
        f.read(reinterpret_cast<char*>(m.data()),
               static_cast<std::streamsize>(nbytes));
        if (!f) throw std::runtime_error("safetensors: short read at " + name);
        out.emplace(std::move(name), std::move(m));
    }
    return out;
}

}  // namespace microtorch
