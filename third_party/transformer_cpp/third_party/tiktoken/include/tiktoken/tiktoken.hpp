#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace tiktoken {

class Encoding {
public:
    explicit Encoding(const std::string& encoding_name = "gpt2");
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    void add_special_token(const std::string& token, int id);
    size_t get_vocab_size() const;
    void load_merges(const std::string& merge_rule);

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, std::string> merges;
};

} // namespace tiktoken 