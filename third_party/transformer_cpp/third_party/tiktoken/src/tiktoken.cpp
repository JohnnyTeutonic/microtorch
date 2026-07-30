#include "tiktoken.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace tiktoken {

void Encoding::add_special_token(const std::string& token, int id) {
    vocab[token] = id;
}

void Encoding::load_merges(const std::string& merge_rule) {
    // Split the merge rule into parts
    size_t space_pos = merge_rule.find(' ');
    if (space_pos != std::string::npos) {
        std::string first = merge_rule.substr(0, space_pos);
        std::string second = merge_rule.substr(space_pos + 1);
        merges[first + " " + second] = first + second;
    }
}

std::vector<int> Encoding::encode(const std::string& text) const {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        auto it = vocab.find(word);
        if (it != vocab.end()) {
            tokens.push_back(it->second);
        }
    }
    
    return tokens;
}

std::string Encoding::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        for (const auto& [str, id] : vocab) {
            if (id == token) {
                result += str + " ";
                break;
            }
        }
    }
    // Remove trailing space if result is not empty
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

size_t Encoding::get_vocab_size() const {
    return vocab.size();
}

} // namespace tiktoken 