#include "tiktoken.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace tiktoken {

Encoding::Encoding(const std::string& encoding_name) {
    std::string vocab_path = "../data/tiktoken_data/";
    if (encoding_name == "gpt2") {
        vocab_path += "gpt2.vocab.json";
    } else {
        vocab_path += "cl100k_base.vocab.json";
    }
    
    // Load vocabulary from JSON file
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }
    
    try {
        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        // Convert JSON vocabulary to our map
        for (auto& [token, id] : vocab_json.items()) {
            vocab[token] = id.get<int>();
        }
        
        std::cout << "Loaded " << vocab.size() << " tokens from " << vocab_path << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse vocabulary file: " + std::string(e.what()));
    }
}

std::vector<int> Encoding::encode(const std::string& text) const {
    #ifdef USE_CUDA
    // Use GPU acceleration if available
    std::vector<int> tokens;
    cuda::parallel_tokenize(text, *this, tokens);
    return tokens;
    #else
    // Original CPU implementation
    std::vector<int> tokens;
    std::string current_token;
    
    for (size_t i = 0; i < text.length(); i++) {
        current_token += text[i];
        
        // Try to find the longest matching token
        std::string best_match;
        int best_match_id = -1;
        
        for (const auto& [token, id] : vocab) {
            if (current_token.find(token) == 0 && token.length() > best_match.length()) {
                best_match = token;
                best_match_id = id;
            }
        }
        
        if (best_match_id != -1) {
            tokens.push_back(best_match_id);
            current_token = current_token.substr(best_match.length());
            i -= current_token.length();  // Rewind to process remaining characters
        }
    }
    
    return tokens;
    #endif
}

std::string Encoding::decode(const std::vector<int>& tokens) const {
    std::string text;
    
    // Create reverse mapping
    std::unordered_map<int, std::string> id_to_token;
    for (const auto& [token, id] : vocab) {
        id_to_token[id] = token;
    }
    
    // Decode each token
    for (int id : tokens) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            text += it->second;
        }
    }
    
    return text;
}

size_t Encoding::vocab_size() const {
    return vocab.size();
}

void Encoding::add_special_token(const std::string& token, int id) {
    vocab[token] = id;
}

void Encoding::load_vocab() {
    // Load cl100k_base vocabulary
    // This is a simplified version - in practice, we'd load from a file
    // For now, adding a few common tokens as an example
    vocab["!"] = 0;
    vocab[" "] = 1;
    vocab["the"] = 2;
    vocab["hello"] = 3;
    vocab["world"] = 4;
    vocab["I"] = 5;
    vocab["am"] = 6;
    vocab["to"] = 7;
    vocab["and"] = 8;
    vocab["a"] = 9;
    // ... more vocabulary entries would be loaded from a file
}

void Encoding::load_merges(const std::string& encoding_type) {
    // Load cl100k_base merges
    // This is a simplified version - in practice, we'd load from a file
    merges["h e"] = "he";
    merges["he l"] = "hel";
    merges["hel l"] = "hell";
    merges["hell o"] = "hello";
    merges["w o"] = "wo";
    merges["wo r"] = "wor";
    merges["wor l"] = "worl";
    merges["worl d"] = "world";
    // ... more merge rules would be loaded from a file
}

std::vector<std::string> Encoding::split_into_bytes(const std::string& text) const {
    std::vector<std::string> bytes;
    for (unsigned char c : text) {
        // Convert each byte to a string representation
        bytes.push_back(std::string(1, c));
    }
    return bytes;
}

bool Encoding::should_merge(const std::string& piece1, const std::string& piece2) const {
    std::string pair = piece1 + " " + piece2;
    return merges.find(pair) != merges.end();
}

std::vector<std::string> Encoding::byte_pair_encode(const std::string& token) const {
    // Start with individual bytes
    std::vector<std::string> pieces = split_into_bytes(token);
    
    // Keep merging according to the merge rules
    bool changes_made;
    do {
        changes_made = false;
        
        // Look for pairs that can be merged
        for (size_t i = 0; i < pieces.size() - 1; i++) {
            if (should_merge(pieces[i], pieces[i + 1])) {
                // Merge the pair
                std::string merged = merges.at(pieces[i] + " " + pieces[i + 1]);
                pieces[i] = merged;
                pieces.erase(pieces.begin() + i + 1);
                changes_made = true;
                break;
            }
        }
    } while (changes_made);
    
    return pieces;
}

} // namespace tiktoken 