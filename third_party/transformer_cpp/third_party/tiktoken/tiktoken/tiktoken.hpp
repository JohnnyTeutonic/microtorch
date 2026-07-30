#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <regex>
#include <cctype>
#include "vocab_loader.hpp"

namespace std {
    template<typename T>
    struct hash<vector<T>> {
        size_t operator()(const vector<T>& v) const {
            size_t hash = v.size();
            for (const T& item : v) {
                hash ^= std::hash<T>{}(item) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
}

namespace tiktoken {

// Special tokens
const int BOS_ID = 100264;  // Beginning of sequence
const int EOS_ID = 100265;  // End of sequence
const int PAD_ID = 100266;  // Padding

class Encoding {
private:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::pair<std::string, int>> special_tokens;
    std::unordered_map<std::vector<uint8_t>, int> byte_encoder;
    std::unordered_map<int, std::vector<uint8_t>> byte_decoder;
    
    void initialize_byte_encoding() {
        // Initialize byte-level encodings for basic character handling
        byte_encoder.clear();
        byte_decoder.clear();
        
        // Initialize basic byte-level encodings (0-255)
        for (int b = 0; b < 256; b++) {
            std::vector<uint8_t> bytes{static_cast<uint8_t>(b)};
            std::string token = std::string(1, static_cast<char>(b));
            
            // Only add to byte encoding if it exists in vocabulary
            if (vocab.find(token) != vocab.end()) {
                byte_encoder[bytes] = vocab[token];
                byte_decoder[vocab[token]] = bytes;
            }
        }
        
        std::cout << "Initialized basic tokenizer:"
                  << "\n- Vocabulary size: " << vocab.size()
                  << "\n- Special tokens: " << special_tokens.size() << std::endl;
    }

    std::vector<std::string> tokenize_text(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string current_token;
        
        // First check for special tokens
        for (const auto& [special_token, _] : special_tokens) {
            size_t pos = text.find(special_token);
            if (pos != std::string::npos) {
                if (pos > 0) {
                    tokens.push_back(text.substr(0, pos));
                }
                tokens.push_back(special_token);
                if (pos + special_token.length() < text.length()) {
                    auto remaining_tokens = tokenize_text(text.substr(pos + special_token.length()));
                    tokens.insert(tokens.end(), remaining_tokens.begin(), remaining_tokens.end());
                }
                return tokens;
            }
        }
        
        // Basic word and punctuation tokenization
        for (size_t i = 0; i < text.length(); i++) {
            char c = text[i];
            if (std::isspace(c)) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
            }
            else if (std::ispunct(c)) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
                tokens.push_back(std::string(1, c));
            }
            else {
                current_token += c;
            }
        }
        
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
        
        return tokens;
    }
    
public:
    explicit Encoding(const std::string& encoding_type = "basic") {
        // Load vocabulary from file
        auto [loaded_vocab, _] = VocabLoader::load_vocab_and_merges(encoding_type);
        vocab = std::move(loaded_vocab);

        // Initialize special tokens
        special_tokens = {
            {"<unk>", 0},
            {"<s>", 1},
            {"</s>", 2},
            {"<pad>", 3},
            {"<mask>", 4}
        };

        // Add special tokens to vocabulary
        for (const auto& [token, id] : special_tokens) {
            vocab[token] = id;
        }

        initialize_byte_encoding();
    }
    
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> result;
        auto tokens = tokenize_text(text);
        
        for (const auto& token : tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                result.push_back(it->second);
            } else {
                // Handle unknown tokens
                result.push_back(vocab.at("<unk>"));
            }
        }
        
        return result;
    }
    
    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        for (int token : tokens) {
            if (!is_valid_token(token)) {
                throw std::runtime_error("Invalid token ID: " + std::to_string(token) + 
                                       " (vocab size: " + std::to_string(vocab.size()) + ")");
            }
            
            // Find token string from ID
            for (const auto& [token_str, id] : vocab) {
                if (id == token) {
                    result += token_str;
                    break;
                }
            }
        }
        return result;
    }
    
    size_t vocab_size() const { 
        return vocab.size(); 
    }

    // Add alias method for API consistency
    size_t get_vocab_size() const {
        return vocab_size();
    }

    bool is_valid_token(int token_id) const {
        return token_id >= 0 && static_cast<size_t>(token_id) < vocab.size();
    }
};

} // namespace tiktoken 