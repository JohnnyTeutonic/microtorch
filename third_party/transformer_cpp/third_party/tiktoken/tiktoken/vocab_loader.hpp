#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace tiktoken {

struct VocabEntry {
    std::string token;
    int id;
    std::vector<uint8_t> bytes;
};

class VocabLoader {
public:
    static std::pair<std::unordered_map<std::string, int>, std::unordered_map<std::string, std::string>> 
    load_vocab_and_merges(const std::string& encoding_type) {
        std::string base_path = get_tiktoken_data_path();
        return load_vocab_and_merges_from_files(
            base_path + "/" + encoding_type + ".vocab",
            base_path + "/" + encoding_type + ".merges"
        );
    }

private:
    static std::string get_tiktoken_data_path() {
        // First check environment variable
        if (const char* env_p = std::getenv("TIKTOKEN_PATH")) {
            return std::string(env_p);
        }
        // Default to the directory containing the executable
        return std::filesystem::current_path().string() + "/tiktoken_data";
    }

    static std::pair<std::unordered_map<std::string, int>, std::unordered_map<std::string, std::string>>
    load_vocab_and_merges_from_files(const std::string& vocab_path, const std::string& merges_path) {
        std::unordered_map<std::string, int> vocab;
        std::unordered_map<std::string, std::string> merges;

        // Load vocabulary
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file) {
            throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
        }

        nlohmann::json vocab_json;
        vocab_file >> vocab_json;
        
        for (const auto& [token, id] : vocab_json.items()) {
            vocab[token] = id;
        }

        // Load merges
        std::ifstream merges_file(merges_path);
        if (!merges_file) {
            throw std::runtime_error("Failed to open merges file: " + merges_path);
        }

        std::string line;
        int merge_rank = 0;
        while (std::getline(merges_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string piece1, piece2;
            if (iss >> piece1 >> piece2) {
                std::string merge_key = piece1 + " " + piece2;
                merges[merge_key] = std::to_string(merge_rank++);
            }
        }

        return {vocab, merges};
    }
};

} // namespace tiktoken 