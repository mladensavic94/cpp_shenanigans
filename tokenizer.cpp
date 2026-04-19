#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <cmath>
#include <numeric>
#include <set>

class Token {
    public:
        std::string word;
        double cap        = 0;
        double pos        = 0;
        double frequency  = 0;
        double rel        = 0;
        double sent       = 0;
        double score      = 0;
        int    tf_upper   = 0; // occurrences where original was uppercase
        int    tf_acronym = 0; // occurrences where original was all-caps
        double tf_freq_norm = 0;
        std::vector<int> sentence_ids;
        std::unordered_set<std::string> left_ctx;
        std::unordered_set<std::string> right_ctx;

        bool operator==(const Token& other) const { return word == other.word; }
};

template<>
struct std::hash<Token> {
    size_t operator()(const Token& t) const {
        return std::hash<std::string>{}(t.word);
    }
};

const std::unordered_set<char> PUNCT = {
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', ';', '.', '/',
    ':', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '-'
};

std::unordered_set<std::string> load_stopwords(const std::string& path) {
    std::unordered_set<std::string> sw;
    std::ifstream file(path);
    std::string word;
    while (file >> word) sw.insert(word);
    return sw;
}

// Split text into sentences on '.', '!', '?'
std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string cur;
    for (char c : text) {
        if (c == '.' || c == '!' || c == '?') {
            if (!cur.empty()) { sentences.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) sentences.push_back(cur);
    return sentences;
}

// Tokenize a single sentence into raw words (preserving case)
std::vector<std::string> tokenize_sentence(const std::string& s) {
    std::vector<std::string> words;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) {
        std::erase_if(w, [](char c){ return PUNCT.count(c); });
        if (!w.empty()) words.push_back(w);
    }
    return words;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

bool is_upper(const std::string& s) { return !s.empty() && std::isupper(s[0]); }
bool is_acronym(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isupper);
}

// Build per-word token map with YAKE features
std::unordered_map<std::string, Token> build_tokens(
    const std::vector<std::string>& sentences,
    const std::unordered_set<std::string>& stop_words)
{
    std::unordered_map<std::string, Token> tokens;
    int total_sentences = sentences.size();

    for (int si = 0; si < (int)sentences.size(); ++si) {
        auto words = tokenize_sentence(sentences[si]);
        for (int wi = 0; wi < (int)words.size(); ++wi) {
            std::string low = to_lower(words[wi]);
            if (low.empty() || stop_words.count(low)) continue;

            auto& t = tokens[low];
            t.word = low;
            t.frequency += 1.0;
            t.sentence_ids.push_back(si);
            if (is_upper(words[wi]))   t.tf_upper++;
            if (is_acronym(words[wi])) t.tf_acronym++;

            // co-occurrence context (window = 1)
            if (wi > 0) {
                std::string left = to_lower(words[wi - 1]);
                if (!stop_words.count(left)) t.left_ctx.insert(left);
            }
            if (wi + 1 < (int)words.size()) {
                std::string right = to_lower(words[wi + 1]);
                if (!stop_words.count(right)) t.right_ctx.insert(right);
            }
        }
    }

    // compute mean and stddev of tf for frequency normalization
    double sum = 0, sq = 0;
    double max_tf = 0;
    for (auto& [w, t] : tokens) {
        sum += t.frequency;
        sq  += t.frequency * t.frequency;
        max_tf = std::max(max_tf, t.frequency);
    }
    double n    = tokens.size();
    double mean = sum / n;
    double std_dev = std::sqrt(sq / n - mean * mean);

    // compute sentence medians for position score
    // position of first sentence containing the word (1-indexed, lower = earlier)
    for (auto& [w, t] : tokens) {
        std::vector<int>& ids = t.sentence_ids;
        std::sort(ids.begin(), ids.end());
        double median_pos = ids[ids.size() / 2] + 1.0; // 1-indexed

        // YAKE feature formulas
        t.cap  = (t.tf_upper + t.tf_acronym) / t.frequency;
        t.pos  = std::log(std::log(3.0 + median_pos));
        t.tf_freq_norm = t.frequency / (mean + std_dev + 1e-9);
        t.rel  = 1.0 + ((double)t.left_ctx.size() + (double)t.right_ctx.size())
                       * (t.tf_freq_norm / (max_tf + 1e-9));
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        t.sent = (double)ids.size() / (double)total_sentences;

        double cap_safe = t.cap + 1e-9;
        t.score = (t.pos * cap_safe) / (t.tf_freq_norm + t.rel / cap_safe + t.sent / cap_safe);
    }

    return tokens;
}

// Score an n-gram candidate given its component word scores
double ngram_score(const std::vector<std::string>& words,
                   const std::unordered_map<std::string, Token>& tokens,
                   const std::unordered_set<std::string>& stop_words)
{
    double prod = 1.0, sum = 0.0;
    int    tf   = 0;
    for (auto& w : words) {
        std::string low = to_lower(w);
        if (stop_words.count(low)) continue;
        auto it = tokens.find(low);
        if (it == tokens.end()) continue;
        double s = it->second.score;
        prod *= s;
        sum  += s;
        tf++;
    }
    if (tf == 0) return 1e9;
    return prod / ((double)tf * (1.0 + sum));
}

// Extract top-k keyword candidates (n-grams up to max_ngram)
std::vector<std::pair<std::string, double>> yake(
    const std::string& text,
    const std::unordered_set<std::string>& stop_words,
    int max_ngram = 3,
    int top_k = 10)
{
    auto sentences = split_sentences(text);
    auto tokens    = build_tokens(sentences, stop_words);

    // collect all n-gram candidates from the text
    std::unordered_map<std::string, double> candidates;
    for (auto& sent : sentences) {
        auto words = tokenize_sentence(sent);
        for (int i = 0; i < (int)words.size(); ++i) {
            for (int n = 1; n <= max_ngram && i + n <= (int)words.size(); ++n) {
                std::vector<std::string> gram(words.begin() + i, words.begin() + i + n);
                std::string key;
                for (auto& w : gram) { if (!key.empty()) key += ' '; key += to_lower(w); }
                // skip if starts or ends with stopword
                if (stop_words.count(to_lower(gram.front())) ||
                    stop_words.count(to_lower(gram.back()))) continue;
                double sc = ngram_score(gram, tokens, stop_words);
                // keep lowest score seen for this candidate
                auto it = candidates.find(key);
                if (it == candidates.end() || sc < it->second)
                    candidates[key] = sc;
            }
        }
    }

    std::vector<std::pair<std::string, double>> ranked(candidates.begin(), candidates.end());
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b){ return a.second < b.second; });
    if ((int)ranked.size() > top_k) ranked.resize(top_k);
    return ranked;
}

int main() {
    auto stop_words = load_stopwords("stop-words.txt");
    std::string text =
            "Obsidian Bookshelf "
"I like keeping a record of the books I’ve read over the years in addition to some notes and thoughts I gather while reading them. Since adopting Obsidian and Zettelkasten I’ve been searching for the best way to organize these records and notes into a unified index page. This index page should show a list of books I’ve read in chronological order, noting their current status (read, reading, todo, ditched) in addition to showing when I started/finished them."
"Given that Obsidian is largely a repository of static Markdown files, creating such an index seemed like a daunting task. In the past I’ve written python scripts that find notes labeled as “books”, extract some metadata, and then output a markdown index page. These work, but it’s not quite ideal because it requires running the script each time you edit a file’s contents."
"Enter the Obsidian Dataview plugin. This addon allows for generating index pages based on Markdown files (and their frontmatter) within your vault. Even better, it exposes a Javascript API that allows for some really in-depth customization of how these index pages are displayed."
"After quite a bit of tinkering, I ended up with this:"
"A screenshot of the bookmark"
"You can find the source code which generates this index here. Remember that you’ll need to wrap the code in a dataview block in order for it to get picked up."
"Each of the pages this index links to begin with some yaml frontmatter in order to assign the metadata:"
"----"
"alias: The Dispossessed"
"author: Ursula K. Ke Guin"
"status: read"
"started: 2022-01-01"
"completed: 2022-01-15"
"tags:"
"- books"
"- fiction"
"- political"
"---"
"The most important of this metadata is the books tag, which allows the Dataview to find the markdown file even though I store it as a standard Zettelkasten note. I try to keep my Obsidian vault entirely flat, that is, there are no folder hierarchies to be found. Everything is organized via links between notes and occasionally tags. This is a pretty standard practice from Zettelkasten which you can learn more about here."
";";
    auto keywords = yake(text, stop_words, 3, 10);
    std::cout << "YAKE keywords:\n";
    for (auto& [kw, sc] : keywords)
        std::cout << "  " << kw << "  (" << sc << ")\n";

    return 0;
}
