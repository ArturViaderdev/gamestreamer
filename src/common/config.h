// Lector mínimo de ~/.config/gamestream/gamestream.conf (formato clave = valor)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace gs {

inline std::string configPath() {
    if (const char* p = getenv("GAMESTREAM_CONFIG")) return p;
    std::string base;
    if (const char* x = getenv("XDG_CONFIG_HOME"); x && *x) base = x;
    else if (const char* h = getenv("HOME")) base = std::string(h) + "/.config";
    else base = "/tmp";
    return base + "/gamestream/gamestream.conf";
}

inline std::string trim(std::string s) {
    // quita también espacios "raros" pegados desde webs (NBSP U+00A0) y el BOM
    for (const char* bad : {"\xc2\xa0", "\xef\xbb\xbf"}) {
        size_t p;
        while ((p = s.find(bad)) != std::string::npos) s.replace(p, strlen(bad), " ");
    }
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// "YouTube Key", "youtube-key", "youtube_key" -> "youtube_key"
inline std::string normKey(std::string k) {
    std::string r;
    for (char c : trim(k)) {
        if (c == ' ' || c == '-' || c == '\t') c = '_';
        r += (char)tolower((unsigned char)c);
    }
    return r;
}

inline std::map<std::string, std::string> loadConfig(const std::string& path) {
    std::map<std::string, std::string> m;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return m;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        std::string l = trim(line);
        if (l.empty() || l[0] == '#' || l[0] == ';') continue;
        size_t eq = l.find('=');
        if (eq == std::string::npos) eq = l.find(':');   // también "clave: valor"
        if (eq == std::string::npos) continue;
        std::string k = normKey(l.substr(0, eq)), v = trim(l.substr(eq + 1));
        size_t hash = v.find(" #");                       // comentario al final de la línea
        if (hash != std::string::npos) v = trim(v.substr(0, hash));
        if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) v = v.substr(1, v.size() - 2);
        m[k] = v;
    }
    fclose(f);
    return m;
}

} // namespace gs
