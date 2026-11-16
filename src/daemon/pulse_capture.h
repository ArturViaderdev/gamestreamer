// Captura de audio con libpulse-simple cargada con dlopen (sin dependencias de
// compilación). Funciona con PulseAudio y con PipeWire (pipewire-pulse).
// Entrega float32 estéreo intercalado a 48 kHz.
#pragma once
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <string>

class PulseCapture {
public:
    ~PulseCapture() { close(); }

    // dev: "auto" o el nombre de una fuente de "pactl list short sources".
    // mic=false: "auto" = lo que suena por la salida predeterminada (monitor)
    // mic=true:  "auto" = el micrófono predeterminado
    bool open(const std::string& dev, std::string& err, bool mic = false) {
        if (!lib_) {
            lib_ = dlopen("libpulse-simple.so.0", RTLD_NOW);
            if (!lib_) { err = "no está libpulse-simple.so.0 (paquete libpulse0)"; return false; }
            new_ = (void* (*)(const char*, const char*, int, const char*, const char*, const SampleSpec*, const void*,
                              const BufferAttr*, int*))dlsym(lib_, "pa_simple_new");
            read_ = (int (*)(void*, void*, size_t, int*))dlsym(lib_, "pa_simple_read");
            free_ = (void (*)(void*))dlsym(lib_, "pa_simple_free");
            latency_ = (uint64_t (*)(void*, int*))dlsym(lib_, "pa_simple_get_latency");
            void* core = dlopen("libpulse.so.0", RTLD_NOW);
            if (core) strerror_ = (const char* (*)(int))dlsym(core, "pa_strerror");
            if (!new_ || !read_ || !free_) { err = "libpulse-simple incompleta"; return false; }
        }
        device = dev != "auto" ? dev : mic ? defaultSource() : defaultMonitor();
        SampleSpec ss{5 /* PA_SAMPLE_FLOAT32LE */, 48000, 2};
        BufferAttr ba{(uint32_t)-1, (uint32_t)-1, (uint32_t)-1, (uint32_t)-1, 480 * 8 /* fragsize: 10 ms */};
        int e = 0;
        s_ = new_(nullptr, "gamestream", 2 /* PA_STREAM_RECORD */, device.c_str(), mic ? "micrófono" : "audio del juego",
                  &ss, nullptr, &ba, &e);
        if (!s_) {
            err = "no se pudo abrir '" + device + "': " + (strerror_ ? strerror_(e) : std::to_string(e));
            return false;
        }
        return true;
    }

    // Lee exactamente 'frames' muestras estéreo (bloquea ~frames/48 ms)
    bool read(float* interleaved, int frames) {
        int e = 0;
        return s_ && read_(s_, interleaved, (size_t)frames * 8, &e) >= 0;
    }

    // Cuánto hace (µs) que se capturó la última muestra leída: retraso del
    // sistema de sonido hasta que nos llega. 0 si no se puede saber.
    uint64_t latencyUs() {
        if (!s_ || !latency_) return 0;
        int e = 0;
        uint64_t l = latency_(s_, &e);
        return l == (uint64_t)-1 || l > 500000 ? 0 : l;   // valores absurdos: no compensar
    }

    void close() {
        if (s_) free_(s_);
        s_ = nullptr;
    }

    std::string device;

private:
    struct SampleSpec { int format; uint32_t rate; uint8_t channels; };
    struct BufferAttr { uint32_t maxlength, tlength, prebuf, minreq, fragsize; };

    static std::string pactl(const char* what) {
        std::string r;
        std::string cmd = std::string("pactl ") + what + " 2>/dev/null";
        if (FILE* p = popen(cmd.c_str(), "r")) {
            char b[256];
            if (fgets(b, sizeof b, p)) r = b;
            pclose(p);
        }
        while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
        return r;
    }
    static std::string defaultMonitor() {
        std::string sink = pactl("get-default-sink");
        return sink.empty() ? "@DEFAULT_MONITOR@" : sink + ".monitor";
    }
    static std::string defaultSource() {
        std::string src = pactl("get-default-source");
        if (src.size() > 8 && src.compare(src.size() - 8, 8, ".monitor") == 0) src.clear();   // eso no es un micro
        return src.empty() ? "@DEFAULT_SOURCE@" : src;
    }

    void* lib_ = nullptr;
    void* s_ = nullptr;
    void* (*new_)(const char*, const char*, int, const char*, const char*, const SampleSpec*, const void*,
                  const BufferAttr*, int*) = nullptr;
    int (*read_)(void*, void*, size_t, int*) = nullptr;
    void (*free_)(void*) = nullptr;
    uint64_t (*latency_)(void*, int*) = nullptr;
    const char* (*strerror_)(int) = nullptr;
};
