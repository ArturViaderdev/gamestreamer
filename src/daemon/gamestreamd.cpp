// gamestreamd - daemon de emisión.
//
//  * Escucha a las capas Vulkan de los juegos en un socket UNIX abstracto.
//  * Importa en CUDA la memoria de vídeo que exporta la capa (sin pasar por la
//    CPU), copia el frame a un surface de NVENC (copia VRAM->VRAM, ~0,05 ms) y
//    lo codifica en H.264/HEVC con el bloque NVENC de la GPU.
//  * Captura el audio del escritorio (PulseAudio / PipeWire-pulse) y lo
//    codifica en AAC.
//  * Mezcla en FLV y lo envía por RTMP a Twitch / YouTube (o a un fichero).
//
// Uso:  gamestreamd            -> arranca el daemon
//       gamestreamd toggle     -> empieza/para la emisión (para atajos del escritorio)
//       gamestreamd start|stop|status|quit

#include "dynlink_cuda.h"   // third_party/nvenc (cabeceras de NVIDIA, MIT)
#include "nvEncodeAPI.h"    // API NVENC 12.0: vale para cualquier driver >= 520

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
}

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <type_traits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "../common/config.h"
#include "../common/protocol.h"
#include "pulse_capture.h"
#include "audio_sync.h"

#ifndef CUDA_EXTERNAL_MEMORY_DEDICATED
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#endif

// Log a stderr y a ~/.local/state/gamestream/gamestreamd.log (siempre en el mismo sitio)
static FILE* g_logFile = nullptr;
static std::mutex g_logMtx;
static std::string g_lastError;   // último mensaje, para enseñarlo en la notificación

static void openLogFile() {
    std::string dir;
    if (const char* x = getenv("XDG_STATE_HOME"); x && *x) dir = x;
    else if (const char* h = getenv("HOME")) dir = std::string(h) + "/.local/state";
    else return;
    dir += "/gamestream";
    if (system(("mkdir -p '" + dir + "'").c_str())) {}
    std::string path = dir + "/gamestreamd.log";
    g_logFile = fopen(path.c_str(), "a");
    if (g_logFile) setvbuf(g_logFile, nullptr, _IOLBF, 0);
}

__attribute__((format(printf, 1, 2))) static void gslog(const char* fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    time_t t = time(nullptr);
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&t));
    std::lock_guard<std::mutex> l(g_logMtx);
    fprintf(stderr, "[gamestreamd] %s\n", msg);
    if (g_logFile) fprintf(g_logFile, "%s %s\n", ts, msg);
    g_lastError = msg;
}
#define LOG(...) gslog(__VA_ARGS__)

static std::string lastError() {
    std::lock_guard<std::mutex> l(g_logMtx);
    return g_lastError;
}

// Los errores internos de FFmpeg/NVENC también al log (son los que explican los fallos)
static void ffLogCb(void* avcl, int level, const char* fmt, va_list vl) {
    if (level > av_log_get_level()) return;
    char line[1024];
    int prefix = 1;
    av_log_format_line2(avcl, level, fmt, vl, line, sizeof line, &prefix);
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    if (n) gslog("ffmpeg: %s", line);
}

static uint64_t nowUs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000;
}

static std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(e, b, sizeof b);
    return b;
}

// ---------------------------------------------------------------------------
// Configuración
// ---------------------------------------------------------------------------
struct Dest {
    std::string name;   // "Twitch", "YouTube", "Grabación"...
    std::string url;    // rtmp://... o ruta de fichero
    int bitrate;        // kbps
    uint32_t w = 0, h = 0;      // resolución de este destino (0x0 = la del juego)
    std::string codec = "h264";
    int stream = 0;             // índice de la salida de la capa que usa (misma resolución = misma salida)
};

struct Config {
    std::vector<Dest> dests;  // a dónde se emite (uno o varios a la vez)
    std::vector<std::pair<uint32_t, uint32_t>> streams;   // resoluciones distintas que se piden a la capa
    uint32_t width = 1920, height = 1080;   // 0x0 = resolución del juego
    uint32_t fps = 60;
    std::string codec = "h264";
    int bitrate = 6000;       // kbps por defecto
    int keyint = 2;           // segundos entre keyframes
    std::string preset = "p4";
    std::string audio = "auto";
    int audioBitrate = 160;
    int audioDelayMs = 0;     // >0 retrasa el audio respecto al vídeo
    int desktopVolume = 100;  // %
    bool microphone = false;
    std::string micDevice = "auto";
    int micVolume = 100;      // %
    // cámara
    bool camera = false;
    std::string camDevice = "/dev/video0";
    int camCapW = 640, camCapH = 480, camFps = 30;
    int camSizePct = 22;      // ancho de la cámara en % del ancho del stream
    std::string camPosition = "abajo-derecha";
    int camMargin = 20;       // px
    int camBorder = 3;        // px (0 = sin borde)
    bool camMirror = true;
    bool notify = true;
};

static std::string expandHome(const std::string& p) {
    if (p.rfind("~/", 0) == 0)
        if (const char* h = getenv("HOME")) return std::string(h) + p.substr(1);
    return p;
}

static Config loadConfig() {
    Config c;
    auto m = gs::loadConfig(gs::configPath());
    auto get = [&](const char* k, const std::string& def) { return m.count(k) && !m[k].empty() ? m[k] : def; };
    auto num = [&](const char* k, int def) { return m.count(k) && !m[k].empty() ? atoi(m[k].c_str()) : def; };

    c.bitrate = num("bitrate", 6000);
    c.codec = get("codec", "h264");
    auto parseRes = [](const std::string& v, uint32_t& w, uint32_t& h) {
        if (v == "auto" || v == "native" || v == "nativa") { w = h = 0; return; }
        if (sscanf(v.c_str(), "%ux%u", &w, &h) != 2) { w = 1920; h = 1080; }
    };
    parseRes(get("resolution", "1920x1080"), c.width, c.height);
    // Cada destino puede tener su resolución y su códec (por defecto, los globales)
    auto dest = [&](const std::string& name, const std::string& prefix, const std::string& url, int bitrate) {
        Dest d;
        d.name = name;
        d.url = url;
        d.bitrate = bitrate;
        d.w = c.width; d.h = c.height;
        std::string r = get((prefix + "_resolution").c_str(), "");
        if (!r.empty()) parseRes(r, d.w, d.h);
        d.codec = get((prefix + "_codec").c_str(), c.codec);
        if (d.codec != "h264" && d.codec != "hevc") d.codec = "h264";
        return d;
    };

    std::string service = get("service", "custom");
    for (auto& ch : service) ch = (char)tolower((unsigned char)ch);
    const bool both = service == "ambos" || service == "both" || service == "todos" || service == "all";
    const bool tw = both || service.find("twitch") != std::string::npos;
    const bool yt = both || service.find("youtube") != std::string::npos;
    const std::string key = get("key", "");   // compatibilidad: clave única de versiones anteriores

    if (tw) {
        std::string k = get("twitch_key", (tw && !yt) ? key : "");
        std::string srv = get("twitch_server", (tw && !yt) ? get("server", "") : "");
        if (srv.empty()) srv = "rtmp://ingest.global-contribute.live-video.net/app";
        if (k.empty()) LOG("AVISO: falta twitch_key en la configuración");
        else {
            Dest d = dest("Twitch", "twitch", srv + "/" + k, num("twitch_bitrate", c.bitrate));
            if (d.codec != "h264") {   // Twitch sólo admite H.264 por RTMP
                if (m.count("twitch_codec")) LOG("AVISO: Twitch sólo admite H.264 por RTMP: se usa h264 para Twitch");
                d.codec = "h264";
            }
            c.dests.push_back(d);
        }
    }
    if (yt) {
        std::string k = get("youtube_key", (yt && !tw) ? key : "");
        std::string srv = get("youtube_server", (yt && !tw) ? get("server", "") : "");
        if (srv.empty()) srv = "rtmp://a.rtmp.youtube.com/live2";
        if (k.empty()) LOG("AVISO: falta youtube_key en la configuración");
        else c.dests.push_back(dest("YouTube", "youtube", srv + "/" + k, num("youtube_bitrate", c.bitrate)));
    }
    if (!tw && !yt) {
        std::string url = get("url", "");
        if (!url.empty()) c.dests.push_back(dest("Destino", "custom", expandHome(url), c.bitrate));
    }
    std::string rec = get("record", "");
    if (!rec.empty()) {
        // Por defecto la grabación copia al destino de más calidad: así reutiliza su codificación
        Dest best = dest("Grabación", "record", expandHome(rec), c.bitrate);
        const Dest* top = nullptr;
        for (auto& d : c.dests)
            if (!top || (uint64_t)d.w * d.h > (uint64_t)top->w * top->h ||
                ((uint64_t)d.w * d.h == (uint64_t)top->w * top->h && d.bitrate > top->bitrate)) top = &d;
        if (top) {
            if (!m.count("record_resolution")) { best.w = top->w; best.h = top->h; }
            if (!m.count("record_codec")) best.codec = top->codec;
            best.bitrate = top->bitrate;
        }
        best.bitrate = num("record_bitrate", best.bitrate);
        c.dests.push_back(best);
    }
    // Resoluciones distintas -> salidas de la capa (la misma resolución se comparte)
    for (auto& d : c.dests) {
        int idx = -1;
        for (size_t i = 0; i < c.streams.size(); i++)
            if (c.streams[i].first == d.w && c.streams[i].second == d.h) idx = (int)i;
        if (idx < 0 && c.streams.size() < GS_MAX_STREAMS) {
            c.streams.push_back({d.w, d.h});
            idx = (int)c.streams.size() - 1;
        }
        d.stream = idx < 0 ? 0 : idx;
    }

    c.fps = (uint32_t)num("fps", 60);
    if (c.fps < 1 || c.fps > 240) c.fps = 60;
    c.keyint = num("keyint", 2);
    c.preset = get("preset", "p4");
    c.audio = get("audio", "auto");
    c.audioBitrate = num("audio_bitrate", 160);
    c.audioDelayMs = num("audio_delay", 0);
    auto yes = [&](const char* k, bool def) {
        std::string v = get(k, def ? "yes" : "no");
        for (auto& ch : v) ch = (char)tolower((unsigned char)ch);
        return v == "yes" || v == "si" || v == "sí" || v == "true" || v == "1" || v == "on";
    };
    c.desktopVolume = num("audio_volume", 100);
    c.microphone = yes("microphone", false);
    c.micDevice = get("microphone_device", "auto");
    c.micVolume = num("microphone_volume", 100);
    c.camera = yes("camera", false);
    c.camDevice = get("camera_device", "/dev/video0");
    if (sscanf(get("camera_capture", "640x480").c_str(), "%dx%d", &c.camCapW, &c.camCapH) != 2) c.camCapW = 640, c.camCapH = 480;
    c.camFps = num("camera_fps", 30);
    c.camSizePct = std::clamp(num("camera_size", 22), 5, 60);
    c.camPosition = get("camera_position", "abajo-derecha");
    c.camMargin = std::max(0, num("camera_margin", 20));
    c.camBorder = std::max(0, num("camera_border", 3));
    c.camMirror = yes("camera_mirror", true);
    c.notify = get("notify", "yes") != "no";
    return c;
}

static void notify(const Config& c, const std::string& msg) {
    LOG("%s", msg.c_str());
    if (!c.notify) return;
    std::string safe;
    for (char ch : msg) safe += ch == '\'' ? '"' : ch;
    if (safe.size() > 200) safe = safe.substr(0, 200) + "...";
    std::string cmd = "notify-send -a gamestream -u low -t 4000 'gamestream' '" + safe + "' >/dev/null 2>&1 &";
    if (system(cmd.c_str())) {}
}

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------
struct Cuda {
    void* lib = nullptr;
    tcuInit* cuInit = nullptr;
    tcuDeviceGetCount* cuDeviceGetCount = nullptr;
    tcuDeviceGet* cuDeviceGet = nullptr;
    tcuDeviceGetUuid* cuDeviceGetUuid = nullptr;
    tcuDevicePrimaryCtxRetain* cuDevicePrimaryCtxRetain = nullptr;
    tcuDevicePrimaryCtxRelease* cuDevicePrimaryCtxRelease = nullptr;
    tcuCtxPushCurrent_v2* cuCtxPushCurrent = nullptr;
    tcuCtxPopCurrent_v2* cuCtxPopCurrent = nullptr;
    tcuMemAllocPitch_v2* cuMemAllocPitch = nullptr;
    tcuMemFree_v2* cuMemFree = nullptr;
    tcuMemcpy2DAsync_v2* cuMemcpy2DAsync = nullptr;
    tcuStreamSynchronize* cuStreamSynchronize = nullptr;
    tcuImportExternalMemory* cuImportExternalMemory = nullptr;
    tcuExternalMemoryGetMappedBuffer* cuExternalMemoryGetMappedBuffer = nullptr;
    tcuDestroyExternalMemory* cuDestroyExternalMemory = nullptr;
    tcuGetErrorName* cuGetErrorName = nullptr;

    CUcontext ctx = nullptr;
    CUdevice dev = 0;
    uint8_t uuid[16]{};

    bool load() {
        if (lib) return true;
        lib = dlopen("libcuda.so.1", RTLD_NOW);
        if (!lib) { LOG("no se pudo cargar libcuda.so.1 (¿driver NVIDIA?)"); return false; }
        bool ok = true;
        auto sym = [&](auto& fn, const char* name) {
            fn = (std::remove_reference_t<decltype(fn)>)dlsym(lib, name);
            if (!fn) { LOG("libcuda sin %s", name); ok = false; }
        };
        sym(cuInit, "cuInit");
        sym(cuDeviceGetCount, "cuDeviceGetCount");
        sym(cuDeviceGet, "cuDeviceGet");
        sym(cuDeviceGetUuid, "cuDeviceGetUuid");
        sym(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        sym(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease");
        sym(cuCtxPushCurrent, "cuCtxPushCurrent_v2");
        sym(cuCtxPopCurrent, "cuCtxPopCurrent_v2");
        sym(cuMemAllocPitch, "cuMemAllocPitch_v2");
        sym(cuMemFree, "cuMemFree_v2");
        sym(cuMemcpy2DAsync, "cuMemcpy2DAsync_v2");
        sym(cuStreamSynchronize, "cuStreamSynchronize");
        sym(cuImportExternalMemory, "cuImportExternalMemory");
        sym(cuExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
        sym(cuDestroyExternalMemory, "cuDestroyExternalMemory");
        sym(cuGetErrorName, "cuGetErrorName");
        if (!ok) { dlclose(lib); lib = nullptr; }
        return ok;
    }

    const char* err(CUresult r) {
        const char* n = nullptr;
        if (cuGetErrorName) cuGetErrorName(r, &n);
        return n ? n : "?";
    }

    bool ensure(bool hasUuid, const uint8_t* want) {
        if (ctx && (!hasUuid || !memcmp(uuid, want, 16))) return true;
        if (!load()) return false;
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { LOG("cuInit: %s", err(r)); return false; }
        int n = 0, idx = 0;
        cuDeviceGetCount(&n);
        if (hasUuid) {
            idx = -1;
            for (int i = 0; i < n; i++) {
                CUdevice d;
                CUuuid u;
                cuDeviceGet(&d, i);
                if (cuDeviceGetUuid(&u, d) == CUDA_SUCCESS && !memcmp(u.bytes, want, 16)) { idx = i; break; }
            }
            if (idx < 0) { LOG("el juego no se ejecuta en una GPU NVIDIA visible por CUDA"); return false; }
            memcpy(uuid, want, 16);
        }
        if (ctx) { cuDevicePrimaryCtxRelease(dev); ctx = nullptr; }
        cuDeviceGet(&dev, idx);
        if ((r = cuDevicePrimaryCtxRetain(&ctx, dev)) != CUDA_SUCCESS) { LOG("contexto CUDA: %s", err(r)); ctx = nullptr; return false; }
        LOG("CUDA listo en GPU %d", idx);
        return true;
    }
    void push() { cuCtxPushCurrent(ctx); }
    void pop() { CUcontext d; cuCtxPopCurrent(&d); }
};
static Cuda g_cuda;


// ---------------------------------------------------------------------------
// Punto de extensión para superposiciones (cámara, logos, alertas...).
// Se ejecuta sobre el frame ya copiado a la superficie de NVENC (BGRA en VRAM),
// en el contexto CUDA activo, justo antes de codificar. Una cámara pequeña en
// una esquina será: capturar V4L2 -> convertir a BGRA a tamaño reducido ->
// subir a un buffer CUDA -> cuMemcpy2DAsync al rectángulo de la esquina.
// ---------------------------------------------------------------------------
struct Overlay {
    virtual ~Overlay() = default;
    virtual void apply(CUdeviceptr frame, size_t pitch, int w, int h, CUstream stream) = 0;
};

#ifdef GS_FAKE_ENC
// SÓLO PRUEBAS: codificador x264 por software que imita a NvEnc (sin GPU)
class NvEnc {
public:
    std::vector<uint8_t> header;
    bool open(const Config& cfg, int w, int h) {
        const AVCodec* c = avcodec_find_encoder_by_name("libx264");
        ctx_ = avcodec_alloc_context3(c);
        ctx_->width = w; ctx_->height = h; ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
        ctx_->time_base = {1, 1000000}; ctx_->framerate = {(int)cfg.fps, 1};
        ctx_->bit_rate = (int64_t)cfg.bitrate * 1000; ctx_->gop_size = cfg.fps * 2; ctx_->max_b_frames = 0;
        ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        av_opt_set(ctx_->priv_data, "preset", "ultrafast", 0);
        av_opt_set(ctx_->priv_data, "tune", "zerolatency", 0);
        if (avcodec_open2(ctx_, c, nullptr) < 0) return false;
        header.assign(ctx_->extradata, ctx_->extradata + ctx_->extradata_size);
        return true;
    }
    void upload(CUdeviceptr, size_t, const std::vector<std::unique_ptr<Overlay>>&) {}
    bool encodeNow(int64_t pts, std::vector<uint8_t>& out, bool& key) {
        AVFrame* f = av_frame_alloc();
        f->format = AV_PIX_FMT_YUV420P; f->width = ctx_->width; f->height = ctx_->height;
        av_frame_get_buffer(f, 0);
        for (int y = 0; y < f->height; y++) memset(f->data[0] + y * f->linesize[0], (int)((y + n_ * 4) & 255), f->width);
        for (int y = 0; y < f->height / 2; y++) { memset(f->data[1] + y * f->linesize[1], 128, f->width / 2); memset(f->data[2] + y * f->linesize[2], 90, f->width / 2); }
        f->pts = pts; n_++;
        avcodec_send_frame(ctx_, f); av_frame_free(&f);
        AVPacket* p = av_packet_alloc(); out.clear();
        if (avcodec_receive_packet(ctx_, p) >= 0) { out.assign(p->data, p->data + p->size); key = p->flags & AV_PKT_FLAG_KEY; }
        av_packet_free(&p);
        return true;
    }
    void close() { avcodec_free_context(&ctx_); }
private:
    AVCodecContext* ctx_ = nullptr;
    int n_ = 0;
};
#else
// ---------------------------------------------------------------------------
// NVENC directo (sin el codificador de FFmpeg).
// Se compila con la API 12.0 y se comprueba al arrancar qué admite el driver:
// así no dependemos de la versión de nv-codec-headers con que se compiló el
// FFmpeg de la distro (que puede exigir un driver más nuevo que el instalado).
// ---------------------------------------------------------------------------
class NvEnc {
public:
    std::vector<uint8_t> header;   // SPS/PPS (Annex B)

    bool open(const Config& cfg, int w, int h) {
        w_ = w; h_ = h;
        lib_ = dlopen("libnvidia-encode.so.1", RTLD_NOW);
        if (!lib_) { LOG("no se pudo cargar libnvidia-encode.so.1 (¿driver NVIDIA completo?)"); return false; }
        auto getMax = (NVENCSTATUS(NVENCAPI*)(uint32_t*))dlsym(lib_, "NvEncodeAPIGetMaxSupportedVersion");
        auto create = (NVENCSTATUS(NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*))dlsym(lib_, "NvEncodeAPICreateInstance");
        if (!getMax || !create) { LOG("libnvidia-encode incompleta"); return false; }
        uint32_t maxv = 0;
        getMax(&maxv);
        const uint32_t need = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
        LOG("NVENC: el driver admite la API %u.%u (usamos %u.%u)", maxv >> 4, maxv & 15, need >> 4, need & 15);
        if (maxv < need) { LOG("driver NVIDIA demasiado antiguo para NVENC 12.0 (hace falta >= 520)"); return false; }
        fl_ = {};
        fl_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        if (create(&fl_) != NV_ENC_SUCCESS) { LOG("NvEncodeAPICreateInstance falló"); return false; }

        g_cuda.push();
        bool ok = setup(cfg);
        g_cuda.pop();
        if (!ok) close();
        return ok;
    }

    // Copia src (BGRA en VRAM) a la superficie de NVENC, avisa con released()
    // y codifica. Devuelve false si hay un error grave.
    // Copia src (BGRA en VRAM) a la superficie de NVENC y aplica overlays.
    // Asíncrono: quien llama sincroniza con cuStreamSynchronize.
    void upload(CUdeviceptr src, size_t srcPitch, const std::vector<std::unique_ptr<Overlay>>& overlays);
    // Codifica la superficie. out vacío = no hay paquete todavía.
    bool encodeNow(int64_t ptsUs, std::vector<uint8_t>& out, bool& key);

    void close() {
        if (enc_) {
            g_cuda.push();
            if (reg_) fl_.nvEncUnregisterResource(enc_, reg_);
            if (bs_) fl_.nvEncDestroyBitstreamBuffer(enc_, bs_);
            fl_.nvEncDestroyEncoder(enc_);
            if (buf_) g_cuda.cuMemFree(buf_);
            g_cuda.pop();
        }
        enc_ = nullptr; reg_ = nullptr; bs_ = nullptr; buf_ = 0;
        header.clear();
        frame_ = 0;
    }

    CUdeviceptr surface() const { return buf_; }
    size_t pitch() const { return pitch_; }

private:
    static GUID presetGuid(const std::string& p) {
        if (p == "p1") return NV_ENC_PRESET_P1_GUID;
        if (p == "p2") return NV_ENC_PRESET_P2_GUID;
        if (p == "p3") return NV_ENC_PRESET_P3_GUID;
        if (p == "p5") return NV_ENC_PRESET_P5_GUID;
        if (p == "p6") return NV_ENC_PRESET_P6_GUID;
        if (p == "p7") return NV_ENC_PRESET_P7_GUID;
        return NV_ENC_PRESET_P4_GUID;
    }
    static void vui(NV_ENC_CONFIG_H264_VUI_PARAMETERS& v) {
        // NVENC convierte BGRA->YUV con matriz BT.601 en rango limitado
        v.videoSignalTypePresentFlag = 1;
        v.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        v.videoFullRangeFlag = 0;
        v.colourDescriptionPresentFlag = 1;
        v.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        v.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        v.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT470BG;
    }

    bool check(NVENCSTATUS st, const char* what) {
        if (st == NV_ENC_SUCCESS) return true;
        const char* msg = enc_ && fl_.nvEncGetLastErrorString ? fl_.nvEncGetLastErrorString(enc_) : "";
        LOG("NVENC %s: error %d %s", what, (int)st, msg ? msg : "");
        return false;
    }

    bool setup(const Config& cfg) {
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op{};
        op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        op.device = g_cuda.ctx;
        op.apiVersion = NVENCAPI_VERSION;
        if (!check(fl_.nvEncOpenEncodeSessionEx(&op, &enc_), "abrir sesión")) { enc_ = nullptr; return false; }

        hevc_ = cfg.codec == "hevc";
        GUID codec = hevc_ ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
        GUID preset = presetGuid(cfg.preset);
        NV_ENC_PRESET_CONFIG pc{};
        pc.version = NV_ENC_PRESET_CONFIG_VER;
        pc.presetCfg.version = NV_ENC_CONFIG_VER;
        if (!check(fl_.nvEncGetEncodePresetConfigEx(enc_, codec, preset, NV_ENC_TUNING_INFO_LOW_LATENCY, &pc), "preset"))
            return false;
        NV_ENC_CONFIG c = pc.presetCfg;
        const uint32_t gop = cfg.fps * (cfg.keyint > 0 ? cfg.keyint : 2);
        const uint32_t bps = (uint32_t)cfg.bitrate * 1000;
        c.gopLength = gop;
        c.frameIntervalP = 1;   // sin B-frames
        c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        c.rcParams.averageBitRate = bps;
        c.rcParams.maxBitRate = bps;
        c.rcParams.vbvBufferSize = bps;
        c.rcParams.vbvInitialDelay = bps;
        c.rcParams.zeroReorderDelay = 1;
        if (hevc_) {
            c.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
            c.encodeCodecConfig.hevcConfig.idrPeriod = gop;
            c.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
            vui(c.encodeCodecConfig.hevcConfig.hevcVUIParameters);
        } else {
            c.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
            c.encodeCodecConfig.h264Config.idrPeriod = gop;
            c.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
            vui(c.encodeCodecConfig.h264Config.h264VUIParameters);
        }

        NV_ENC_INITIALIZE_PARAMS ip{};
        ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
        ip.encodeGUID = codec;
        ip.presetGUID = preset;
        ip.encodeWidth = ip.darWidth = ip.maxEncodeWidth = (uint32_t)w_;
        ip.encodeHeight = ip.darHeight = ip.maxEncodeHeight = (uint32_t)h_;
        ip.frameRateNum = cfg.fps;
        ip.frameRateDen = 1;
        ip.enablePTD = 1;
        ip.encodeConfig = &c;
        ip.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
        if (!check(fl_.nvEncInitializeEncoder(enc_, &ip), "inicializar")) return false;

        NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
        cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        if (!check(fl_.nvEncCreateBitstreamBuffer(enc_, &cb), "bitstream")) return false;
        bs_ = cb.bitstreamBuffer;

        CUresult r = g_cuda.cuMemAllocPitch(&buf_, &pitch_, (size_t)w_ * 4, (size_t)h_, 16);
        if (r != CUDA_SUCCESS) { LOG("cuMemAllocPitch: %s", g_cuda.err(r)); buf_ = 0; return false; }

        NV_ENC_REGISTER_RESOURCE rr{};
        rr.version = NV_ENC_REGISTER_RESOURCE_VER;
        rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        rr.width = (uint32_t)w_;
        rr.height = (uint32_t)h_;
        rr.pitch = (uint32_t)pitch_;
        rr.resourceToRegister = (void*)buf_;
        rr.bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;   // en memoria: B,G,R,A = lo que entrega la capa
        rr.bufferUsage = NV_ENC_INPUT_IMAGE;
        if (!check(fl_.nvEncRegisterResource(enc_, &rr), "registrar superficie")) return false;
        reg_ = rr.registeredResource;

        uint8_t tmp[1024];
        uint32_t sz = 0;
        NV_ENC_SEQUENCE_PARAM_PAYLOAD sp{};
        sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        sp.inBufferSize = sizeof tmp;
        sp.spsppsBuffer = tmp;
        sp.outSPSPPSPayloadSize = &sz;
        if (!check(fl_.nvEncGetSequenceParams(enc_, &sp), "cabecera SPS/PPS")) return false;
        header.assign(tmp, tmp + sz);
        return true;
    }

    void* lib_ = nullptr;
    NV_ENCODE_API_FUNCTION_LIST fl_{};
    void* enc_ = nullptr;
    NV_ENC_OUTPUT_PTR bs_ = nullptr;
    NV_ENC_REGISTERED_PTR reg_ = nullptr;
    CUdeviceptr buf_ = 0;
    size_t pitch_ = 0;
    int w_ = 0, h_ = 0;
    bool hevc_ = false;
    uint32_t frame_ = 0;
};

void NvEnc::upload(CUdeviceptr src, size_t srcPitch, const std::vector<std::unique_ptr<Overlay>>& overlays) {
    g_cuda.push();
    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = src;
    cp.srcPitch = srcPitch;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = buf_;
    cp.dstPitch = pitch_;
    cp.WidthInBytes = (size_t)w_ * 4;
    cp.Height = (size_t)h_;
    g_cuda.cuMemcpy2DAsync(&cp, nullptr);
    for (auto& o : overlays) o->apply(buf_, pitch_, w_, h_, nullptr);
    g_cuda.pop();
}

bool NvEnc::encodeNow(int64_t ptsUs, std::vector<uint8_t>& out, bool& key) {
    g_cuda.push();
    bool ok = false;
    NV_ENC_MAP_INPUT_RESOURCE mr{};
    mr.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    mr.registeredResource = reg_;
    if (check(fl_.nvEncMapInputResource(enc_, &mr), "map")) {
        NV_ENC_PIC_PARAMS pp{};
        pp.version = NV_ENC_PIC_PARAMS_VER;
        pp.inputWidth = (uint32_t)w_;
        pp.inputHeight = (uint32_t)h_;
        pp.inputPitch = (uint32_t)pitch_;
        pp.inputBuffer = mr.mappedResource;
        pp.bufferFmt = mr.mappedBufferFmt;
        pp.outputBitstream = bs_;
        pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        pp.inputTimeStamp = (uint64_t)ptsUs;
        pp.frameIdx = frame_;
        if (frame_ == 0) pp.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        frame_++;
        NVENCSTATUS st = fl_.nvEncEncodePicture(enc_, &pp);
        if (st == NV_ENC_SUCCESS) {
            NV_ENC_LOCK_BITSTREAM lb{};
            lb.version = NV_ENC_LOCK_BITSTREAM_VER;
            lb.outputBitstream = bs_;
            if (check(fl_.nvEncLockBitstream(enc_, &lb), "lock")) {
                out.assign((uint8_t*)lb.bitstreamBufferPtr, (uint8_t*)lb.bitstreamBufferPtr + lb.bitstreamSizeInBytes);
                key = lb.pictureType == NV_ENC_PIC_TYPE_IDR || lb.pictureType == NV_ENC_PIC_TYPE_I;
                fl_.nvEncUnlockBitstream(enc_, bs_);
                ok = true;
            }
        } else if (st == NV_ENC_ERR_NEED_MORE_INPUT) {
            out.clear();
            ok = true;
        } else {
            check(st, "codificar");
        }
        fl_.nvEncUnmapInputResource(enc_, mr.mappedResource);
    }
    g_cuda.pop();
    return ok;
}
#endif

// ---------------------------------------------------------------------------
// Salida: encoders + muxer + hilo de red + hilo de audio
// ---------------------------------------------------------------------------
#include "camera.h"

// Un destino de emisión: su propio muxer, conexión, cola e hilo de red.
// Si uno falla o va lento, los demás siguen.
struct Sink {
    std::string name, url;
    int enc = 0;   // qué codificador (tasa de bits) usa
    AVFormatContext* oc = nullptr;
    AVStream* vst = nullptr;
    AVStream* ast = nullptr;
    bool headerOk = false;
    std::thread thr;
    std::mutex m;
    std::condition_variable cv;
    std::deque<AVPacket*> q;
    bool stop = false, dropUntilKey = false;
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> bytes{0};
};

class Output {
public:
    bool isOpen() const { return open_; }
    bool failed() const {
        if (!open_) return false;
        if (encFailed_) return true;
        for (auto& s : sinks_) if (!s->failed) return false;
        return true;   // todos los destinos han fallado
    }
    const std::vector<std::pair<int, int>>& sizes() const { return sizes_; }
    int stereo() const { return stereo_; }
    std::string startErrors;   // destinos que no conectaron al empezar
    std::string failedNames() const {
        std::string r;
        for (auto& s : sinks_) if (s->failed) r += (r.empty() ? "" : ", ") + s->name;
        return r;
    }
    int failedCount() const {
        int n = 0;
        for (auto& s : sinks_) n += s->failed ? 1 : 0;
        return n;
    }
    std::string summary() const {
        std::string r;
        for (auto& s : sinks_) if (!s->failed) r += (r.empty() ? "" : " + ") + s->name;
        return r;
    }

    // sizes: tamaño real de cada salida de la capa (las "auto" ya resueltas)
    bool open(const Config& cfg, const std::vector<std::pair<int, int>>& sizes, int stereo = 0) {
        cfg_ = cfg;
        sizes_ = sizes;
        stereo_ = stereo;
        encFailed_ = false;
        startErrors.clear();
        if (cfg.dests.empty()) { LOG("no hay destino: configura service y las claves en %s", gs::configPath().c_str()); return false; }

        // Un codificador NVENC por cada combinación distinta de (resolución, tasa de bits, códec)
        std::vector<int> sinkEnc;
        for (auto& d : cfg.dests) {
            int idx = -1;
            for (size_t i = 0; i < encs_.size(); i++)
                if (encStream_[i] == d.stream && rates_[i] == d.bitrate && codecs_[i] == d.codec) idx = (int)i;
            if (idx < 0) {
                if (d.stream >= (int)sizes.size()) { LOG("%s: salida %d inexistente", d.name.c_str(), d.stream); close(); return false; }
                auto e = std::make_unique<NvEnc>();
                Config c = cfg;
                c.bitrate = d.bitrate;
                c.codec = d.codec;
                if (!e->open(c, sizes[d.stream].first, sizes[d.stream].second)) { close(); return false; }
                encs_.push_back(std::move(e));
                encStream_.push_back(d.stream);
                rates_.push_back(d.bitrate);
                codecs_.push_back(d.codec);
                idx = (int)encs_.size() - 1;
            }
            sinkEnc.push_back(idx);
        }

        openAudio();

        if (cfg.camera) {
            CameraOptions co;
            co.device = cfg.camDevice;
            co.capW = cfg.camCapW;
            co.capH = cfg.camCapH;
            co.fps = cfg.camFps;
            co.sizePct = cfg.camSizePct;
            co.position = cfg.camPosition;
            co.margin = cfg.camMargin;
            co.border = cfg.camBorder;
            co.mirror = cfg.camMirror;
            co.stereo = stereo;
            overlays_.push_back(std::make_unique<CameraOverlay>(co, sizes));
        }

        for (size_t i = 0; i < cfg.dests.size(); i++) {
            auto s = std::make_unique<Sink>();
            if (openSink(*s, cfg.dests[i], sinkEnc[i])) sinks_.push_back(std::move(s));
            else {
                startErrors += (startErrors.empty() ? "" : ", ") + cfg.dests[i].name;
                closeSink(*s);
            }
        }
        if (sinks_.empty()) { LOG("no se pudo conectar con ningún destino"); close(); return false; }

        t0_ = nowUs();
        lastVPts_ = -1;
        frames_ = 0;
        quit_ = false;
        open_ = true;
        for (auto& s : sinks_) s->thr = std::thread(&Output::writerLoop, this, s.get());
        if (haveAudio_) startAudio();
        for (auto& sk : sinks_) {
            int e = sk->enc;
            LOG("emitiendo %dx%d @ %u fps, %s %d kbps -> %s", sizes_[encStream_[e]].first, sizes_[encStream_[e]].second,
                cfg.fps, codecs_[e] == "hevc" ? "HEVC" : "H.264", rates_[e], sk->name.c_str());
        }
        LOG("%zu imagen(es) desde el juego, %zu codificación(es) NVENC, %zu destino(s)", sizes_.size(), encs_.size(), sinks_.size());
        return true;
    }

    // Copia el frame del juego (en VRAM, importado de Vulkan) a cada
    // codificador y lo codifica. released() se llama tras las copias.
    // src/pitch: un buffer por salida de la capa (misma VRAM del juego)
    void video(const CUdeviceptr* src, const uint32_t* pitch, uint64_t tsNs, const std::function<void()>& released) {
        int64_t pts = (int64_t)(tsNs / 1000) - (int64_t)t0_;
        if (pts < 0) { released(); return; }
        if (pts <= lastVPts_) pts = lastVPts_ + 1;
        lastVPts_ = pts;
        for (size_t i = 0; i < encs_.size(); i++) encs_[i]->upload(src[encStream_[i]], pitch[encStream_[i]], overlays_);
#ifndef GS_FAKE_ENC
        g_cuda.push();
        g_cuda.cuStreamSynchronize(nullptr);
        g_cuda.pop();
#endif
        released();   // el buffer del juego vuelve a la capa
        for (size_t i = 0; i < encs_.size(); i++) {
            bool used = false;   // no gastar NVENC en una tasa cuyo destino se ha caído
            for (auto& sk : sinks_) used |= sk->enc == (int)i && !sk->failed;
            if (!used) continue;
            bool key = false;
            if (!encs_[i]->encodeNow(pts, bits_, key)) { encFailed_ = true; return; }
            if (bits_.empty()) continue;
            AVPacket* p = av_packet_alloc();
            av_new_packet(p, (int)bits_.size());
            memcpy(p->data, bits_.data(), bits_.size());
            p->pts = p->dts = pts;
            if (key) p->flags |= AV_PKT_FLAG_KEY;
            distribute(p, {1, 1000000}, true, (int)i);
        }
        frames_++;
    }

    void close() {
        if (open_) {
            quit_ = true;
            stopAudio();
            if (aenc_) { avcodec_send_frame(aenc_, nullptr); drain(aenc_); }
            for (auto& s : sinks_) {
                {
                    std::lock_guard<std::mutex> l(s->m);
                    s->stop = true;
                }
                s->cv.notify_all();
            }
            for (auto& s : sinks_) {
                if (s->thr.joinable()) s->thr.join();
                if (s->headerOk && !s->failed) av_write_trailer(s->oc);
            }
            LOG("emisión terminada (%llu frames, %.0f s)", (unsigned long long)frames_, (nowUs() - t0_) / 1e6);
        }
        for (auto& s : sinks_) closeSink(*s);
        sinks_.clear();
        for (auto& e : encs_) e->close();
        encs_.clear();
        overlays_.clear();   // para la cámara
        rates_.clear();
        encStream_.clear();
        codecs_.clear();
        sizes_.clear();
        avcodec_free_context(&aenc_);
        stopAudio();
        haveAudio_ = false;
        open_ = false;
    }

    std::string stats() {
        double secs = (nowUs() - t0_) / 1e6;
        char b[160];
        snprintf(b, sizeof b, "%.0f s, %llu frames (%.1f fps)", secs, (unsigned long long)frames_,
                 secs > 0 ? frames_ / secs : 0.0);
        std::string r = b;
        for (auto& s : sinks_) {
            size_t qs;
            {
                std::lock_guard<std::mutex> l(s->m);
                qs = s->q.size();
            }
            snprintf(b, sizeof b, " | %s: %s, %.0f kbps, cola %zu", s->name.c_str(), s->failed ? "CAÍDO" : "ok",
                     secs > 0 ? s->bytes * 8 / 1000.0 / secs : 0.0, qs);
            r += b;
        }
        return r;
    }

private:
    bool openSink(Sink& s, const Dest& d, int enc) {
        s.name = d.name;
        s.url = d.url;
        s.enc = enc;
        const bool rtmp = d.url.rfind("rtmp", 0) == 0;
        int r = avformat_alloc_output_context2(&s.oc, nullptr, rtmp ? "flv" : nullptr, d.url.c_str());
        if (r < 0 || !s.oc) r = avformat_alloc_output_context2(&s.oc, nullptr, "matroska", d.url.c_str());
        if (r < 0 || !s.oc) { LOG("%s: no se pudo crear la salida: %s", s.name.c_str(), averr(r).c_str()); return false; }
        s.oc->max_interleave_delta = 1000000;   // nunca retener vídeo más de 1 s esperando al audio

        NvEnc& e = *encs_[enc];
        s.vst = avformat_new_stream(s.oc, nullptr);
        auto* vp = s.vst->codecpar;
        vp->codec_type = AVMEDIA_TYPE_VIDEO;
        vp->codec_id = codecs_[enc] == "hevc" ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
        vp->width = sizes_[encStream_[enc]].first;
        vp->height = sizes_[encStream_[enc]].second;
        vp->bit_rate = (int64_t)rates_[enc] * 1000;
        vp->extradata = (uint8_t*)av_mallocz(e.header.size() + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(vp->extradata, e.header.data(), e.header.size());
        vp->extradata_size = (int)e.header.size();
        s.vst->avg_frame_rate = {(int)cfg_.fps, 1};
        s.vst->time_base = {1, 1000};
        if (aenc_) {
            s.ast = avformat_new_stream(s.oc, nullptr);
            avcodec_parameters_from_context(s.ast->codecpar, aenc_);
            s.ast->time_base = {1, 1000};
        }

        if (!(s.oc->oformat->flags & AVFMT_NOFILE)) {
            AVDictionary* o = nullptr;
            if (rtmp) av_dict_set(&o, "rw_timeout", "10000000", 0);
            r = avio_open2(&s.oc->pb, d.url.c_str(), AVIO_FLAG_WRITE, nullptr, &o);
            av_dict_free(&o);
            if (r < 0) { LOG("%s: no se pudo conectar: %s", s.name.c_str(), averr(r).c_str()); return false; }
        }
        AVDictionary* mo = nullptr;
        if (rtmp) av_dict_set(&mo, "flvflags", "no_duration_filesize", 0);
        r = avformat_write_header(s.oc, &mo);
        av_dict_free(&mo);
        if (r < 0) { LOG("%s: error al iniciar: %s", s.name.c_str(), averr(r).c_str()); return false; }
        s.headerOk = true;
        LOG("%s: conectado (%d kbps)", s.name.c_str(), rates_[enc]);
        return true;
    }

    void closeSink(Sink& s) {
        for (auto* p : s.q) av_packet_free(&p);
        s.q.clear();
        if (s.oc && !(s.oc->oformat->flags & AVFMT_NOFILE)) avio_closep(&s.oc->pb);
        if (s.oc) avformat_free_context(s.oc);
        s.oc = nullptr;
    }

    // Reparte un paquete a los destinos que lo necesitan (toma posesión de p)
    void distribute(AVPacket* p, AVRational tb, bool isVideo, int enc) {
        for (auto& sp : sinks_) {
            Sink& s = *sp;
            if (s.failed) continue;
            if (isVideo && s.enc != enc) continue;
            AVStream* st = isVideo ? s.vst : s.ast;
            if (!st) continue;
            AVPacket* c = av_packet_clone(p);
            if (!c) continue;
            c->stream_index = st->index;
            av_packet_rescale_ts(c, tb, st->time_base);
            std::lock_guard<std::mutex> l(s.m);
            if (s.q.size() > 400) s.dropUntilKey = true;   // ese destino va atascado
            if (s.dropUntilKey && isVideo) {
                if (c->flags & AV_PKT_FLAG_KEY) s.dropUntilKey = false;
                else { av_packet_free(&c); continue; }
            }
            s.q.push_back(c);
            s.cv.notify_one();
        }
        av_packet_free(&p);
    }

    void writerLoop(Sink* s) {
        while (true) {
            AVPacket* p;
            {
                std::unique_lock<std::mutex> l(s->m);
                s->cv.wait(l, [&] { return s->stop || !s->q.empty(); });
                if (s->q.empty()) return;
                p = s->q.front();
                s->q.pop_front();
            }
            if (!s->failed) {
                int sz = p->size;
                int r = av_interleaved_write_frame(s->oc, p);
                if (r < 0) {
                    LOG("%s: error enviando (%s): se deja de emitir a este destino", s->name.c_str(), averr(r).c_str());
                    s->failed = true;
                } else {
                    s->bytes += (uint64_t)sz;
                }
            }
            av_packet_free(&p);
        }
    }

    // Fuentes de audio: la primera marca el ritmo; las demás (micrófono) se
    // capturan en su propio hilo, se corrigen al mismo reloj y se mezclan.
    struct AudioSource {
        std::string name;
        PulseCapture cap;
        AudioSync sync;
        float gain = 1.f;
        std::thread thr;
        std::mutex m;
        std::deque<float> fifo;   // estéreo intercalado, ya sincronizado
        bool primed = false;
    };

    bool openAudio() {
        auto add = [&](const char* name, const std::string& dev, bool mic, int volume) {
            auto src = std::make_unique<AudioSource>();
            std::string err;
            if (!src->cap.open(dev, err, mic)) { LOG("%s: %s (se emite sin él)", name, err.c_str()); return; }
            src->name = name;
            src->gain = std::max(0, volume) / 100.f;
            LOG("%s: %s (volumen %d%%)", name, src->cap.device.c_str(), volume);
            asrc_.push_back(std::move(src));
        };
        if (cfg_.audio != "none") add("audio del juego", cfg_.audio, false, cfg_.desktopVolume);
        if (cfg_.microphone) add("micrófono", cfg_.micDevice, true, cfg_.micVolume);
        if (asrc_.empty()) { LOG("sin fuentes de audio: se emite sin audio"); return false; }

        const AVCodec* ac = avcodec_find_encoder(AV_CODEC_ID_AAC);
        aenc_ = avcodec_alloc_context3(ac);
        aenc_->sample_fmt = AV_SAMPLE_FMT_FLTP;
        aenc_->sample_rate = 48000;
        av_channel_layout_default(&aenc_->ch_layout, 2);
        aenc_->bit_rate = (int64_t)cfg_.audioBitrate * 1000;
        aenc_->time_base = {1, 48000};
        aenc_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;   // FLV y MKV lo necesitan
        int r = avcodec_open2(aenc_, ac, nullptr);
        if (r < 0) {
            LOG("AAC: %s", averr(r).c_str());
            avcodec_free_context(&aenc_);
            asrc_.clear();
            return false;
        }
        haveAudio_ = true;
        return true;
    }

    void startAudio() {
        for (size_t i = 1; i < asrc_.size(); i++) asrc_[i]->thr = std::thread(&Output::secondaryLoop, this, asrc_[i].get());
        audioThr_ = std::thread(&Output::audioLoop, this);
    }

    void stopAudio() {
        if (audioThr_.joinable()) audioThr_.join();
        for (auto& s : asrc_) if (s->thr.joinable()) s->thr.join();
        asrc_.clear();
    }

    // Hilo de una fuente secundaria (micrófono): lee bloques de 10 ms y los deja
    // en su cola, ya ajustados al reloj del sistema.
    void secondaryLoop(AudioSource* s) {
        const int n = 480;
        std::vector<float> in(n * 2), out;
        while (!quit_) {
            if (!s->cap.read(in.data(), n)) { LOG("%s: error de lectura, se deja de mezclar", s->name.c_str()); break; }
            out.clear();
            // instante real de captura = llegada - latencia del sistema de sonido
            s->sync.push(in.data(), n, (int64_t)nowUs() - (int64_t)t0_ - (int64_t)s->cap.latencyUs(), out);
            std::lock_guard<std::mutex> l(s->m);
            s->fifo.insert(s->fifo.end(), out.begin(), out.end());
            const size_t maxF = 48000 / 4 * 2, keep = 48000 / 20 * 2;   // >250 ms acumulado: recortar a 50 ms
            if (s->fifo.size() > maxF) s->fifo.erase(s->fifo.begin(), s->fifo.end() - (ptrdiff_t)keep);
        }
    }

    // Suma 'frames' muestras de las fuentes secundarias sobre l/r
    void mixSecondary(float* l, float* r, int frames) {
        for (size_t k = 1; k < asrc_.size(); k++) {
            AudioSource& s = *asrc_[k];
            std::lock_guard<std::mutex> lk(s.m);
            if (!s.primed) {
                if (s.fifo.size() < 48000 / 50 * 2) continue;   // esperar a tener 20 ms de colchón
                s.primed = true;
            }
            int n = std::min(frames, (int)(s.fifo.size() / 2));
            for (int i = 0; i < n; i++) {
                l[i] += s.fifo[2 * i] * s.gain;
                r[i] += s.fifo[2 * i + 1] * s.gain;
            }
            s.fifo.erase(s.fifo.begin(), s.fifo.begin() + 2 * n);
            if (n < frames) s.primed = false;   // se ha vaciado: rellenar colchón de nuevo
        }
    }

    void audioLoop() {
        AudioSource& prim = *asrc_[0];
        AVFrame* fr = av_frame_alloc();
        const int fs = aenc_->frame_size ? aenc_->frame_size : 1024;
        std::vector<float> in((size_t)fs * 2), pending;
        const int64_t delay = (int64_t)cfg_.audioDelayMs * 48;   // ajuste manual de lip-sync
        int64_t done = 0;
        uint64_t lastLog = nowUs();
        uint64_t latSum = 0, latN = 0;
        while (!quit_) {
            if (!prim.cap.read(in.data(), fs)) { LOG("%s: error de lectura, se para el audio", prim.name.c_str()); break; }
            const uint64_t lat = prim.cap.latencyUs();
            latSum += lat;
            latN++;
            // instante real de captura = llegada - latencia del sistema de sonido
            prim.sync.push(in.data(), fs, (int64_t)nowUs() - (int64_t)t0_ - (int64_t)lat, pending);
            if (latN == 50) LOG("%s: latencia de captura %.0f ms (compensada)", prim.name.c_str(), latSum / 50 / 1000.0);
            size_t off = 0;
            while (pending.size() - off >= (size_t)fs * 2) {
                fr->nb_samples = fs;
                fr->format = AV_SAMPLE_FMT_FLTP;
                fr->sample_rate = 48000;
                av_channel_layout_default(&fr->ch_layout, 2);
                if (av_frame_get_buffer(fr, 0) < 0) break;
                float* l = (float*)fr->data[0];
                float* rch = (float*)fr->data[1];
                for (int i = 0; i < fs; i++) {
                    l[i] = pending[off + 2 * i] * prim.gain;
                    rch[i] = pending[off + 2 * i + 1] * prim.gain;
                }
                mixSecondary(l, rch, fs);
                for (int i = 0; i < fs; i++) {
                    l[i] = std::clamp(l[i], -1.f, 1.f);
                    rch[i] = std::clamp(rch[i], -1.f, 1.f);
                }
                off += (size_t)fs * 2;
                fr->pts = prim.sync.startSamples() + done + delay;
                done += fs;
                if (fr->pts >= 0) {
                    std::lock_guard<std::mutex> lk(encMtx_);
                    if (avcodec_send_frame(aenc_, fr) >= 0) drain(aenc_, false);
                }
                av_frame_unref(fr);
            }
            pending.erase(pending.begin(), pending.begin() + (ptrdiff_t)off);
            if (nowUs() - lastLog > 300000000ull) {   // cada 5 min, para diagnóstico
                lastLog = nowUs();
                LOG("audio: desfase %+.1f ms, corrección de reloj %+.0f ppm", prim.sync.driftMs(), prim.sync.correctionPpm());
                for (size_t k = 1; k < asrc_.size(); k++)
                    LOG("%s: corrección de reloj %+.0f ppm", asrc_[k]->name.c_str(), asrc_[k]->sync.correctionPpm());
            }
        }
        av_frame_free(&fr);
    }

    void drain(AVCodecContext* enc, bool lock = true) {
        std::unique_lock<std::mutex> l(encMtx_, std::defer_lock);
        if (lock) l.lock();
        AVPacket* p = av_packet_alloc();
        while (avcodec_receive_packet(enc, p) >= 0) {
            distribute(p, enc->time_base, false, 0);
            p = av_packet_alloc();
        }
        av_packet_free(&p);
    }

    Config cfg_;
    bool open_ = false;
    std::atomic<bool> encFailed_{false}, quit_{false};
    int w_ = 0, h_ = 0;
    uint64_t t0_ = 0;
    int64_t lastVPts_ = -1;
    uint64_t frames_ = 0;

    std::vector<std::unique_ptr<NvEnc>> encs_;
    std::vector<int> rates_;
    std::vector<int> encStream_;          // salida de la capa que usa cada codificador
    std::vector<std::string> codecs_;
    std::vector<std::pair<int, int>> sizes_;
    std::vector<uint8_t> bits_;
    std::vector<std::unique_ptr<Overlay>> overlays_;
    int stereo_ = 0;
    std::vector<std::unique_ptr<Sink>> sinks_;

    AVCodecContext* aenc_ = nullptr;
    std::vector<std::unique_ptr<AudioSource>> asrc_;
    bool haveAudio_ = false;
    std::thread audioThr_;
    std::mutex encMtx_;
};

// ---------------------------------------------------------------------------
// Clientes (juegos y órdenes de control)
// ---------------------------------------------------------------------------
struct Client {
    int fd = -1;
    uint32_t role = GS_ROLE_GAME;
    int pid = 0;
    std::string exe = "?";
    uint64_t connectedAt = 0;
    // texturas importadas
    uint32_t gen = 0, nslots = 0, nstreams = 0;
    uint32_t pitch[GS_MAX_STREAMS]{};
    // [slot * GS_MAX_STREAMS + salida]
    CUexternalMemory ext[GS_MAX_SLOTS * GS_MAX_STREAMS]{};
    CUdeviceptr ptr[GS_MAX_SLOTS * GS_MAX_STREAMS]{};
};

class Daemon {
public:
    int run() {
        cfg_ = loadConfig();
        listen_ = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        sockaddr_un a{};
        a.sun_family = AF_UNIX;
        size_t n = strlen(GS_SOCKET_NAME);
        memcpy(a.sun_path + 1, GS_SOCKET_NAME, n);
        if (bind(listen_, (sockaddr*)&a, (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + n)) != 0) {
            LOG("ya hay otro gamestreamd en marcha (%s)", strerror(errno));
            return 1;
        }
        listen(listen_, 16);
        LOG("esperando juegos. Config: %s", gs::configPath().c_str());
        if (cfg_.dests.empty())
            LOG("AVISO: no hay destino configurado (service y claves)");

        uint64_t lastStats = 0;
        while (!g_quit) {
            std::vector<pollfd> pfds;
            pfds.push_back({listen_, POLLIN, 0});
            for (auto& c : clients_) pfds.push_back({c->fd, POLLIN, 0});
            int r = poll(pfds.data(), pfds.size(), 200);
            if (r < 0 && errno != EINTR) break;
            if (r > 0) {
                if (pfds[0].revents & POLLIN) acceptClient();
                for (size_t i = 1; i < pfds.size(); i++) {
                    if (!pfds[i].revents) continue;
                    Client* c = findFd(pfds[i].fd);
                    if (!c) continue;
                    if (!readClient(c)) dropClient(c);
                }
            }
            if (out_.isOpen() && out_.failed()) {
                notify(cfg_, "Error en la emisión: se detiene");
                stopStreaming();
            }
            if (out_.isOpen() && out_.failedCount() != lastFailed_) {
                lastFailed_ = out_.failedCount();
                if (lastFailed_) notify(cfg_, "Se ha caído: " + out_.failedNames() + ". Sigue en directo: " + out_.summary());
            }
            if (out_.isOpen() && nowUs() - lastStats > 10000000) {
                lastStats = nowUs();
                LOG("%s", out_.stats().c_str());
            }
        }
        stopStreaming();
        while (!clients_.empty()) dropClient(clients_.back().get());
        close(listen_);
        return 0;
    }

    static std::atomic<bool> g_quit;

private:
    Client* findFd(int fd) {
        for (auto& c : clients_) if (c->fd == fd) return c.get();
        return nullptr;
    }

    void acceptClient() {
        int fd;
        while ((fd = accept4(listen_, nullptr, nullptr, SOCK_CLOEXEC)) >= 0) {
            auto c = std::make_unique<Client>();
            c->fd = fd;
            c->connectedAt = nowUs();
            clients_.push_back(std::move(c));
        }
    }

    void dropClient(Client* c) {
        if (c == active_) {
            if (out_.isOpen()) notify(cfg_, "El juego se ha cerrado: emisión detenida");
            stopStreaming();
        }
        if (c->role == GS_ROLE_GAME && c->pid) LOG("juego desconectado: %s (%d)", c->exe.c_str(), c->pid);
        releaseImports(c);
        close(c->fd);
        for (auto it = clients_.begin(); it != clients_.end(); ++it)
            if (it->get() == c) { clients_.erase(it); break; }
    }

    void sendTo(Client* c, const void* d, size_t n) {
        if (c) send(c->fd, d, n, MSG_NOSIGNAL | MSG_DONTWAIT);
    }

    bool readClient(Client* c) {
        char buf[GS_MAX_MSG];
        char cbuf[CMSG_SPACE(sizeof(int) * GS_MAX_SLOTS * GS_MAX_STREAMS)];
        iovec iov{buf, sizeof buf};
        msghdr mh{};
        mh.msg_iov = &iov;
        mh.msg_iovlen = 1;
        mh.msg_control = cbuf;
        mh.msg_controllen = sizeof cbuf;
        ssize_t n = recvmsg(c->fd, &mh, MSG_CMSG_CLOEXEC);
        if (n <= 0) return false;
        std::vector<int> fds;
        for (cmsghdr* cm = CMSG_FIRSTHDR(&mh); cm; cm = CMSG_NXTHDR(&mh, cm)) {
            if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) {
                size_t k = (cm->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                int* p = (int*)CMSG_DATA(cm);
                fds.assign(p, p + k);
            }
        }
        uint32_t type = 0;
        if (n >= 4) memcpy(&type, buf, 4);
        switch (type) {
        case GS_HELLO: {
            GsHello h{};
            memcpy(&h, buf, std::min((size_t)n, sizeof h));
            h.exe[sizeof h.exe - 1] = 0;
            c->role = h.role;
            c->pid = h.pid;
            c->exe = h.exe;
            if (c->role == GS_ROLE_GAME) LOG("juego conectado: %s (pid %d)", c->exe.c_str(), c->pid);
            break;
        }
        case GS_TEXTURES:
            if ((size_t)n >= offsetof(GsTextures, stereo)) {   // "stereo" sólo lo manda la capa 3D Vision
                GsTextures t{};
                memcpy(&t, buf, std::min((size_t)n, sizeof t));
                if (t.stereo > 2) t.stereo = 0;
                onTextures(c, t, fds);
                fds.clear();
            }
            break;
        case GS_FRAME:
            if ((size_t)n >= sizeof(GsFrame)) {
                GsFrame f;
                memcpy(&f, buf, sizeof f);
                onFrame(c, f);
            }
            break;
        case GS_TOGGLE:
            toggle(c);
            break;
        case GS_CTL:
            if ((size_t)n >= sizeof(GsCtl)) {
                GsCtl ctl;
                memcpy(&ctl, buf, sizeof ctl);
                onCtl(c, ctl.cmd);
            }
            break;
        }
        for (int fd : fds) close(fd);
        return true;
    }

    void releaseImports(Client* c) {
        if (!c->nslots || !g_cuda.ctx) { c->nslots = 0; return; }
        g_cuda.push();
        for (uint32_t i = 0; i < GS_MAX_SLOTS * GS_MAX_STREAMS; i++) {
            if (c->ptr[i]) g_cuda.cuMemFree(c->ptr[i]);
            if (c->ext[i]) g_cuda.cuDestroyExternalMemory(c->ext[i]);
            c->ptr[i] = 0;
            c->ext[i] = nullptr;
        }
        g_cuda.pop();
        c->nslots = 0;
    }

    void onTextures(Client* c, const GsTextures& t, std::vector<int>& fds) {
        releaseImports(c);
        if (c != active_) return;
        if (t.version != GS_PROTO_VERSION) {
            failStart("La capa Vulkan es de otra versión: reinstala gamestream y reinicia el juego");
            return;
        }
        if (t.nslots > GS_MAX_SLOTS || t.nstreams < 1 || t.nstreams > GS_MAX_STREAMS || fds.size() != t.nslots * t.nstreams) return;
        if (!g_cuda.ensure(t.has_uuid, t.device_uuid)) { failStart("CUDA/NVENC: " + lastError()); return; }
        g_cuda.push();
        bool ok = true;
        for (uint32_t i = 0; i < t.nslots && ok; i++) {
            for (uint32_t k = 0; k < t.nstreams && ok; k++) {
                int& fd = fds[i * t.nstreams + k];
                const uint32_t idx = i * GS_MAX_STREAMS + k;
                CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{};
                d.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
                d.handle.fd = fd;
                d.size = t.alloc_size[k];
                d.flags = t.dedicated ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;
                if (g_cuda.cuImportExternalMemory(&c->ext[idx], &d) == CUDA_SUCCESS) {
                    fd = -1;   // ahora es de CUDA
                    CUDA_EXTERNAL_MEMORY_BUFFER_DESC b{};
                    b.size = (unsigned long long)t.pitch[k] * t.height[k];
                    if (g_cuda.cuExternalMemoryGetMappedBuffer(&c->ptr[idx], c->ext[idx], &b) != CUDA_SUCCESS) ok = false;
                } else ok = false;
            }
        }
        g_cuda.pop();
        for (int& fd : fds) if (fd >= 0) close(fd);
        fds.clear();
        c->nslots = t.nslots;
        if (!ok) { releaseImports(c); failStart("No se pudo importar la memoria del juego en CUDA"); return; }
        c->gen = t.generation;
        c->nstreams = t.nstreams;
        std::vector<std::pair<int, int>> sizes;
        for (uint32_t k = 0; k < t.nstreams; k++) {
            c->pitch[k] = t.pitch[k];
            sizes.push_back({(int)t.width[k], (int)t.height[k]});
        }

        if (out_.isOpen() && (out_.sizes() != sizes || out_.stereo() != (int)t.stereo)) out_.close();
        if (!out_.isOpen()) {
            if (!out_.open(cfg_, sizes, (int)t.stereo)) { failStart("No se pudo iniciar: " + lastError()); return; }
            notify(cfg_, "EN DIRECTO (" + out_.summary() + "): " + c->exe +
                             (out_.startErrors.empty() ? "" : ". No conectó: " + out_.startErrors));
        }
    }

    void onFrame(Client* c, const GsFrame& f) {
        GsRelease rel{GS_RELEASE, f.generation, f.slot};
        if (c == active_ && out_.isOpen() && f.generation == c->gen && f.slot < c->nslots && c->ptr[f.slot * GS_MAX_STREAMS]) {
            out_.video(&c->ptr[f.slot * GS_MAX_STREAMS], c->pitch, f.ts_ns, [&] { sendTo(c, &rel, sizeof rel); });
        } else {
            sendTo(c, &rel, sizeof rel);
        }
    }

    Client* pickGame(Client* hint) {
        if (hint && hint->role == GS_ROLE_GAME) return hint;
        Client* best = nullptr;
        for (auto& c : clients_)
            if (c->role == GS_ROLE_GAME && (!best || c->connectedAt > best->connectedAt)) best = c.get();
        return best;
    }

    bool startStreaming(Client* hint) {
        if (active_) return true;
        cfg_ = loadConfig();   // se relee en cada inicio: puedes cambiarla sin reiniciar el daemon
        if (cfg_.dests.empty()) {
            notify(cfg_, "Configura service/key o url en " + gs::configPath());
            return false;
        }
        Client* g = pickGame(hint);
        if (!g) { LOG("no hay ningún juego Vulkan conectado"); return false; }
        active_ = g;
        GsStart s{};
        s.type = GS_START;
        s.version = GS_PROTO_VERSION;
        s.fps = cfg_.fps;
        s.nstreams = (uint32_t)cfg_.streams.size();
        for (size_t k = 0; k < cfg_.streams.size(); k++) {
            s.width[k] = cfg_.streams[k].first;
            s.height[k] = cfg_.streams[k].second;
        }
        sendTo(g, &s, sizeof s);
        LOG("iniciando emisión de %s...", g->exe.c_str());
        return true;
    }

    void stopStreaming() {
        if (!active_) return;
        GsSimple s{GS_STOP};
        sendTo(active_, &s, sizeof s);
        bool was = out_.isOpen();
        lastFailed_ = 0;
        out_.close();
        releaseImports(active_);
        active_ = nullptr;
        if (was) notify(cfg_, "Emisión detenida");
    }

    void failStart(const std::string& why) {
        notify(cfg_, why);
        stopStreaming();
    }

    void toggle(Client* hint) {
        uint64_t now = nowUs();
        if (now - lastToggle_ < 700000) return;   // varios procesos pueden ver la misma pulsación
        lastToggle_ = now;
        if (active_) stopStreaming();
        else startStreaming(hint);
    }

    void onCtl(Client* c, uint32_t cmd) {
        GsCtlReply rep{};
        rep.type = GS_CTL_REPLY;
        std::string msg;
        switch (cmd) {
        case GS_CTL_TOGGLE:
            lastToggle_ = 0;
            toggle(nullptr);
            msg = active_ ? "emisión iniciándose" : "emisión detenida";
            break;
        case GS_CTL_START:
            msg = startStreaming(nullptr) ? "emisión iniciándose" : "no se pudo iniciar (¿hay un juego Vulkan conectado y destino configurado?)";
            break;
        case GS_CTL_STOP:
            stopStreaming();
            msg = "emisión detenida";
            break;
        case GS_CTL_QUIT:
            g_quit = true;
            msg = "saliendo";
            break;
        default: {
            int games = 0;
            for (auto& cl : clients_) games += cl->role == GS_ROLE_GAME;
            msg = std::to_string(games) + " juego(s) conectado(s). ";
            msg += out_.isOpen() ? ("EN DIRECTO (" + active_->exe + "): " + out_.stats()) : "No se está emitiendo.";
        }
        }
        snprintf(rep.text, sizeof rep.text, "%s", msg.c_str());
        sendTo(c, &rep, sizeof rep);
    }

    Config cfg_;
    int listen_ = -1;
    std::vector<std::unique_ptr<Client>> clients_;
    Client* active_ = nullptr;
    Output out_;
    uint64_t lastToggle_ = 0;
    int lastFailed_ = 0;
};
std::atomic<bool> Daemon::g_quit{false};

// ---------------------------------------------------------------------------
// Cliente de control: gamestreamd toggle|start|stop|status|quit
// ---------------------------------------------------------------------------
static int ctl(uint32_t cmd) {
    int s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    size_t n = strlen(GS_SOCKET_NAME);
    memcpy(a.sun_path + 1, GS_SOCKET_NAME, n);
    if (connect(s, (sockaddr*)&a, (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + n)) != 0) {
        fprintf(stderr, "gamestreamd no está en marcha\n");
        return 1;
    }
    GsHello h{};
    h.type = GS_HELLO; h.version = GS_PROTO_VERSION; h.pid = getpid(); h.role = GS_ROLE_CTL;
    snprintf(h.exe, sizeof h.exe, "ctl");
    send(s, &h, sizeof h, 0);
    GsCtl c{GS_CTL, cmd};
    send(s, &c, sizeof c, 0);
    pollfd p{s, POLLIN, 0};
    if (poll(&p, 1, 3000) > 0) {
        GsCtlReply r{};
        if (recv(s, &r, sizeof r, 0) > 0) {
            r.text[sizeof r.text - 1] = 0;
            printf("%s\n", r.text);
        }
    }
    close(s);
    return 0;
}

// gamestreamd config: muestra a dónde se emitiría (claves ocultas)
static int showConfig() {
    Config c = loadConfig();
    printf("Config: %s\n", gs::configPath().c_str());
    static const char* known[] = {"service", "key", "server", "url", "record", "record_bitrate", "twitch_key",
                                  "twitch_server", "twitch_bitrate", "youtube_key", "youtube_server", "youtube_bitrate",
                                  "resolution", "fps", "codec", "bitrate", "keyint", "preset", "audio",
                                  "audio_bitrate", "audio_delay", "audio_volume", "microphone", "microphone_device",
                                  "microphone_volume", "camera", "camera_device", "camera_capture", "camera_fps",
                                  "camera_size", "camera_position", "camera_margin", "camera_border", "camera_mirror",
                                  "hotkey", "notify", "twitch_resolution", "youtube_resolution",
                                  "record_resolution", "custom_resolution", "twitch_codec", "youtube_codec",
                                  "record_codec", "custom_codec"};
    for (auto& kv : gs::loadConfig(gs::configPath())) {
        bool ok = false;
        for (auto k : known) ok |= kv.first == k;
        if (!ok) printf("  AVISO: no reconozco la opción '%s' (¿errata?)\n", kv.first.c_str());
    }
    if (c.dests.empty()) printf("  (sin destinos: revisa service y las claves)\n");
    for (auto& d : c.dests) {
        std::string u = d.url;
        size_t k = u.rfind('/');
        if (u.rfind("rtmp", 0) == 0 && k != std::string::npos && u.size() - k > 5)
            u = u.substr(0, k + 1) + u.substr(k + 1, 4) + std::string(u.size() - k - 5, '*');
        std::string res = d.w ? std::to_string(d.w) + "x" + std::to_string(d.h) : "auto";
        printf("  %-10s %-9s %-4s %5d kbps  %s\n", d.name.c_str(), res.c_str(), d.codec.c_str(), d.bitrate, u.c_str());
    }
    printf("  %zu imagen(es) distinta(s) desde el juego @ %u fps, preset %s | audio: %s %d kbps\n",
           c.streams.size(), c.fps, c.preset.c_str(), c.audio.c_str(), c.audioBitrate);
    printf("  micrófono: %s", c.microphone ? "sí" : "no");
    if (c.microphone) printf(" (%s, volumen %d%%)", c.micDevice.c_str(), c.micVolume);
    printf(" | cámara: %s", c.camera ? "sí" : "no");
    if (c.camera) printf(" (%s, %d%% del ancho, %s)", c.camDevice.c_str(), c.camSizePct, c.camPosition.c_str());
    printf("\n");
    return c.dests.empty() ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string a = argv[1];
        if (a == "toggle") return ctl(GS_CTL_TOGGLE);
        if (a == "start") return ctl(GS_CTL_START);
        if (a == "stop") return ctl(GS_CTL_STOP);
        if (a == "status") return ctl(GS_CTL_STATUS);
        if (a == "quit") return ctl(GS_CTL_QUIT);
        if (a == "config") return showConfig();
#ifdef GS_FAKE_ENC
        if (a == "testout") {   // SÓLO PRUEBAS: emite N segundos de vídeo sintético a los destinos
            avformat_network_init();
            av_log_set_callback(ffLogCb);
            Output o;
            Config tc = loadConfig();
            std::vector<std::pair<int, int>> sizes;
            for (auto& st : tc.streams) sizes.push_back({st.first ? (int)st.first : 640, st.second ? (int)st.second : 360});
            if (!o.open(tc, sizes)) return 1;
            CUdeviceptr ptrs[GS_MAX_STREAMS] = {};
            uint32_t pitches[GS_MAX_STREAMS] = {};
            int secs = argc > 2 ? atoi(argv[2]) : 5;
            for (int i = 0; i < secs * 30; i++) {
                timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                o.video(ptrs, pitches, (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec, [] {});
                usleep(33333);
                if (i % 60 == 59) LOG("%s", o.stats().c_str());
            }
            o.close();
            return 0;
        }
#endif
        if (a != "run") {
            printf("uso: %s [run|toggle|start|stop|status|config|quit]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{};
    sa.sa_handler = [](int) { Daemon::g_quit = true; };
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    av_log_set_level(getenv("GAMESTREAM_DEBUG") ? AV_LOG_INFO : AV_LOG_WARNING);
    av_log_set_callback(ffLogCb);
    openLogFile();
    LOG("---- gamestreamd iniciado ----");
    avformat_network_init();
    Daemon d;
    return d.run();
}
