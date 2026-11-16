// gamestream - capa Vulkan implícita de captura.
//
// Qué hace dentro del proceso del juego (y nada más):
//   * Añade TRANSFER_SRC a las imágenes del swapchain.
//   * En cada vkQueuePresentKHR (sólo mientras se emite y respetando los fps
//     objetivo) graba un comando de copia GPU->GPU de la imagen presentada a un
//     buffer en VRAM exportable (VK_KHR_external_memory_fd), escalando con un
//     blit si hace falta. No hay ninguna lectura a CPU ni espera en el hilo del
//     juego: la copia se encola en la misma cola, encadenada con semáforos.
//   * Un hilo aparte espera la fence de la copia y avisa al daemon por un
//     socket UNIX. El daemon importa esa misma memoria en CUDA y la manda a
//     NVENC: cero copias por la CPU, cero cambios de contexto en el juego.
//   * Otro hilo vigila el atajo de teclado (X11/XWayland) y mantiene la
//     conexión con el daemon.
// Si no hay daemon, o no se está emitiendo, el coste es prácticamente cero.

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "../common/config.h"
#include "../common/protocol.h"

#define EXPORT extern "C" __attribute__((visibility("default")))

static const bool g_debug = getenv("GAMESTREAM_DEBUG") != nullptr;
#define LOG(...) do { fprintf(stderr, "[gamestream] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#define DBG(...) do { if (g_debug) LOG(__VA_ARGS__); } while (0)

template <typename T> static inline void* dkey(T h) { return *(void**)h; }

static uint64_t nowNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// Instancia
// ---------------------------------------------------------------------------
struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceExternalBufferProperties GetPhysicalDeviceExternalBufferProperties = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
};

// ---------------------------------------------------------------------------
// Dispositivo
// ---------------------------------------------------------------------------
#define DEVICE_FUNCS(X)                                                        \
    X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2)                      \
    X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR)      \
    X(QueuePresentKHR) X(CreateCommandPool) X(DestroyCommandPool)              \
    X(AllocateCommandBuffers) X(FreeCommandBuffers) X(BeginCommandBuffer)      \
    X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdPipelineBarrier)            \
    X(CmdCopyImageToBuffer) X(CmdBlitImage) X(CmdClearColorImage)              \
    X(QueueSubmit) X(CreateFence) X(DestroyFence) X(WaitForFences)             \
    X(ResetFences) X(CreateSemaphore) X(DestroySemaphore) X(CreateBuffer)      \
    X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory)          \
    X(FreeMemory) X(BindBufferMemory) X(CreateImage) X(DestroyImage)           \
    X(GetImageMemoryRequirements) X(BindImageMemory) X(GetMemoryFdKHR)         \
    X(DeviceWaitIdle)

struct SwapchainInfo {
    std::vector<VkImage> images;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
};

class Capturer;

struct DeviceData {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    InstanceData* inst = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;

    bool captureOk = false;
    bool exportDedicatedOnly = false;
    bool hasUuid = false;
    uint8_t uuid[16]{};
    VkPhysicalDeviceMemoryProperties memProps{};
    std::vector<VkQueueFamilyProperties> qfams;

    std::mutex mtx;   // protege los dos mapas siguientes
    std::unordered_map<VkQueue, uint32_t> queueFamily;
    std::unordered_map<VkSwapchainKHR, SwapchainInfo> swapchains;

    Capturer* cap = nullptr;

#define DECL(n) PFN_vk##n n = nullptr;
    DEVICE_FUNCS(DECL)
#undef DECL
};

static std::mutex g_mapMtx;
static std::unordered_map<void*, InstanceData*> g_instances;
static std::unordered_map<void*, DeviceData*> g_devices;

static InstanceData* getInstance(void* k) {
    std::lock_guard<std::mutex> l(g_mapMtx);
    auto it = g_instances.find(k);
    return it == g_instances.end() ? nullptr : it->second;
}
static DeviceData* getDevice(void* k) {
    std::lock_guard<std::mutex> l(g_mapMtx);
    auto it = g_devices.find(k);
    return it == g_devices.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------
// Atajo de teclado (X11 / XWayland vía dlopen, sin dependencia de enlazado)
// ---------------------------------------------------------------------------
class Hotkey {
public:
    bool init() {
        std::string spec;
        if (const char* e = getenv("GAMESTREAM_HOTKEY")) spec = e;
        else {
            auto cfg = gs::loadConfig(gs::configPath());
            spec = cfg.count("hotkey") ? cfg["hotkey"] : "Ctrl+Shift+F9";
        }
        if (spec.empty() || spec == "none") return false;

        lib_ = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) { DBG("libX11 no disponible: sin hotkey (usa 'gamestreamd toggle')"); return false; }
        auto XOpenDisplay = (void* (*)(const char*))dlsym(lib_, "XOpenDisplay");
        auto XStringToKeysym = (unsigned long (*)(const char*))dlsym(lib_, "XStringToKeysym");
        auto XKeysymToKeycode = (unsigned char (*)(void*, unsigned long))dlsym(lib_, "XKeysymToKeycode");
        XQueryKeymap_ = (int (*)(void*, char*))dlsym(lib_, "XQueryKeymap");
        XCloseDisplay_ = (int (*)(void*))dlsym(lib_, "XCloseDisplay");
        if (!XOpenDisplay || !XStringToKeysym || !XKeysymToKeycode || !XQueryKeymap_) return false;
        dpy_ = XOpenDisplay(nullptr);
        if (!dpy_) { DBG("no hay DISPLAY X11: sin hotkey (usa 'gamestreamd toggle')"); return false; }

        auto codes = [&](std::initializer_list<const char*> names) {
            std::vector<unsigned char> v;
            for (auto n : names) {
                unsigned long ks = XStringToKeysym(n);
                if (ks) if (unsigned char kc = XKeysymToKeycode(dpy_, ks)) v.push_back(kc);
            }
            return v;
        };
        size_t pos = 0;
        std::vector<std::string> parts;
        while (true) {
            size_t p = spec.find('+', pos);
            parts.push_back(gs::trim(spec.substr(pos, p == std::string::npos ? std::string::npos : p - pos)));
            if (p == std::string::npos) break;
            pos = p + 1;
        }
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            std::string m = parts[i];
            for (auto& c : m) c = (char)tolower((unsigned char)c);
            if (m == "ctrl" || m == "control") mods_.push_back(codes({"Control_L", "Control_R"}));
            else if (m == "shift") mods_.push_back(codes({"Shift_L", "Shift_R"}));
            else if (m == "alt") mods_.push_back(codes({"Alt_L", "Alt_R", "Meta_L", "ISO_Level3_Shift"}));
            else if (m == "super" || m == "win") mods_.push_back(codes({"Super_L", "Super_R"}));
            else LOG("modificador desconocido en hotkey: %s", parts[i].c_str());
        }
        std::string k = parts.back();
        unsigned long ks = XStringToKeysym(k.c_str());
        if (!ks && k.size() == 1) { std::string l(1, (char)tolower((unsigned char)k[0])); ks = XStringToKeysym(l.c_str()); }
        key_ = ks ? XKeysymToKeycode(dpy_, ks) : 0;
        if (!key_) { LOG("tecla de hotkey no reconocida: '%s'", k.c_str()); return false; }
        DBG("hotkey: %s", spec.c_str());
        return true;
    }
    // true en el flanco de pulsación
    bool poll() {
        if (!dpy_ || !key_) return false;
        char keys[32];
        XQueryKeymap_(dpy_, keys);
        auto down = [&](unsigned char kc) { return (keys[kc >> 3] >> (kc & 7)) & 1; };
        bool pressed = down(key_);
        for (auto& g : mods_) {
            bool any = false;
            for (auto kc : g) any |= down(kc);
            pressed &= any;
        }
        bool edge = pressed && !prev_;
        prev_ = pressed;
        return edge;
    }
    ~Hotkey() {
        if (dpy_ && XCloseDisplay_) XCloseDisplay_(dpy_);
        // no hacemos dlclose: libX11 suele estar cargada por el propio juego
    }

private:
    void* lib_ = nullptr;
    void* dpy_ = nullptr;
    int (*XQueryKeymap_)(void*, char*) = nullptr;
    int (*XCloseDisplay_)(void*) = nullptr;
    unsigned char key_ = 0;
    std::vector<std::vector<unsigned char>> mods_;
    bool prev_ = false;
};

// ---------------------------------------------------------------------------
// Capturador (uno por VkDevice)
// ---------------------------------------------------------------------------
enum { SLOT_FREE = 0, SLOT_GPU = 1, SLOT_DAEMON = 2 };

struct Slot {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::atomic<int> state{SLOT_FREE};
    uint64_t ts = 0;
};

static bool isSrgb(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return true;
    default: return false;
    }
}

class Capturer {
public:
    explicit Capturer(DeviceData* d) : d_(d) {}

    void ensureStarted() {
        std::call_once(started_, [this] {
            const uint32_t kSems = 16;
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            for (uint32_t i = 0; i < kSems; i++) {
                VkSemaphore s;
                if (d_->CreateSemaphore(d_->device, &sci, nullptr, &s) == VK_SUCCESS) sems_.push_back(s);
            }
            ioThr_ = std::thread(&Capturer::ioLoop, this);
            fenceThr_ = std::thread(&Capturer::fenceLoop, this);
            running_ = true;
        });
    }

    void shutdown() {
        if (d_->DeviceWaitIdle) d_->DeviceWaitIdle(d_->device);
        quit_ = true;
        fcv_.notify_all();
        if (ioThr_.joinable()) ioThr_.join();
        if (fenceThr_.joinable()) fenceThr_.join();
        {
            std::lock_guard<std::mutex> l(resMtx_);
            destroyAll();
        }
        for (auto s : sems_) d_->DestroySemaphore(d_->device, s, nullptr);
        sems_.clear();
        std::lock_guard<std::mutex> l(sockMtx_);
        if (sock_ >= 0) close(sock_);
        sock_ = -1;
    }

    // Devuelve true si ha encolado una copia; entonces 'out' es el
    // VkPresentInfoKHR a usar (espera sólo a nuestro semáforo).
    bool onPresent(VkQueue q, const VkPresentInfoKHR* pi, VkPresentInfoKHR* out, VkSemaphore* sigOut) {
        if (!running_ || sems_.empty()) return false;
        if (!wantCapture_) {
            if (valid_) tryDestroy();
            return false;
        }
        if (valid_ && resSession_ != session_.load()) {
            if (!tryDestroy()) return false;
        }

        // ¿Qué swapchain capturamos? El de mayor área de este present.
        VkImage srcImg = VK_NULL_HANDLE;
        VkFormat srcFmt = VK_FORMAT_UNDEFINED;
        VkExtent2D srcExt{};
        uint32_t fam = 0;
        {
            std::lock_guard<std::mutex> l(d_->mtx);
            uint64_t best = 0;
            for (uint32_t i = 0; i < pi->swapchainCount; i++) {
                auto it = d_->swapchains.find(pi->pSwapchains[i]);
                if (it == d_->swapchains.end()) continue;
                const auto& sc = it->second;
                uint32_t idx = pi->pImageIndices[i];
                if (idx >= sc.images.size()) continue;
                uint64_t area = (uint64_t)sc.extent.width * sc.extent.height;
                if (area > best) {
                    best = area; srcImg = sc.images[idx]; srcFmt = sc.format; srcExt = sc.extent;
                }
            }
            auto qi = d_->queueFamily.find(q);
            if (qi == d_->queueFamily.end()) return false;
            fam = qi->second;
        }
        if (!srcImg || srcExt.width < 16 || srcExt.height < 16) return false;
        if (!(d_->qfams[fam].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) return false;

        if (valid_ && fam != family_) {
            if (!tryDestroy()) return false;
        }
        if (!valid_ && !create(fam, srcExt)) {
            wantCapture_ = false;
            return false;
        }

        // Ritmo de captura: sólo los fps que pide el stream.
        const uint64_t now = nowNs();
        const uint64_t interval = 1000000000ull / (fps_ ? fps_ : 60);
        if (nextDue_ && now + interval / 4 < nextDue_) return false;

        int s = -1;
        for (uint32_t i = 0; i < nslots_; i++)
            if (slots_[i].state.load() == SLOT_FREE) { s = i; break; }
        if (s < 0) { dropped_++; return false; }   // el daemon va lento: se descarta, el juego no espera

        const bool fast = (srcFmt == VK_FORMAT_B8G8R8A8_UNORM || srcFmt == VK_FORMAT_B8G8R8A8_SRGB) &&
                          srcExt.width == outW_ && srcExt.height == outH_;
        if (!fast) {
            VkFormat want = isSrgb(srcFmt) ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
            if (interm_ && intermFmt_ != want) destroyInterm();
            if (!interm_ && !createInterm(want)) return false;
            if (!canBlitFrom(srcFmt)) return false;
        }
        nextDue_ = std::max(nextDue_ + interval, now + interval / 2);

        Slot& sl = slots_[s];
        record(sl, srcImg, srcFmt, srcExt, fast);

        std::vector<VkPipelineStageFlags> stages(pi->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkSemaphore sig = sems_[semIdx_++ % sems_.size()];
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = pi->waitSemaphoreCount;
        si.pWaitSemaphores = pi->pWaitSemaphores;
        si.pWaitDstStageMask = stages.data();
        si.commandBufferCount = 1;
        si.pCommandBuffers = &sl.cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sig;
        d_->ResetFences(d_->device, 1, &sl.fence);
        VkResult r = d_->QueueSubmit(q, 1, &si, sl.fence);
        if (r != VK_SUCCESS) {
            LOG("vkQueueSubmit de la copia falló (%d)", r);
            return false;
        }
        sl.ts = now;
        sl.state = SLOT_GPU;
        {
            std::lock_guard<std::mutex> l(fq_);
            fqueue_.push_back((uint32_t)s);
        }
        fcv_.notify_one();

        *out = *pi;
        *sigOut = sig;
        out->waitSemaphoreCount = 1;
        out->pWaitSemaphores = sigOut;
        return true;
    }

private:
    // ---- grabación de la copia -------------------------------------------
    void record(Slot& sl, VkImage src, VkFormat srcFmt, VkExtent2D ext, bool fast) {
        d_->ResetCommandBuffer(sl.cb, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        d_->BeginCommandBuffer(sl.cb, &bi);

        const VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da,
                           VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = sa; b.dstAccessMask = da;
            b.oldLayout = from; b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img; b.subresourceRange = rng;
            d_->CmdPipelineBarrier(sl.cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
        };

        barrier(src, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkImage copyFrom = src;
        if (!fast) {
            barrier(interm_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            // Escalado conservando la proporción (bandas negras si hace falta)
            double sc = std::min((double)outW_ / ext.width, (double)outH_ / ext.height);
            int32_t dw = (int32_t)(ext.width * sc + 0.5), dh = (int32_t)(ext.height * sc + 0.5);
            int32_t x0 = ((int32_t)outW_ - dw) / 2, y0 = ((int32_t)outH_ - dh) / 2;
            if (dw != (int32_t)outW_ || dh != (int32_t)outH_) {
                VkClearColorValue black{};
                black.float32[3] = 1.f;
                d_->CmdClearColorImage(sl.cb, interm_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rng);
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                d_->CmdPipelineBarrier(sl.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {(int32_t)ext.width, (int32_t)ext.height, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[0] = {x0, y0, 0};
            blit.dstOffsets[1] = {x0 + dw, y0 + dh, 1};
            d_->CmdBlitImage(sl.cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             interm_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                             canFilterLinear(srcFmt) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
            barrier(interm_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            copyFrom = interm_;
        }

        VkBufferImageCopy region{};
        region.bufferRowLength = pitch_ / 4;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {outW_, outH_, 1};
        d_->CmdCopyImageToBuffer(sl.cb, copyFrom, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sl.buf, 1, &region);

        barrier(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_READ_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        d_->EndCommandBuffer(sl.cb);
    }

    bool canBlitFrom(VkFormat f) {
        auto it = blitCache_.find(f);
        if (it != blitCache_.end()) return it->second;
        VkFormatProperties fp{};
        d_->inst->GetPhysicalDeviceFormatProperties(d_->phys, f, &fp);
        bool ok = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if (!ok) LOG("el formato del swapchain (%d) no admite blit: no se puede escalar", f);
        blitCache_[f] = ok;
        return ok;
    }
    bool canFilterLinear(VkFormat f) {
        VkFormatProperties fp{};
        d_->inst->GetPhysicalDeviceFormatProperties(d_->phys, f, &fp);
        return fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }

    int memType(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < d_->memProps.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (d_->memProps.memoryTypes[i].propertyFlags & want) == want) return (int)i;
        for (uint32_t i = 0; i < d_->memProps.memoryTypeCount; i++)
            if (bits & (1u << i)) return (int)i;
        return -1;
    }

    // ---- recursos ----------------------------------------------------------
    bool create(uint32_t fam, VkExtent2D ext) {
        std::lock_guard<std::mutex> l(resMtx_);
        uint32_t w = reqW_, h = reqH_;
        if (!w || !h) { w = ext.width; h = ext.height; }
        outW_ = w & ~1u; outH_ = h & ~1u;           // NVENC quiere dimensiones pares
        pitch_ = (outW_ * 4 + 255) & ~255u;
        fps_ = reqFps_;
        nslots_ = 4;
        family_ = fam;

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = fam;
        if (d_->CreateCommandPool(d_->device, &pci, nullptr, &pool_) != VK_SUCCESS) { destroyAll(); return false; }

        std::vector<int> fds;
        bool ok = true;
        for (uint32_t i = 0; i < nslots_ && ok; i++) {
            Slot& s = slots_[i];
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = pool_;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            ok = d_->AllocateCommandBuffers(d_->device, &ai, &s.cb) == VK_SUCCESS;
            if (!ok) break;
            if (d_->setLoaderData) d_->setLoaderData(d_->device, s.cb);
            else *(void**)s.cb = *(void**)d_->device;

            VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ebi};
            bci.size = (VkDeviceSize)pitch_ * outH_;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ok = d_->CreateBuffer(d_->device, &bci, nullptr, &s.buf) == VK_SUCCESS;
            if (!ok) break;
            VkMemoryRequirements mr;
            d_->GetBufferMemoryRequirements(d_->device, s.buf, &mr);
            int mt = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            ded.buffer = s.buf;
            VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                           d_->exportDedicatedOnly ? (void*)&ded : nullptr};
            exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exp};
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = (uint32_t)mt;
            ok = mt >= 0 && d_->AllocateMemory(d_->device, &mai, nullptr, &s.mem) == VK_SUCCESS;
            if (!ok) break;
            allocSize_ = mr.size;
            ok = d_->BindBufferMemory(d_->device, s.buf, s.mem, 0) == VK_SUCCESS;
            if (!ok) break;
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            ok = d_->CreateFence(d_->device, &fci, nullptr, &s.fence) == VK_SUCCESS;
            if (!ok) break;
            VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
            gi.memory = s.mem;
            gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1;
            ok = d_->GetMemoryFdKHR(d_->device, &gi, &fd) == VK_SUCCESS && fd >= 0;
            if (ok) fds.push_back(fd);
            s.state = SLOT_FREE;
        }
        if (ok) {
            GsTextures t{};
            t.type = GS_TEXTURES;
            t.generation = ++gen_;
            t.width = outW_; t.height = outH_; t.pitch = pitch_;
            t.nslots = nslots_;
            t.alloc_size = allocSize_;
            t.dedicated = d_->exportDedicatedOnly;
            t.has_uuid = d_->hasUuid;
            memcpy(t.device_uuid, d_->uuid, 16);
            t.pixfmt = 0;
            ok = sendMsg(&t, sizeof t, fds.data(), (int)fds.size());
            if (!ok) LOG("no se pudieron enviar los buffers al daemon");
        }
        for (int fd : fds) close(fd);
        if (!ok) { destroyAll(); return false; }
        valid_ = true;
        resSession_ = session_.load();
        nextDue_ = 0;
        LOG("capturando %ux%u @ %u fps (%u buffers en VRAM)", outW_, outH_, fps_, nslots_);
        return true;
    }

    bool createInterm(VkFormat fmt) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {outW_, outH_, 1};
        ici.mipLevels = ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (d_->CreateImage(d_->device, &ici, nullptr, &interm_) != VK_SUCCESS) { interm_ = VK_NULL_HANDLE; return false; }
        VkMemoryRequirements mr;
        d_->GetImageMemoryRequirements(d_->device, interm_, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        int mt = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        mai.memoryTypeIndex = (uint32_t)mt;
        if (mt < 0 || d_->AllocateMemory(d_->device, &mai, nullptr, &intermMem_) != VK_SUCCESS ||
            d_->BindImageMemory(d_->device, interm_, intermMem_, 0) != VK_SUCCESS) {
            destroyInterm();
            return false;
        }
        intermFmt_ = fmt;
        return true;
    }
    void destroyInterm() {
        if (interm_) d_->DestroyImage(d_->device, interm_, nullptr);
        if (intermMem_) d_->FreeMemory(d_->device, intermMem_, nullptr);
        interm_ = VK_NULL_HANDLE; intermMem_ = VK_NULL_HANDLE; intermFmt_ = VK_FORMAT_UNDEFINED;
    }

    // Sólo si ninguna copia está en vuelo. Llamar con resMtx_ cogido.
    void destroyAll() {
        for (uint32_t i = 0; i < GS_MAX_SLOTS; i++) {
            Slot& s = slots_[i];
            if (s.buf) d_->DestroyBuffer(d_->device, s.buf, nullptr);
            if (s.mem) d_->FreeMemory(d_->device, s.mem, nullptr);   // el daemon conserva su propia referencia
            if (s.fence) d_->DestroyFence(d_->device, s.fence, nullptr);
            if (s.cb) d_->FreeCommandBuffers(d_->device, pool_, 1, &s.cb);
            s.buf = VK_NULL_HANDLE; s.mem = VK_NULL_HANDLE; s.fence = VK_NULL_HANDLE; s.cb = VK_NULL_HANDLE;
            s.state = SLOT_FREE;
        }
        if (pool_) d_->DestroyCommandPool(d_->device, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
        destroyInterm();
        valid_ = false;
    }
    bool tryDestroy() {
        for (uint32_t i = 0; i < nslots_; i++)
            if (slots_[i].state.load() == SLOT_GPU) return false;
        std::lock_guard<std::mutex> l(resMtx_);
        destroyAll();
        if (dropped_) DBG("frames descartados por falta de buffers libres: %llu", (unsigned long long)dropped_);
        dropped_ = 0;
        return true;
    }

    // ---- comunicación -------------------------------------------------------
    bool sendMsg(const void* data, size_t len, const int* fds = nullptr, int nfds = 0) {
        std::lock_guard<std::mutex> l(sockMtx_);
        if (sock_ < 0) return false;
        iovec iov{const_cast<void*>(data), len};
        msghdr mh{};
        mh.msg_iov = &iov;
        mh.msg_iovlen = 1;
        char cbuf[CMSG_SPACE(sizeof(int) * GS_MAX_SLOTS)];
        if (nfds > 0) {
            memset(cbuf, 0, sizeof cbuf);
            mh.msg_control = cbuf;
            mh.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
            cmsghdr* c = CMSG_FIRSTHDR(&mh);
            c->cmsg_level = SOL_SOCKET;
            c->cmsg_type = SCM_RIGHTS;
            c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
            memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
        }
        return sendmsg(sock_, &mh, MSG_NOSIGNAL) == (ssize_t)len;
    }

    bool tryConnect() {
        int s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (s < 0) return false;
        sockaddr_un a{};
        a.sun_family = AF_UNIX;
        size_t n = strlen(GS_SOCKET_NAME);
        memcpy(a.sun_path + 1, GS_SOCKET_NAME, n);
        if (connect(s, (sockaddr*)&a, (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + n)) != 0) {
            close(s);
            return false;
        }
        {
            std::lock_guard<std::mutex> l(sockMtx_);
            sock_ = s;
        }
        GsHello h{};
        h.type = GS_HELLO;
        h.version = GS_PROTO_VERSION;
        h.pid = getpid();
        h.role = GS_ROLE_GAME;
        {   // /proc/self/comm: con Wine/Proton es el .exe del juego (truncado a 15 letras)
            char comm[64] = {0};
            FILE* f = fopen("/proc/self/comm", "r");
            if (f) { if (!fgets(comm, sizeof comm, f)) comm[0] = 0; fclose(f); }
            size_t n = strlen(comm);
            while (n && (comm[n - 1] == '\n' || comm[n - 1] == ' ')) comm[--n] = 0;
            snprintf(h.exe, sizeof h.exe, "%s", n ? comm : program_invocation_short_name);
        }
        sendMsg(&h, sizeof h);
        LOG("conectado con gamestreamd");
        return true;
    }

    void disconnect() {
        {
            std::lock_guard<std::mutex> l(sockMtx_);
            if (sock_ >= 0) close(sock_);
            sock_ = -1;
        }
        wantCapture_ = false;
        std::lock_guard<std::mutex> l(resMtx_);
        for (uint32_t i = 0; i < GS_MAX_SLOTS; i++) {
            int e = SLOT_DAEMON;
            slots_[i].state.compare_exchange_strong(e, SLOT_FREE);
        }
        LOG("desconectado de gamestreamd");
    }

    void handle(const char* buf, ssize_t n) {
        if (n < 4) return;
        uint32_t type;
        memcpy(&type, buf, 4);
        switch (type) {
        case GS_START:
            if (n >= (ssize_t)sizeof(GsStart)) {
                GsStart s;
                memcpy(&s, buf, sizeof s);
                reqW_ = s.width; reqH_ = s.height; reqFps_ = s.fps ? s.fps : 60;
                session_++;
                wantCapture_ = true;
                LOG("inicio de emisión pedido por el daemon");
            }
            break;
        case GS_STOP:
            wantCapture_ = false;
            LOG("fin de emisión");
            break;
        case GS_RELEASE:
            if (n >= (ssize_t)sizeof(GsRelease)) {
                GsRelease r;
                memcpy(&r, buf, sizeof r);
                std::lock_guard<std::mutex> l(resMtx_);
                if (r.generation == gen_.load() && r.slot < nslots_) {
                    int e = SLOT_DAEMON;
                    slots_[r.slot].state.compare_exchange_strong(e, SLOT_FREE);
                }
            }
            break;
        }
    }

    void ioLoop() {
        Hotkey hk;
        bool hkOk = hk.init();
        uint64_t lastTry = 0;
        while (!quit_) {
            int s;
            {
                std::lock_guard<std::mutex> l(sockMtx_);
                s = sock_;
            }
            if (s < 0) {
                uint64_t now = nowNs();
                if (now - lastTry > 2000000000ull) { lastTry = now; tryConnect(); }
                usleep(40000);
                continue;
            }
            pollfd p{s, POLLIN, 0};
            int r = ::poll(&p, 1, 40);
            if (r > 0) {
                if (p.revents & POLLIN) {
                    char buf[GS_MAX_MSG];
                    ssize_t n = recv(s, buf, sizeof buf, 0);
                    if (n <= 0) { disconnect(); continue; }
                    handle(buf, n);
                } else if (p.revents & (POLLHUP | POLLERR)) {
                    disconnect();
                    continue;
                }
            }
            if (hkOk && hk.poll()) {
                GsSimple t{GS_TOGGLE};
                sendMsg(&t, sizeof t);
            }
        }
    }

    void fenceLoop() {
        while (true) {
            uint32_t s;
            {
                std::unique_lock<std::mutex> l(fq_);
                fcv_.wait(l, [&] { return quit_ || !fqueue_.empty(); });
                if (fqueue_.empty()) return;   // quit
                s = fqueue_.front();
                fqueue_.pop_front();
            }
            Slot& sl = slots_[s];
            VkResult r;
            do {
                r = d_->WaitForFences(d_->device, 1, &sl.fence, VK_TRUE, 500000000ull);
            } while (r == VK_TIMEOUT && !quit_);
            bool sent = false;
            if (r == VK_SUCCESS && wantCapture_ && !quit_) {
                GsFrame f{GS_FRAME, gen_.load(), s, 0, sl.ts};
                sl.state = SLOT_DAEMON;
                sent = sendMsg(&f, sizeof f);
            }
            if (!sent) sl.state = SLOT_FREE;
        }
    }

    DeviceData* d_;
    std::once_flag started_;
    bool running_ = false;
    std::thread ioThr_, fenceThr_;
    std::atomic<bool> quit_{false};

    std::mutex sockMtx_;
    int sock_ = -1;

    std::atomic<bool> wantCapture_{false};
    std::atomic<uint32_t> reqW_{0}, reqH_{0}, reqFps_{60};
    std::atomic<uint32_t> session_{0};

    std::mutex resMtx_;
    bool valid_ = false;
    uint32_t resSession_ = 0;
    uint32_t family_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    uint32_t outW_ = 0, outH_ = 0, pitch_ = 0, fps_ = 60, nslots_ = 4;
    VkDeviceSize allocSize_ = 0;
    Slot slots_[GS_MAX_SLOTS];
    std::atomic<uint32_t> gen_{0};
    VkImage interm_ = VK_NULL_HANDLE;
    VkDeviceMemory intermMem_ = VK_NULL_HANDLE;
    VkFormat intermFmt_ = VK_FORMAT_UNDEFINED;
    std::unordered_map<int, bool> blitCache_;

    std::vector<VkSemaphore> sems_;
    uint32_t semIdx_ = 0;
    uint64_t nextDue_ = 0;
    uint64_t dropped_ = 0;

    std::mutex fq_;
    std::condition_variable fcv_;
    std::deque<uint32_t> fqueue_;
};

// ---------------------------------------------------------------------------
// Funciones interceptadas
// ---------------------------------------------------------------------------
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gs_GetInstanceProcAddr(VkInstance, const char*);
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gs_GetDeviceProcAddr(VkDevice, const char*);

static VKAPI_ATTR VkResult VKAPI_CALL gs_CreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* alloc,
                                                         VkInstance* out) {
    auto* link = (VkLayerInstanceCreateInfo*)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo*)link->pNext;
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");

    // Exportar memoria necesita Vulkan 1.1. Si la app pide 1.0 (algunos juegos
    // nativos), subimos la instancia a 1.1: es compatible hacia atrás.
    uint32_t api = (ci->pApplicationInfo && ci->pApplicationInfo->apiVersion) ? ci->pApplicationInfo->apiVersion
                                                                             : VK_API_VERSION_1_0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    VkInstanceCreateInfo ici = *ci;
    VkLayerInstanceLink* nextLink = link->u.pLayerInfo;
    VkResult r;
    if (api < VK_API_VERSION_1_1) {
        if (ci->pApplicationInfo) app = *ci->pApplicationInfo;
        app.apiVersion = VK_API_VERSION_1_1;
        ici.pApplicationInfo = &app;
        r = create(&ici, alloc, out);
        if (r == VK_SUCCESS) api = VK_API_VERSION_1_1;
        else {
            link->u.pLayerInfo = nextLink;   // las capas siguientes avanzaron la cadena: se restaura
            r = create(ci, alloc, out);
        }
    } else {
        r = create(ci, alloc, out);
    }
    if (r != VK_SUCCESS) return r;

    auto* id = new InstanceData;
    id->instance = *out;
    id->gipa = gipa;
    id->apiVersion = api;
#define IGET(n) id->n = (PFN_vk##n)gipa(*out, "vk" #n)
    IGET(DestroyInstance);
    IGET(EnumerateDeviceExtensionProperties);
    IGET(GetPhysicalDeviceProperties);
    IGET(GetPhysicalDeviceMemoryProperties);
    IGET(GetPhysicalDeviceFormatProperties);
    IGET(GetPhysicalDeviceQueueFamilyProperties);
    IGET(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    if (id->apiVersion >= VK_API_VERSION_1_1) {
        IGET(GetPhysicalDeviceProperties2);
        IGET(GetPhysicalDeviceExternalBufferProperties);
    }
#undef IGET
    std::lock_guard<std::mutex> l(g_mapMtx);
    g_instances[dkey(*out)] = id;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL gs_DestroyInstance(VkInstance inst, const VkAllocationCallbacks* alloc) {
    InstanceData* id;
    {
        std::lock_guard<std::mutex> l(g_mapMtx);
        auto it = g_instances.find(dkey(inst));
        if (it == g_instances.end()) return;
        id = it->second;
        g_instances.erase(it);
    }
    id->DestroyInstance(inst, alloc);
    delete id;
}

static VKAPI_ATTR VkResult VKAPI_CALL gs_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* ci,
                                                       const VkAllocationCallbacks* alloc, VkDevice* out) {
    InstanceData* id = getInstance(dkey(phys));
    auto* link = (VkLayerDeviceCreateInfo*)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo*)link->pNext;
    auto* cbinfo = (VkLayerDeviceCreateInfo*)ci->pNext;
    while (cbinfo && !(cbinfo->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && cbinfo->function == VK_LOADER_DATA_CALLBACK))
        cbinfo = (VkLayerDeviceCreateInfo*)cbinfo->pNext;
    if (!link || !id) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateDevice)gipa(id->instance, "vkCreateDevice");

    // ¿Podemos exportar memoria? Necesitamos Vulkan 1.1 y VK_KHR_external_memory_fd.
    VkPhysicalDeviceProperties props{};
    id->GetPhysicalDeviceProperties(phys, &props);
    bool ok = id->apiVersion >= VK_API_VERSION_1_1 && props.apiVersion >= VK_API_VERSION_1_1 &&
              id->GetPhysicalDeviceExternalBufferProperties;
    if (ok) {
        uint32_t n = 0;
        id->EnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        id->EnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
        bool has = false;
        for (auto& e : exts) has |= !strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        ok = has;
    }
    std::vector<const char*> exts(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    if (ok) {
        bool already = false;
        for (auto e : exts) already |= !strcmp(e, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        if (!already) exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    }
    VkDeviceCreateInfo dci = *ci;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();
    VkResult r = create(phys, &dci, alloc, out);
    if (r != VK_SUCCESS) return r;

    auto* dd = new DeviceData;
    dd->device = *out;
    dd->phys = phys;
    dd->inst = id;
    dd->gdpa = gdpa;
    dd->setLoaderData = cbinfo ? cbinfo->u.pfnSetDeviceLoaderData : nullptr;
#define DGET(n) dd->n = (PFN_vk##n)gdpa(*out, "vk" #n);
    DEVICE_FUNCS(DGET)
#undef DGET
    id->GetPhysicalDeviceMemoryProperties(phys, &dd->memProps);
    uint32_t nq = 0;
    id->GetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    dd->qfams.resize(nq);
    id->GetPhysicalDeviceQueueFamilyProperties(phys, &nq, dd->qfams.data());

    if (ok && id->GetPhysicalDeviceProperties2) {
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
        id->GetPhysicalDeviceProperties2(phys, &p2);
        memcpy(dd->uuid, idp.deviceUUID, 16);
        dd->hasUuid = true;
    }
    if (ok) {
        VkPhysicalDeviceExternalBufferInfo ebi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        ebi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ebi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkExternalBufferProperties ebp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        id->GetPhysicalDeviceExternalBufferProperties(phys, &ebi, &ebp);
        auto f = ebp.externalMemoryProperties.externalMemoryFeatures;
        ok = (f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) && dd->GetMemoryFdKHR;
        dd->exportDedicatedOnly = f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT;
    }
    dd->captureOk = ok;
    if (ok) dd->cap = new Capturer(dd);
    else DBG("dispositivo '%s' sin soporte de exportación de memoria: captura desactivada", props.deviceName);

    std::lock_guard<std::mutex> l(g_mapMtx);
    g_devices[dkey(*out)] = dd;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL gs_DestroyDevice(VkDevice dev, const VkAllocationCallbacks* alloc) {
    DeviceData* dd;
    {
        std::lock_guard<std::mutex> l(g_mapMtx);
        auto it = g_devices.find(dkey(dev));
        if (it == g_devices.end()) return;
        dd = it->second;
        g_devices.erase(it);
    }
    if (dd->cap) {
        dd->cap->shutdown();
        delete dd->cap;
    }
    dd->DestroyDevice(dev, alloc);
    delete dd;
}

static VKAPI_ATTR void VKAPI_CALL gs_GetDeviceQueue(VkDevice dev, uint32_t fam, uint32_t idx, VkQueue* q) {
    DeviceData* dd = getDevice(dkey(dev));
    dd->GetDeviceQueue(dev, fam, idx, q);
    if (*q) {
        std::lock_guard<std::mutex> l(dd->mtx);
        dd->queueFamily[*q] = fam;
    }
}

static VKAPI_ATTR void VKAPI_CALL gs_GetDeviceQueue2(VkDevice dev, const VkDeviceQueueInfo2* info, VkQueue* q) {
    DeviceData* dd = getDevice(dkey(dev));
    dd->GetDeviceQueue2(dev, info, q);
    if (*q) {
        std::lock_guard<std::mutex> l(dd->mtx);
        dd->queueFamily[*q] = info->queueFamilyIndex;
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL gs_CreateSwapchainKHR(VkDevice dev, const VkSwapchainCreateInfoKHR* ci,
                                                             const VkAllocationCallbacks* alloc, VkSwapchainKHR* out) {
    DeviceData* dd = getDevice(dkey(dev));
    VkSwapchainCreateInfoKHR c = *ci;
    bool canCopy = false;
    if (dd->captureOk && dd->inst->GetPhysicalDeviceSurfaceCapabilitiesKHR) {
        VkSurfaceCapabilitiesKHR caps{};
        if (dd->inst->GetPhysicalDeviceSurfaceCapabilitiesKHR(dd->phys, ci->surface, &caps) == VK_SUCCESS &&
            (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            c.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            canCopy = true;
        }
    }
    VkResult r = dd->CreateSwapchainKHR(dev, &c, alloc, out);
    if (r != VK_SUCCESS && canCopy) {
        canCopy = false;
        r = dd->CreateSwapchainKHR(dev, ci, alloc, out);
    }
    if (r != VK_SUCCESS || !canCopy) return r;

    SwapchainInfo info;
    info.format = c.imageFormat;
    info.extent = c.imageExtent;
    uint32_t n = 0;
    dd->GetSwapchainImagesKHR(dev, *out, &n, nullptr);
    info.images.resize(n);
    dd->GetSwapchainImagesKHR(dev, *out, &n, info.images.data());
    {
        std::lock_guard<std::mutex> l(dd->mtx);
        dd->swapchains[*out] = std::move(info);
    }
    DBG("swapchain %ux%u formato %d", c.imageExtent.width, c.imageExtent.height, c.imageFormat);
    dd->cap->ensureStarted();
    return r;
}

static VKAPI_ATTR void VKAPI_CALL gs_DestroySwapchainKHR(VkDevice dev, VkSwapchainKHR sc, const VkAllocationCallbacks* alloc) {
    DeviceData* dd = getDevice(dkey(dev));
    {
        std::lock_guard<std::mutex> l(dd->mtx);
        dd->swapchains.erase(sc);
    }
    dd->DestroySwapchainKHR(dev, sc, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL gs_QueuePresentKHR(VkQueue q, const VkPresentInfoKHR* pi) {
    DeviceData* dd = getDevice(dkey(q));
    if (dd->cap) {
        VkPresentInfoKHR mod;
        VkSemaphore sem;
        if (dd->cap->onPresent(q, pi, &mod, &sem)) return dd->QueuePresentKHR(q, &mod);
    }
    return dd->QueuePresentKHR(q, pi);
}

#define DEVICE_HOOKS(X) X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) \
    X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(QueuePresentKHR)

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gs_GetDeviceProcAddr(VkDevice dev, const char* name) {
#define HOOK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)&gs_##n;
    HOOK(GetDeviceProcAddr)
    DEVICE_HOOKS(HOOK)
    DeviceData* dd = getDevice(dkey(dev));
    return dd ? dd->gdpa(dev, name) : nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gs_GetInstanceProcAddr(VkInstance inst, const char* name) {
    HOOK(GetInstanceProcAddr)
    HOOK(CreateInstance)
    HOOK(DestroyInstance)
    HOOK(CreateDevice)
    HOOK(GetDeviceProcAddr)
    DEVICE_HOOKS(HOOK)
#undef HOOK
    if (!inst) return nullptr;
    InstanceData* id = getInstance(dkey(inst));
    return id ? id->gipa(inst, name) : nullptr;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* p) {
    if (!p || p->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (p->loaderLayerInterfaceVersion > 2) p->loaderLayerInterfaceVersion = 2;
    p->pfnGetInstanceProcAddr = gs_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = gs_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
