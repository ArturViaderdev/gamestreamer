// Prueba de la capa sin NVIDIA: hace de daemon, importa la memoria exportada por
// la capa en OTRO proceso Vulkan (p. ej. lavapipe), lee los píxeles y guarda un
// PPM. Espera al hotkey (GS_TOGGLE) o empieza solo con --autostart.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "../src/common/protocol.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FALLO %s = %d\n", #x, r_); exit(1); } } while (0)

static VkInstance inst; static VkPhysicalDevice phys; static VkDevice dev; static VkQueue queue; static uint32_t qfam = 0;
static VkPhysicalDeviceMemoryProperties mp;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & f) == f) return i;
    return 0;
}

static void initVk() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &phys);
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    float pr = 1; VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueCount = 1; q.pQueuePriorities = &pr;
    const char* ext[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = ext;
    CK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, 0, 0, &queue);
}

struct Imp { VkBuffer buf; VkDeviceMemory mem; };
static GsStart makeStart() {
    GsStart s{}; s.type = GS_START; s.version = GS_PROTO_VERSION; s.fps = 30; s.nstreams = 2;
    s.width[0] = 0; s.height[0] = 0;        // salida 0: resolución del juego
    s.width[1] = 320; s.height[1] = 180;    // salida 1: reducida (blit)
    return s;
}

int main(int argc, char** argv) {
    bool autostart = argc > 1 && !strcmp(argv[1], "--autostart");
    initVk();
    int ls = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    sockaddr_un a{}; a.sun_family = AF_UNIX; memcpy(a.sun_path + 1, GS_SOCKET_NAME, strlen(GS_SOCKET_NAME));
    if (bind(ls, (sockaddr*)&a, offsetof(sockaddr_un, sun_path) + 1 + strlen(GS_SOCKET_NAME))) { perror("bind"); return 1; }
    listen(ls, 4);
    printf("fake daemon esperando\n"); fflush(stdout);
    int c = accept(ls, nullptr, nullptr);

    // staging para leer
    VkBuffer stg; VkDeviceMemory stgMem; void* map = nullptr; VkDeviceSize stgSize = 0;
    VkCommandPool pool; VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = qfam;
    CK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
    VkCommandBuffer cb; VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.commandBufferCount = 1; CK(vkAllocateCommandBuffers(dev, &cai, &cb));

    std::vector<Imp> imps; GsTextures tex{};
    int frames = 0, saved = 0, textures = 0, toggles = 0;
    uint64_t firstTs = 0, lastTs = 0;
    bool started = false;
    for (int iter = 0; iter < 100000; iter++) {
        pollfd p{c, POLLIN, 0};
        if (poll(&p, 1, 8000) <= 0) { printf("timeout esperando mensajes\n"); break; }
        char buf[GS_MAX_MSG]; char cbuf[CMSG_SPACE(sizeof(int) * GS_MAX_SLOTS * GS_MAX_STREAMS)];
        iovec iov{buf, sizeof buf}; msghdr mh{}; mh.msg_iov = &iov; mh.msg_iovlen = 1; mh.msg_control = cbuf; mh.msg_controllen = sizeof cbuf;
        ssize_t n = recvmsg(c, &mh, 0);
        if (n <= 0) { printf("el juego se desconectó\n"); break; }
        std::vector<int> fds;
        for (cmsghdr* cm = CMSG_FIRSTHDR(&mh); cm; cm = CMSG_NXTHDR(&mh, cm))
            if (cm->cmsg_type == SCM_RIGHTS) { int* q = (int*)CMSG_DATA(cm); fds.assign(q, q + (cm->cmsg_len - CMSG_LEN(0)) / sizeof(int)); }
        uint32_t type; memcpy(&type, buf, 4);
        if (type == GS_HELLO) {
            GsHello h; memcpy(&h, buf, sizeof h); printf("HELLO de %s pid %d\n", h.exe, h.pid);
            if (autostart) { GsStart s = makeStart(); send(c, &s, sizeof s, 0); started = true; }
        } else if (type == GS_TOGGLE) {
            toggles++; printf("TOGGLE recibido (hotkey)\n");
            if (!started) { GsStart s = makeStart(); send(c, &s, sizeof s, 0); started = true; }
        } else if (type == GS_TEXTURES) {
            memcpy(&tex, buf, sizeof tex); textures++;
            printf("TEXTURES v%u stereo %u gen %u slots %u salidas %u fds %zu\n", tex.version, tex.stereo, tex.generation, tex.nslots, tex.nstreams, fds.size());
            for (uint32_t k = 0; k < tex.nstreams; k++)
                printf("  salida %u: %ux%u pitch %u alloc %llu\n", k, tex.width[k], tex.height[k], tex.pitch[k], (unsigned long long)tex.alloc_size[k]);
            imps.clear();
            for (uint32_t i = 0; i < tex.nslots * tex.nstreams; i++) {
                uint32_t k = i % tex.nstreams;
                Imp im;
                VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
                eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb};
                bci.size = (VkDeviceSize)tex.pitch[k] * tex.height[k]; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                CK(vkCreateBuffer(dev, &bci, nullptr, &im.buf));
                VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, im.buf, &mr);
                VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
                imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT; imp.fd = fds[i];
                VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp};
                mai.allocationSize = tex.alloc_size[k]; mai.memoryTypeIndex = memType(mr.memoryTypeBits, 0);
                CK(vkAllocateMemory(dev, &mai, nullptr, &im.mem));
                CK(vkBindBufferMemory(dev, im.buf, im.mem, 0));
                imps.push_back(im);
            }
            stgSize = (VkDeviceSize)tex.pitch[0] * tex.height[0];
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = stgSize; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            CK(vkCreateBuffer(dev, &bci, nullptr, &stg));
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, stg, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size;
            mai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            CK(vkAllocateMemory(dev, &mai, nullptr, &stgMem)); CK(vkBindBufferMemory(dev, stg, stgMem, 0));
            CK(vkMapMemory(dev, stgMem, 0, stgSize, 0, &map));
        } else if (type == GS_FRAME) {
            GsFrame f; memcpy(&f, buf, sizeof f);
            if (!firstTs) firstTs = f.ts_ns; lastTs = f.ts_ns;
            frames++;
            if (frames == 20 || frames == 40) {
                for (uint32_t k = 0; k < tex.nstreams; k++) {
                    VkDeviceSize sz = (VkDeviceSize)tex.pitch[k] * tex.height[k];
                    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    vkResetCommandBuffer(cb, 0);
                    vkBeginCommandBuffer(cb, &bi);
                    VkBufferCopy r{0, 0, sz}; vkCmdCopyBuffer(cb, imps[f.slot * tex.nstreams + k].buf, stg, 1, &r);
                    vkEndCommandBuffer(cb);
                    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                    CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); vkQueueWaitIdle(queue);
                    char name[64]; snprintf(name, sizeof name, "frame_%d_s%u.ppm", frames, k);
                    FILE* o = fopen(name, "wb"); fprintf(o, "P6\n%u %u\n255\n", tex.width[k], tex.height[k]);
                    uint64_t sum = 0;
                    for (uint32_t y = 0; y < tex.height[k]; y++) for (uint32_t x = 0; x < tex.width[k]; x++) {
                        uint8_t* px = (uint8_t*)map + (size_t)y * tex.pitch[k] + x * 4;   // BGRA
                        uint8_t rgb[3] = {px[2], px[1], px[0]}; fwrite(rgb, 1, 3, o); sum += px[0] + px[1] + px[2];
                    }
                    fclose(o); saved++;
                    printf("frame %d salida %u (slot %u) -> %s, brillo medio %.1f\n", frames, k, f.slot, name, sum / (3.0 * tex.width[k] * tex.height[k]));
                }
            }
            GsRelease rel{GS_RELEASE, f.generation, f.slot}; send(c, &rel, sizeof rel, 0);
            if (frames == 90) {
                printf("90 frames en %.2f s => %.1f fps (objetivo 30)\n", (lastTs - firstTs) / 1e9, 89 / ((lastTs - firstTs) / 1e9));
                GsSimple s{GS_STOP}; send(c, &s, sizeof s, 0);
                printf("STOP enviado\n");
                break;
            }
        }
        for (int fd : fds) if (fd >= 0 && type != GS_TEXTURES) close(fd);
    }
    usleep(500000);
    printf("RESUMEN frames=%d textures=%d toggles=%d guardados=%d\n", frames, textures, toggles, saved);
    close(c);
    return frames >= 90 && saved == 4 ? 0 : 1;
}
