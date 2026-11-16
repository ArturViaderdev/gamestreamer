// Cámara web superpuesta en una esquina del stream.
//
//  * Un hilo captura de V4L2 (/dev/videoN) en YUYV o MJPEG, sin libavdevice.
//  * Convierte y escala en CPU a un tamaño pequeño BGRA (p. ej. 420x316: <2 ms).
//  * apply() se ejecuta en el camino de NVENC: copia esa imagen a la esquina de
//    la superficie de vídeo en VRAM (~0,5 MB por frame). El juego no se entera.
//  * Si la cámara no está o se desconecta, se reintenta cada pocos segundos y
//    el stream sigue sin ella.
//
// Se incluye desde gamestreamd.cpp después de definir Overlay, g_cuda y LOG.
#pragma once
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
}

// ---------------------------------------------------------------------------
// Conversión YUV -> BGRA con escalado bilineal, espejo y borde (CPU)
// ---------------------------------------------------------------------------
struct YuvView {
    int w = 0, h = 0;
    bool yuyv = false;   // empaquetado Y0 U Y1 V
    const uint8_t* p[3] = {nullptr, nullptr, nullptr};
    int stride[3] = {0, 0, 0};
    int cwShift = 1, chShift = 0;   // submuestreo de croma (planar)
    bool fullRange = false;          // JPEG = rango completo
};

inline void yuvToBgra(const YuvView& in, uint8_t* out, int ow, int oh, int outPitch, bool mirror, int border) {
    auto Y = [&](int x, int y) -> int {
        return in.yuyv ? in.p[0][y * in.stride[0] + 2 * x] : in.p[0][y * in.stride[0] + x];
    };
    auto U = [&](int x, int y) -> int {
        if (in.yuyv) return in.p[0][y * in.stride[0] + 4 * (x >> 1) + 1];
        return in.p[1][(y >> in.chShift) * in.stride[1] + (x >> in.cwShift)];
    };
    auto V = [&](int x, int y) -> int {
        if (in.yuyv) return in.p[0][y * in.stride[0] + 4 * (x >> 1) + 3];
        return in.p[2][(y >> in.chShift) * in.stride[2] + (x >> in.cwShift)];
    };
    const float sx = (float)in.w / ow, sy = (float)in.h / oh;
    for (int oy = 0; oy < oh; oy++) {
        uint8_t* row = out + (size_t)oy * outPitch;
        float fy = (oy + 0.5f) * sy - 0.5f;
        int y0 = std::clamp((int)fy, 0, in.h - 1), y1 = std::min(y0 + 1, in.h - 1);
        float wy = std::clamp(fy - y0, 0.f, 1.f);
        for (int ox = 0; ox < ow; ox++) {
            uint8_t* px = row + 4 * ox;
            if (border > 0 && (ox < border || oy < border || ox >= ow - border || oy >= oh - border)) {
                px[0] = px[1] = px[2] = px[3] = 255;
                continue;
            }
            int srcX = mirror ? ow - 1 - ox : ox;
            float fx = (srcX + 0.5f) * sx - 0.5f;
            int x0 = std::clamp((int)fx, 0, in.w - 1), x1 = std::min(x0 + 1, in.w - 1);
            float wx = std::clamp(fx - x0, 0.f, 1.f);
            float yy = (Y(x0, y0) * (1 - wx) + Y(x1, y0) * wx) * (1 - wy) + (Y(x0, y1) * (1 - wx) + Y(x1, y1) * wx) * wy;
            float u = U(x0, y0) - 128.f, v = V(x0, y0) - 128.f;
            float r, g, b;
            if (in.fullRange) {
                r = yy + 1.402f * v;
                g = yy - 0.344f * u - 0.714f * v;
                b = yy + 1.772f * u;
            } else {
                float c = 1.164f * (yy - 16.f);
                r = c + 1.596f * v;
                g = c - 0.392f * u - 0.813f * v;
                b = c + 2.017f * u;
            }
            px[0] = (uint8_t)std::clamp(b, 0.f, 255.f);
            px[1] = (uint8_t)std::clamp(g, 0.f, 255.f);
            px[2] = (uint8_t)std::clamp(r, 0.f, 255.f);
            px[3] = 255;
        }
    }
}

// Decodificador MJPEG (las webcams suelen dar MJPEG a 720p/1080p)
class MjpegDecoder {
public:
    ~MjpegDecoder() {
        av_frame_free(&fr_);
        av_packet_free(&pkt_);
        avcodec_free_context(&ctx_);
    }
    bool decode(const uint8_t* data, size_t size, YuvView& out) {
        if (!ctx_) {
            const AVCodec* c = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
            if (!c) return false;
            ctx_ = avcodec_alloc_context3(c);
            ctx_->thread_count = 2;
            if (avcodec_open2(ctx_, c, nullptr) < 0) { avcodec_free_context(&ctx_); return false; }
            fr_ = av_frame_alloc();
            pkt_ = av_packet_alloc();
        }
        buf_.assign(data, data + size);
        buf_.resize(size + AV_INPUT_BUFFER_PADDING_SIZE, 0);
        pkt_->data = buf_.data();
        pkt_->size = (int)size;
        if (avcodec_send_packet(ctx_, pkt_) < 0) return false;
        if (avcodec_receive_frame(ctx_, fr_) < 0) return false;
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((AVPixelFormat)fr_->format);
        if (!d || !(d->flags & AV_PIX_FMT_FLAG_PLANAR) || d->nb_components < 3) return false;
        out.w = fr_->width;
        out.h = fr_->height;
        out.yuyv = false;
        for (int i = 0; i < 3; i++) { out.p[i] = fr_->data[i]; out.stride[i] = fr_->linesize[i]; }
        out.cwShift = d->log2_chroma_w;
        out.chShift = d->log2_chroma_h;
        out.fullRange = fr_->color_range == AVCOL_RANGE_JPEG || fr_->format == AV_PIX_FMT_YUVJ420P ||
                        fr_->format == AV_PIX_FMT_YUVJ422P || fr_->format == AV_PIX_FMT_YUVJ444P;
        return true;
    }

private:
    AVCodecContext* ctx_ = nullptr;
    AVFrame* fr_ = nullptr;
    AVPacket* pkt_ = nullptr;
    std::vector<uint8_t> buf_;
};

#ifndef GS_CAMERA_NO_OVERLAY
// ---------------------------------------------------------------------------
// Superposición de la cámara
// ---------------------------------------------------------------------------
struct CameraOptions {
    std::string device = "/dev/video0";
    int capW = 640, capH = 480, fps = 30;
    int sizePct = 22;
    std::string position = "abajo-derecha";
    int margin = 20, border = 3;
    bool mirror = true;
    int stereo = 0;   // 0 = 2D, 1 = half SBS, 2 = full SBS: la cámara se pone en los dos ojos
};

class CameraOverlay : public Overlay {
public:
    CameraOverlay(const CameraOptions& o, int streamW, int streamH) : o_(o), sw_(streamW), sh_(streamH) {
        thr_ = std::thread(&CameraOverlay::loop, this);
    }
    ~CameraOverlay() override {
        quit_ = true;
        if (thr_.joinable()) thr_.join();
    }

    // Copia la última imagen de la cámara a la esquina del frame (en VRAM).
    // En 3D lado a lado, en la misma esquina de cada ojo (a la profundidad de la pantalla).
    void apply(CUdeviceptr frame, size_t pitch, int w, int h, CUstream stream) override {
        std::lock_guard<std::mutex> l(m_);
        if (o_.stereo) {
            int hw = w / 2, mx = o_.stereo == 1 ? o_.margin / 2 : o_.margin;
            place(frame, pitch, hw, h, mx, stream);
            place(frame + (size_t)hw * 4, pitch, hw, h, mx, stream);
        } else
            place(frame, pitch, w, h, o_.margin, stream);
    }

private:
    void place(CUdeviceptr frame, size_t pitch, int w, int h, int mx, CUstream stream) {
        if (!have_ || ow_ > w || oh_ > h) return;
        int x = mx, y = o_.margin;
        const std::string& p = o_.position;
        bool right = p.find("derech") != std::string::npos || p.find("right") != std::string::npos;
        bool bottom = p.find("abajo") != std::string::npos || p.find("bottom") != std::string::npos;
        if (right) x = w - ow_ - mx;
        if (bottom) y = h - oh_ - o_.margin;
        x = std::clamp(x, 0, w - ow_);
        y = std::clamp(y, 0, h - oh_);
        CUDA_MEMCPY2D cp{};
        cp.srcMemoryType = CU_MEMORYTYPE_HOST;
        cp.srcHost = img_.data();
        cp.srcPitch = (size_t)ow_ * 4;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.dstDevice = frame + (size_t)y * pitch + (size_t)x * 4;
        cp.dstPitch = pitch;
        cp.WidthInBytes = (size_t)ow_ * 4;
        cp.Height = (size_t)oh_;
        // memoria de CPU normal: la llamada vuelve cuando ya ha copiado img_, así que el lock basta
        g_cuda.cuMemcpy2DAsync(&cp, stream);
    }

    bool openDev() {
        fd_ = ::open(o_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) return fail(std::string("no se puede abrir ") + o_.device + ": " + strerror(errno));
        v4l2_format f{};
        f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        auto tryFmt = [&](uint32_t fourcc) {
            memset(&f, 0, sizeof f);
            f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            f.fmt.pix.width = (uint32_t)o_.capW;
            f.fmt.pix.height = (uint32_t)o_.capH;
            f.fmt.pix.pixelformat = fourcc;
            f.fmt.pix.field = V4L2_FIELD_ANY;
            return ioctl(fd_, VIDIOC_S_FMT, &f) == 0 && f.fmt.pix.pixelformat == fourcc;
        };
        // YUYV no necesita decodificar; MJPEG para cámaras que sólo lo dan a esa resolución/fps
        if (tryFmt(V4L2_PIX_FMT_YUYV) && (int)f.fmt.pix.width >= o_.capW / 2) mjpeg_ = false;
        else if (tryFmt(V4L2_PIX_FMT_MJPEG)) mjpeg_ = true;
        else if (tryFmt(V4L2_PIX_FMT_YUYV)) mjpeg_ = false;
        else return fail("la cámara no ofrece YUYV ni MJPEG");
        capW_ = (int)f.fmt.pix.width;
        capH_ = (int)f.fmt.pix.height;
        stride_ = (int)f.fmt.pix.bytesperline ? (int)f.fmt.pix.bytesperline : capW_ * 2;

        v4l2_streamparm sp{};
        sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        sp.parm.capture.timeperframe.numerator = 1;
        sp.parm.capture.timeperframe.denominator = (uint32_t)o_.fps;
        ioctl(fd_, VIDIOC_S_PARM, &sp);   // si no lo admite, se queda con el que tenga

        v4l2_requestbuffers rb{};
        rb.count = 4;
        rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        rb.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &rb) < 0 || rb.count < 2) return fail("VIDIOC_REQBUFS falló");
        for (uint32_t i = 0; i < rb.count; i++) {
            v4l2_buffer b{};
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            b.memory = V4L2_MEMORY_MMAP;
            b.index = i;
            if (ioctl(fd_, VIDIOC_QUERYBUF, &b) < 0) return fail("VIDIOC_QUERYBUF falló");
            void* p = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
            if (p == MAP_FAILED) return fail("mmap falló");
            bufs_.push_back({p, b.length});
            if (ioctl(fd_, VIDIOC_QBUF, &b) < 0) return fail("VIDIOC_QBUF falló");
        }
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) return fail("VIDIOC_STREAMON falló (¿la usa otro programa?)");

        // tamaño en el stream: % del ancho, conservando la proporción de la cámara
        // en 3D, % del ancho de un ojo tal como se ve (half SBS: el ojo se estira al doble, así que la
        // imagen se guarda comprimida a la mitad de ancho para que al verla tenga su proporción)
        int eyeW = o_.stereo == 2 ? sw_ / 2 : sw_;
        int ow = std::max(16, (eyeW * o_.sizePct / 100) & ~1);
        int oh = std::max(16, (ow * capH_ / capW_) & ~1);
        if (oh > sh_ / 2) { oh = (sh_ / 2) & ~1; ow = (oh * capW_ / capH_) & ~1; }
        if (o_.stereo == 1) ow = std::max(8, (ow / 2) & ~1);
        {
            std::lock_guard<std::mutex> l(m_);
            ow_ = ow;
            oh_ = oh;
            img_.assign((size_t)ow * oh * 4, 0);
            have_ = false;
        }
        work_.assign((size_t)ow * oh * 4, 0);
        LOG("cámara: %s %dx%d %s -> %dx%d en %s%s", o_.device.c_str(), capW_, capH_, mjpeg_ ? "MJPEG" : "YUYV", ow, oh,
            o_.position.c_str(), o_.stereo ? " (3D: en los dos ojos)" : "");
        lastError_.clear();
        return true;
    }

    void closeDev() {
        if (fd_ >= 0) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
        }
        for (auto& b : bufs_) munmap(b.first, b.second);
        bufs_.clear();
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        std::lock_guard<std::mutex> l(m_);
        have_ = false;   // sin cámara: el stream sigue sin superposición
    }

    bool fail(const std::string& e) {
        if (e != lastError_) LOG("cámara: %s (se reintentará)", e.c_str());
        lastError_ = e;
        return false;
    }

    void process(const uint8_t* data, size_t size) {
        YuvView v;
        if (mjpeg_) {
            if (!dec_.decode(data, size, v)) return;
        } else {
            if (size < (size_t)stride_ * capH_) return;
            v.w = capW_;
            v.h = capH_;
            v.yuyv = true;
            v.p[0] = data;
            v.stride[0] = stride_;
        }
        int ow, oh;
        {
            std::lock_guard<std::mutex> l(m_);
            ow = ow_;
            oh = oh_;
        }
        yuvToBgra(v, work_.data(), ow, oh, ow * 4, o_.mirror, o_.border);
        std::lock_guard<std::mutex> l(m_);
        img_.swap(work_);
        have_ = true;
    }

    void loop() {
        while (!quit_) {
            if (!openDev()) {
                closeDev();
                for (int i = 0; i < 30 && !quit_; i++) usleep(100000);   // reintento cada 3 s
                continue;
            }
            while (!quit_) {
                fd_set fs;
                FD_ZERO(&fs);
                FD_SET(fd_, &fs);
                timeval tv{0, 200000};
                int r = select(fd_ + 1, &fs, nullptr, nullptr, &tv);
                if (r < 0 && errno != EINTR) break;
                if (r <= 0) continue;
                v4l2_buffer b{};
                b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                b.memory = V4L2_MEMORY_MMAP;
                if (ioctl(fd_, VIDIOC_DQBUF, &b) < 0) {
                    if (errno == EAGAIN) continue;
                    fail(std::string("se ha desconectado (") + strerror(errno) + ")");
                    break;
                }
                if (b.index < bufs_.size() && !(b.flags & V4L2_BUF_FLAG_ERROR))
                    process((const uint8_t*)bufs_[b.index].first, b.bytesused);
                ioctl(fd_, VIDIOC_QBUF, &b);
            }
            closeDev();
        }
        closeDev();
    }

    CameraOptions o_;
    int sw_, sh_;
    std::thread thr_;
    std::atomic<bool> quit_{false};
    int fd_ = -1;
    bool mjpeg_ = false;
    int capW_ = 0, capH_ = 0, stride_ = 0;
    std::vector<std::pair<void*, size_t>> bufs_;
    MjpegDecoder dec_;
    std::string lastError_;

    std::mutex m_;
    std::vector<uint8_t> img_, work_;
    int ow_ = 0, oh_ = 0;
    bool have_ = false;
};
#endif
