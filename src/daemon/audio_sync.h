// Corrección de deriva de reloj del audio.
//
// El vídeo lleva marcas de tiempo del reloj del sistema (CLOCK_MONOTONIC) y el
// audio cuenta muestras del reloj de la tarjeta de sonido. Los dos relojes
// nunca van exactamente igual (decenas de ppm), así que en emisiones largas el
// audio se adelanta o se retrasa poco a poco.
//
// AudioSync compara continuamente "cuánto audio hemos emitido" con "cuánto
// tiempo real ha pasado" y remuestrea muy ligeramente (como mucho ±0,5 %,
// inaudible) para que la diferencia vuelva a cero. Sin saltos ni chasquidos.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

class AudioSync {
public:
    static constexpr double kRate = 48000.0;

    // in: 'frames' muestras estéreo intercaladas, recién leídas.
    // arrivalUs: instante (reloj del sistema, µs desde el inicio) en que llegaron.
    // Añade a 'out' las muestras corregidas (estéreo intercalado).
    void push(const float* in, int frames, int64_t arrivalUs, std::vector<float>& out) {
        if (!started_) {
            started_ = true;
            startUs_ = arrivalUs - (int64_t)(frames * 1e6 / kRate);   // la 1ª muestra se capturó antes
            last_[0] = in[0];
            last_[1] = in[1];
        }
        // Deriva: audio emitido (incluyendo lo que saldrá de este bloque) vs tiempo real
        const double emittedS = (outFrames_ + frames / step_) / kRate;
        const double realS = (arrivalUs - startUs_) / 1e6;
        const double d = emittedS - realS;   // >0: el audio va por delante (sonaría tarde)
        drift_ = nSamples_ < 50 ? d : drift_ + 0.02 * (d - drift_);   // media móvil (~1 s)
        nSamples_++;
        // Control proporcional: corregir la deriva en ~10 s, limitado a ±0,5 %
        step_ = 1.0 + std::clamp(drift_ * 0.1, -0.005, 0.005);

        // Remuestreo lineal: posición fraccionaria sobre [last, in...]
        for (;;) {
            double p = pos_;
            int i = (int)std::floor(p);
            if (i + 1 > frames) break;   // necesitamos in[i] (i-1 es 'last' si i==0)
            double f = p - i;
            for (int c = 0; c < 2; c++) {
                float a = i == 0 ? last_[c] : in[(i - 1) * 2 + c];
                float b = in[i * 2 + c];
                out.push_back((float)(a + (b - a) * f));
            }
            outFrames_++;
            pos_ += step_;
        }
        pos_ -= frames;
        last_[0] = in[(frames - 1) * 2];
        last_[1] = in[(frames - 1) * 2 + 1];
    }

    // Marca de tiempo (en muestras desde el inicio del stream) de la próxima
    // muestra que se entregue
    int64_t startSamples() const { return (int64_t)std::llround(startUs_ * kRate / 1e6); }
    double driftMs() const { return drift_ * 1000.0; }
    double correctionPpm() const { return (step_ - 1.0) * 1e6; }

private:
    bool started_ = false;
    int64_t startUs_ = 0;
    double outFrames_ = 0;
    double pos_ = 0;
    double step_ = 1.0;
    double drift_ = 0;
    int64_t nSamples_ = 0;
    float last_[2] = {0, 0};
};
