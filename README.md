# gamestream

**[Versión en español más abajo](#versión-en-español)**

Live streaming of Vulkan games on Linux (Twitch / YouTube) with the lowest possible performance impact.

An **implicit Vulkan layer** copies the game image GPU→GPU at `vkQueuePresentKHR` time, and a **daemon** imports it into CUDA (same VRAM, never touching the CPU) and encodes it with **NVENC**. It works with native games, Proton/DXVK/VKD3D (DirectX 9-12 through Vulkan) and standalone Proton. OpenGL is not supported.

```
 Game process                                        gamestreamd
┌──────────────────────────────┐   UNIX socket   ┌─────────────────────────────────┐
│ game → vkQueuePresentKHR     │   (VRAM fds)    │ cuImportExternalMemory (once)   │
│   └ layer: GPU copy/scale    │ ──────────────► │ VRAM→VRAM copy to surface       │
│     into exportable VRAM buf │  "frame ready"  │ [overlays: camera...]           │
│   └ thread: waits the fence  │ ◄────────────── │ NVENC (HW encoder) H.264/HEVC   │
│   └ thread: X11 hotkey+socket│  "buffer free"  │ + PulseAudio/PipeWire audio AAC │
└──────────────────────────────┘                 │ → FLV → RTMP (Twitch/YouTube)   │
                                                 └─────────────────────────────────┘
```

What it **never** does on the game thread: read pixels back to the CPU, wait for the GPU, encode, or touch the network. If the daemon falls behind, the layer drops frames instead of slowing the game down. While not streaming, the layer only adds one check per frame.

## Requirements

- NVIDIA GPU with the proprietary driver (CUDA + NVENC; `libcuda.so` and `libnvidia-encode.so` ship with the driver).
- FFmpeg (only for audio encoding, FLV and RTMP). NVENC is used directly through API 12.0, negotiated with the driver at runtime, so it does not matter which NVENC version your distro's FFmpeg was built against. NVIDIA driver >= 520.
- No distro `nvidia-*` packages are needed: headers only.

## Installation

```sh
./install.sh --deps      # Debian/Ubuntu dependencies + build + install to /usr
```

It installs:
- `/usr/lib/x86_64-linux-gnu/VkLayer_gamestream_64.so` (+ the 32-bit one if `g++-multilib` is present)
- `/usr/share/vulkan/implicit_layer.d/VkLayer_gamestream_{64,32}.json`
- `/usr/bin/gamestreamd`, `/usr/bin/gamestream`, user service `gamestreamd.service`

It installs to `/usr` on purpose: Steam's container (pressure-vessel) imports layers from there. The layer only depends on libc (libstdc++ is linked statically), so it works inside the Steam runtime.

## Configuration

`~/.config/gamestream/gamestream.conf` (the installer copies `gamestream.conf.example`):

```ini
service = both            # twitch | youtube | both | custom
twitch_key = live_xxxxxxxx
youtube_key = xxxx-xxxx-xxxx-xxxx
twitch_bitrate = 6000     # optional, per service
youtube_bitrate = 12000
resolution = 1920x1080    # or "auto" (the game's resolution)
fps = 60
hotkey = Ctrl+Shift+F9
```

With `both` it streams to Twitch and YouTube at the same time. Video is encoded once per distinct bitrate (once in total if they match), and each destination has its own connection and network thread: if one drops or lags, the other keeps going and you get a notification. `gamestreamd config` shows where it will stream (keys masked) and warns about unknown options; `gamestreamd status` shows each destination's state while live.

The config is re-read every time you start streaming. To test without going live: `service = custom` and `url = ~/test.mkv`. With `record = ~/Videos/match.mkv` it also records locally while streaming.

## Usage

```sh
systemctl --user enable --now gamestreamd     # or simply: gamestreamd &
gamestream ./game                              # native game
gamestream proton run game.exe                 # standalone Proton
```

In Steam → Properties → Launch options: `gamestream %command%`

During the game press **Ctrl+Shift+F9** to start/stop. It can also be controlled from outside the game (handy as a global desktop shortcut, or for native Wayland games):

```sh
gamestreamd toggle | start | stop | status | config
```

Daemon log: `~/.local/state/gamestream/gamestreamd.log` (includes FFmpeg/NVENC errors). `GAMESTREAM_DEBUG=1` prints more information in the game's output. `DISABLE_GAMESTREAM=1` disables the layer.

## Performance details

| Step | Where | Cost |
|---|---|---|
| Swapchain copy/scale | game queue, chained with semaphores | ~0.05-0.2 ms of GPU per streamed frame |
| Waiting for the copy | the layer's own thread | 0 on the game thread |
| Copy to NVENC surface | daemon, CUDA, VRAM→VRAM | ~0.05 ms |
| Encoding | NVENC chip (independent from shaders) | ~0% 3D performance |
| AAC audio + network | daemon (CPU, own threads) | negligible |

- Only the stream's frame rate is captured: if the game runs at 144 fps and you stream at 60, 60 are copied.
- If the swapchain is already BGRA and the stream's size, it is copied directly (no blit).
- With a different resolution it is scaled on the GPU with linear filtering, keeping the aspect ratio.
- HDR/10-bit formats are converted to 8 bits without tone mapping.
- Audio/video sync: video uses the system clock and audio uses the sound card clock. The drift between them (tens of ppm, ~0.2-0.4 s per hour) is corrected continuously by inaudibly resampling the audio (max ±0.5%); in a 2-hour simulation the offset stays below 5 ms. The sound system's capture latency is measured and compensated as well. The log shows the offset every 5 minutes; `audio_delay` (ms, negative = earlier) allows a fixed manual lip-sync adjustment.

## Camera and microphone (optional)

```ini
microphone = yes            # mixed with the game audio
microphone_volume = 100
camera = yes                # webcam in a corner
camera_size = 22            # % of the video width
camera_position = bottom-right
camera_mirror = no
```

- **Microphone**: captured on its own thread with its own clock correction and mixed with the game audio (each with its own volume). With `audio = none` only the microphone is streamed. Use headphones so the mic does not pick up the game sound.
- **Camera**: read directly through V4L2 (YUYV or MJPEG), scaled on the CPU to the small size (≈1.5-4 ms per frame) and copied into the frame's corner in VRAM right before NVENC. The game is not affected. If the camera is missing, busy in another program or gets unplugged, the stream continues without it and it retries every 3 s.
- `gamestreamd config` shows whether they are enabled.

## Current limitations

- NVIDIA only (CUDA/NVENC). For AMD/Intel the daemon's import would have to switch to VAAPI (dmabuf); the layer already works.
- The hotkey uses X11/XWayland (`XQueryKeymap`): Proton games and most native ones work; for native Wayland games use `gamestreamd toggle` with a desktop shortcut.
- Only one game is captured at a time, even when streaming to several destinations (the last one that connected or the one that received the hotkey).
- Resolution is shared by all destinations (no 4K to YouTube + 1080p to Twitch at the same time yet).
- If the game stops presenting (frozen loading screen), video pauses until the next frame.

## Testing without NVIDIA

`test/fake_daemon.cpp` acts as the daemon: it imports the memory exported by the layer into another Vulkan process, reads the pixels back and saves PPM images. It was used to validate the layer with lavapipe + Xvfb + `VK_LAYER_KHRONOS_validation` (no validation errors, exactly 30.2 fps with a 30 fps target, scaling and hotkey verified).

```sh
g++ -O1 -o build/fake_daemon test/fake_daemon.cpp -lvulkan
build/fake_daemon --autostart &   # or without --autostart and press Ctrl+Shift+F9 in the game
GAMESTREAM=1 vkcube
```

Building the daemon with `-DGS_FAKE_ENC` replaces NVENC with software x264 and adds `gamestreamd testout <seconds>`, which streams synthetic video plus real audio to the configured destinations (used to test multi-destination streaming, audio mixing and A/V timestamps).

---

## Versión en español

Emisión en directo de juegos Vulkan en Linux (Twitch / YouTube) con el mínimo impacto posible en el rendimiento.

Una **capa Vulkan implícita** copia la imagen del juego GPU→GPU en el momento del `vkQueuePresentKHR` y un **daemon** la importa en CUDA (misma VRAM, sin pasar por la CPU) y la codifica con **NVENC**. Funciona con juegos nativos, Proton/DXVK/VKD3D (DirectX 9-12 a través de Vulkan) y Proton standalone. OpenGL no está soportado.

```
 Proceso del juego                                   gamestreamd
┌──────────────────────────────┐   socket UNIX   ┌─────────────────────────────────┐
│ juego → vkQueuePresentKHR    │  (fds de VRAM)  │ cuImportExternalMemory (1 vez)  │
│   └ capa: copia/escala en GPU│ ──────────────► │ copia VRAM→VRAM a superficie    │
│     a buffer VRAM exportable │  "frame listo"  │ [overlays: cámara...]           │
│   └ hilo: espera la fence    │ ◄────────────── │ NVENC (chip dedicado) H.264/HEVC│
│   └ hilo: hotkey X11 + socket│  "buffer libre" │ + audio PulseAudio/PipeWire AAC │
└──────────────────────────────┘                 │ → FLV → RTMP (Twitch/YouTube)   │
                                                 └─────────────────────────────────┘
```

Lo que **no** hace nunca en el hilo del juego: leer píxeles a la CPU, esperar a la GPU, codificar ni tocar la red. Si el daemon va lento, la capa descarta frames en vez de frenar al juego. Mientras no se emite, la capa sólo añade una comprobación por frame.

### Requisitos

- GPU NVIDIA con driver propietario (CUDA + NVENC; `libcuda.so` y `libnvidia-encode.so` vienen con el driver).
- FFmpeg (sólo para audio, FLV y RTMP). NVENC se usa directamente con la API 12.0 negociada con el driver, así que da igual con qué versión de NVENC se compilase el FFmpeg de tu distro. Driver NVIDIA >= 520.
- No hace falta instalar ningún paquete `nvidia-*` de la distro: sólo cabeceras.

### Instalación

```sh
./install.sh --deps      # dependencias de Debian/Ubuntu + compilar + instalar en /usr
```

Instala:
- `/usr/lib/x86_64-linux-gnu/VkLayer_gamestream_64.so` (+ la de 32 bits si hay `g++-multilib`)
- `/usr/share/vulkan/implicit_layer.d/VkLayer_gamestream_{64,32}.json`
- `/usr/bin/gamestreamd`, `/usr/bin/gamestream`, servicio de usuario `gamestreamd.service`

Se instala en `/usr` a propósito: el contenedor de Steam (pressure-vessel) importa las capas de ahí. La capa sólo depende de libc (libstdc++ va enlazada estáticamente), así que funciona dentro del runtime de Steam.

### Configuración

`~/.config/gamestream/gamestream.conf` (el instalador copia `gamestream.conf.example`):

```ini
service = ambos           # twitch | youtube | ambos | custom
twitch_key = live_xxxxxxxx
youtube_key = xxxx-xxxx-xxxx-xxxx
twitch_bitrate = 6000     # opcional, por servicio
youtube_bitrate = 12000
resolution = 1920x1080    # o "auto" (la del juego)
fps = 60
hotkey = Ctrl+Shift+F9
```

Con `ambos` se emite a Twitch y YouTube a la vez. El vídeo se codifica una vez por cada tasa de bits distinta (con la misma tasa, una sola vez) y cada destino tiene su propia conexión e hilo de red: si uno se cae o va lento, el otro sigue y recibes una notificación. `gamestreamd config` muestra a dónde se va a emitir (con las claves ocultas) y avisa de opciones desconocidas; `gamestreamd status` el estado de cada destino durante la emisión.

Se relee cada vez que empiezas a emitir. Para probar sin emitir: `service = custom` y `url = ~/prueba.mkv`. Con `record = ~/Vídeos/partida.mkv` además graba en local mientras emite.

### Uso

```sh
systemctl --user enable --now gamestreamd     # o simplemente: gamestreamd &
gamestream ./juego                             # juego nativo
gamestream proton run juego.exe                # Proton standalone
```

En Steam → Propiedades → Opciones de lanzamiento: `gamestream %command%`

Durante la partida pulsa **Ctrl+Shift+F9** para empezar/parar. También desde fuera del juego (útil como atajo global del escritorio, o en juegos Wayland nativos):

```sh
gamestreamd toggle | start | stop | status | config
```

Log del daemon: `~/.local/state/gamestream/gamestreamd.log` (incluye los errores de FFmpeg/NVENC). `GAMESTREAM_DEBUG=1` muestra más información en la salida del juego. `DISABLE_GAMESTREAM=1` desactiva la capa.

### Detalles de rendimiento

| Paso | Dónde | Coste |
|---|---|---|
| Copia/escalado del swapchain | cola del juego, encadenada con semáforos | ~0,05-0,2 ms de GPU por frame emitido |
| Espera a que acabe la copia | hilo propio de la capa | 0 en el hilo del juego |
| Copia a superficie NVENC | daemon, CUDA, VRAM→VRAM | ~0,05 ms |
| Codificación | chip NVENC (independiente de los shaders) | ~0 % de rendimiento 3D |
| Audio AAC + red | daemon (CPU, hilos propios) | despreciable |

- Sólo se capturan los fps del stream: si el juego va a 144 fps y emites a 60, se copian 60.
- Si el swapchain ya es BGRA y del tamaño del stream, se copia directamente (sin blit).
- Con resolución distinta se escala en GPU con filtro lineal, manteniendo proporción.
- Formatos HDR/10 bits se convierten a 8 bits sin tone-mapping.
- Sincronía audio/vídeo: el vídeo usa el reloj del sistema y el audio el de la tarjeta de sonido. La deriva entre ambos (decenas de ppm, ~0,2-0,4 s por hora) se corrige continuamente remuestreando el audio de forma inaudible (máx. ±0,5 %); en simulación de 2 h el desfase se mantiene por debajo de 5 ms. También se mide y compensa la latencia de captura del sistema de sonido. El log muestra el desfase cada 5 minutos; `audio_delay` (ms, negativo = antes) permite un ajuste manual fijo de lip-sync.

### Cámara y micrófono (opcionales)

```ini
microphone = yes            # se mezcla con el audio del juego
microphone_volume = 100
camera = yes                # cámara en una esquina
camera_size = 22            # % del ancho del vídeo
camera_position = abajo-derecha
camera_mirror = no
```

- **Micrófono**: se captura en su propio hilo, con su propia corrección de reloj, y se mezcla con el audio del juego (cada uno con su volumen). Con `audio = none` sólo se emite el micrófono. Usa auriculares para que el micro no recoja el sonido del juego.
- **Cámara**: se lee directamente con V4L2 (YUYV o MJPEG), se escala en CPU al tamaño pequeño (≈1,5-4 ms por frame) y se copia a la esquina del frame en VRAM justo antes de NVENC. El juego no se ve afectado. Si la cámara no está, está ocupada por otro programa o se desconecta, el directo sigue sin ella y se reintenta cada 3 s.
- `gamestreamd config` muestra si están activados.

### Limitaciones actuales

- Sólo NVIDIA (CUDA/NVENC). Para AMD/Intel habría que cambiar la importación por VAAPI (dmabuf) en el daemon; la capa ya sirve.
- El hotkey usa X11/XWayland (`XQueryKeymap`): los juegos Proton y la mayoría de nativos funcionan; en juegos Wayland nativos usa `gamestreamd toggle` con un atajo del escritorio.
- Sólo se captura un juego a la vez, aunque se emita a varios destinos (el último que se conectó o el que recibió el hotkey).
- La resolución es la misma para todos los destinos (todavía no se puede 4K a YouTube + 1080p a Twitch a la vez).
- Si el juego deja de presentar (pantalla de carga congelada), el vídeo se detiene hasta el siguiente frame.

### Prueba sin NVIDIA

`test/fake_daemon.cpp` hace de daemon, importa la memoria exportada por la capa en otro proceso Vulkan, lee los píxeles y guarda PPM. Se usó para validar la capa con lavapipe + Xvfb + `VK_LAYER_KHRONOS_validation` (sin errores de validación, 30,2 fps exactos con objetivo 30, escalado y hotkey verificados).

```sh
g++ -O1 -o build/fake_daemon test/fake_daemon.cpp -lvulkan
build/fake_daemon --autostart &   # o sin --autostart y pulsar Ctrl+Shift+F9 en el juego
GAMESTREAM=1 vkcube
```

Compilando el daemon con `-DGS_FAKE_ENC` se sustituye NVENC por x264 por software y se añade `gamestreamd testout <segundos>`, que emite vídeo sintético con audio real a los destinos configurados (se usó para probar la emisión a varios destinos, la mezcla de audio y las marcas de tiempo).
