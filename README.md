# gamestream

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

## Requisitos

- GPU NVIDIA con driver propietario (CUDA + NVENC; `libcuda.so` y `libnvidia-encode.so` vienen con el driver).
- FFmpeg (sólo para audio, FLV y RTMP). NVENC se usa directamente con la API 12.0 negociada con el driver, así que da igual con qué versión de NVENC se compilase el FFmpeg de tu distro. Driver NVIDIA >= 520.
- No hace falta instalar ningún paquete `nvidia-*` de la distro: sólo cabeceras.

## Instalación

```sh
./install.sh --deps      # dependencias de Debian/Ubuntu + compilar + instalar en /usr
```

Instala:
- `/usr/lib/x86_64-linux-gnu/VkLayer_gamestream_64.so` (+ la de 32 bits si hay `g++-multilib`)
- `/usr/share/vulkan/implicit_layer.d/VkLayer_gamestream_{64,32}.json`
- `/usr/bin/gamestreamd`, `/usr/bin/gamestream`, servicio de usuario `gamestreamd.service`

Se instala en `/usr` a propósito: el contenedor de Steam (pressure-vessel) importa las capas de ahí. La capa sólo depende de libc (libstdc++ va enlazada estáticamente), así que funciona dentro del runtime de Steam.

## Configuración

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

Con `ambos` se emite a Twitch y YouTube a la vez. El vídeo se codifica una vez por cada tasa de bits distinta (con la misma tasa, una sola vez) y cada destino tiene su propia conexión e hilo de red: si uno se cae o va lento, el otro sigue y recibes una notificación. `gamestreamd config` muestra a dónde se va a emitir (con las claves ocultas) y `gamestreamd status` el estado de cada destino durante la emisión.

Se relee cada vez que empiezas a emitir. Para probar sin emitir: `service = custom` y `url = ~/prueba.mkv`. Con `record = ~/Vídeos/partida.mkv` además graba en local mientras emite.

## Uso

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

## Detalles de rendimiento

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
- Sincronía audio/vídeo: el vídeo usa el reloj del sistema y el audio el de la tarjeta de sonido. La deriva entre ambos (decenas de ppm, ~0,2-0,4 s por hora) se corrige continuamente remuestreando el audio de forma inaudible (máx. ±0,5 %); en simulación de 2 h el desfase se mantiene por debajo de 5 ms. El log muestra el desfase cada 5 minutos.

## Cámara y micrófono (opcionales)

```ini
microphone = yes            # se mezcla con el audio del juego
microphone_volume = 100
camera = yes                # cámara en una esquina
camera_size = 22            # % del ancho del vídeo
camera_position = abajo-derecha
```

- **Micrófono**: se captura en su propio hilo, con su propia corrección de reloj, y se mezcla con el audio del juego (cada uno con su volumen). Con `audio = none` sólo se emite el micrófono. Usa auriculares para que el micro no recoja el sonido del juego.
- **Cámara**: se lee directamente con V4L2 (YUYV o MJPEG), se escala en CPU al tamaño pequeño (≈1,5-4 ms por frame) y se copia a la esquina del frame en VRAM justo antes de NVENC. El juego no se ve afectado. Con juegos 3D lado a lado (la capa 3D Vision) la cámara se pone en la misma esquina de los dos ojos, a la profundidad de la pantalla y con su proporción correcta en half SBS; `camera_size` es entonces el % del ancho de un ojo. Si la cámara no está, está ocupada por otro programa o se desconecta, el directo sigue sin ella y se reintenta cada 3 s.
- `gamestreamd config` muestra si están activados.

## Limitaciones actuales

- Sólo NVIDIA (CUDA/NVENC). Para AMD/Intel habría que cambiar la importación por VAAPI (dmabuf) en el daemon; la capa ya sirve.
- El hotkey usa X11/XWayland (`XQueryKeymap`): los juegos Proton y la mayoría de nativos funcionan; en juegos Wayland nativos usa `gamestreamd toggle` con un atajo del escritorio.
- Sólo se captura un juego a la vez (aunque se emita a varios destinos) (el último que se conectó o el que recibió el hotkey).
- Si el juego deja de presentar (pantalla de carga congelada), el vídeo se detiene hasta el siguiente frame.

## Prueba sin NVIDIA

`test/fake_daemon.cpp` hace de daemon, importa la memoria exportada por la capa en otro proceso Vulkan, lee los píxeles y guarda PPM. Se usó para validar la capa con lavapipe + Xvfb + `VK_LAYER_KHRONOS_validation` (sin errores de validación, 30,2 fps exactos con objetivo 30, escalado y hotkey verificados).

```sh
g++ -O1 -o build/fake_daemon test/fake_daemon.cpp -lvulkan
build/fake_daemon --autostart &   # o sin --autostart y pulsar Ctrl+Shift+F9 en el juego
GAMESTREAM=1 vkcube
```
