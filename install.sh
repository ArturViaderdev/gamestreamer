#!/bin/sh
# Compila e instala gamestream en /usr (capa 64 bits + 32 bits si es posible).
#   ./install.sh            -> compila e instala (pide sudo para instalar)
#   ./install.sh --deps     -> instala antes las dependencias de Debian/Ubuntu
set -e
cd "$(dirname "$0")"

if [ "$1" = "--deps" ]; then
    sudo apt-get install -y build-essential cmake ninja-build pkg-config \
        libvulkan-dev libavcodec-dev libavformat-dev libavutil-dev \
        libpulse0 libnotify-bin g++-multilib
fi

ARCH64=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR="lib/$ARCH64"
# --clean-first: recompila siempre (si se copian fuentes con fechas antiguas, ninja no lo detectaría)
cmake --build build --clean-first

BUILD32=0
if echo 'int main(){}' | g++ -m32 -x c++ - -o /tmp/gs32test 2>/dev/null; then
    cmake -S . -B build32 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_DAEMON=OFF \
          -DLAYER_SUFFIX=32 -DCMAKE_CXX_FLAGS=-m32 \
          -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/i386-linux-gnu
    cmake --build build32 --clean-first
    BUILD32=1
else
    echo "AVISO: sin g++-multilib no se compila la capa de 32 bits (juegos de 32 bits no se capturarán)"
fi

sudo ninja -C build install
# reinicia el daemon si estaba en marcha, para usar el binario nuevo
/usr/bin/gamestreamd quit >/dev/null 2>&1 || true
pkill -x gamestreamd 2>/dev/null || true
[ $BUILD32 = 1 ] && sudo ninja -C build32 install

CONF="${XDG_CONFIG_HOME:-$HOME/.config}/gamestream/gamestream.conf"
if [ ! -f "$CONF" ]; then
    mkdir -p "$(dirname "$CONF")"
    cp gamestream.conf.example "$CONF"
    echo "Config creada en $CONF: pon tu clave de emisión ahí."
fi
systemctl --user daemon-reload 2>/dev/null || true
systemctl --user is-enabled gamestreamd >/dev/null 2>&1 && systemctl --user restart gamestreamd
echo
echo "Listo. Para lanzar un juego:  gamestream ./juego   |  Steam: gamestream %command%"
echo "Arranque automático del daemon: systemctl --user enable --now gamestreamd"
