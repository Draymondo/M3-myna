
## Install

On Ubuntu 24.04 LTS or later, install Myna Settings from the PPA:

```
sudo add-apt-repository ppa:canonical-desktop-team/myna
sudo apt install myna-config
```

Myna Settings then sets up the rest: the Dictation app and a speech model. They come from the Snap Store's `latest/edge` channel as [myna](https://snapcraft.io/myna), [myna-parakeet](https://snapcraft.io/myna-parakeet), [myna-whisper](https://snapcraft.io/myna-whisper) and [myna-funasr](https://snapcraft.io/myna-funasr).

Myna works on GNOME, Xfce and KDE Plasma 6.

arm64 is experimental: the speech models run on the CPU, and Parakeet's NVIDIA GPU engine is built for arm64 but has not been tried on hardware yet.
