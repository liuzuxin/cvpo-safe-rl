# depends on the system and needs
sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt install libglew-dev
