# install mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
rm -r mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200

cd ~/.mujoco
wget https://www.roboti.us/file/mjkey.txt


export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.mujoco/mujoco200/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:~/.mujoco/mujoco200/bin" >> ~/.bashrc
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.bashrc
