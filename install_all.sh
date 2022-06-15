# setup safety gym
cd envs/safety-gym
bash setup_mujoco.sh
source ~/.bashrc
pip install -e .
cd ../..
pip install -r requirement.txt
pip install -e .

echo "********************************************************"
echo "********************************************************"
echo "********************************************************"
echo "                                                        "
echo "Please install pytorch manually to finish the env setup."
echo "                                                        "
echo "********************************************************"
echo "********************************************************"
echo "********************************************************"