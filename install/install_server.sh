# Important step before installing mongoDB
sudo apt update
sudo apt upgrade -y

# Install MongoDB
sudo apt-get install gnupg curl

curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt-get update

sudo apt-get install -y mongodb-org

# Start and enable mongoDB
sudo systemctl start mongod
sudo systemctl enable mongod


## Option 1: Installing using a venv
python3 -m venv rag-env
source rag-env/bin/activate

# ## Option 2: Installing as a pip user
# # Remove tensorflow, because some packages (e.g.: transformers) will try to use it, but shouldn't
# sudo apt remove python3-tensorflow-cuda -y
# sudo apt autoremove -y

# # To deal with broken scikit-learn ionstall, install numpy 2 
# pip install --no-binary :all: scipy scikit-learn
# # Now torchvision will ause problems... Fuck this approach, just use a venv

# Install python server
pip install --upgrade pip
# Get your CUDA version from: 'nvidia-smi'
# Check the instructions at https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt