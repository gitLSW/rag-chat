### Before installing
```bash
sudo apt update
sudo apt upgrade -y
```

### Install MongoDB
```bash
sudo apt-get install gnupg curl

curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor

echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

sudo apt-get update

sudo apt-get install -y mongodb-org
```

### Start and enable mongoDB
```bash
sudo systemctl start mongod
sudo systemctl enable mongod
```


### Create a virtual enviorment for the server
```bash
python3 -m venv rag-env
source rag-env/bin/activate
pip install --upgrade pip
```

### Install pyTorch
Check the instructions at https://pytorch.org/get-started/locally/
To get the CUDA version run: `nvidia-smi`
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Install server requirements
```bash
pip install -r requirements.txt
```

### Open server port to the internet (check the .env file)
```bash
sudo ufw allow 7500/tcp
sudo ufw reload
```

### Start the server:
```bash
bash start_server.sh
```