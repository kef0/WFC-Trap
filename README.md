# WRC-Trap

## Установка
git clone https://github.com/kef0/WFC-Trap.git

sudo apt install -y python3 python3-pip python3-venv git build-essential cmake libssl-dev libffi-dev pkg-config iw wireless-tools tcpdump hostapd dnsmasq usbutils

pip install -r requirements.txt

git clone https://github.com/aircrack-ng/rtl8812au.git\n
cd rtl8812au\n
make\n
sudo make install\n
sudo modprobe 8812au
