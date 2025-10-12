# NTNU Project Server Setup Report — *iplvr.it.ntnu.no*

**Project:** RAGdoll  
**Server (host):** `iplvr01.os.it.ntnu.no`  
**Service (DNS):** `iplvr.it.ntnu.no`  
**Date:** 2025-10-12  
**Prepared by:** Tobia F. with guidance from ChatGPT  

---

## 🧭 Overview

This document summarizes the complete setup of the **halvadministrert Ubuntu server** provided by **NTNU IT**, configured to host the **RAGdoll** FastAPI + MongoDB + Docker application.

---

## 🧩 1. Initial Information from NTNU IT

| Item | Value |
|------|--------|
| Service name (DNS) | `iplvr.it.ntnu.no` |
| Server hostname | `iplvr01.os.it.ntnu.no` |
| Admin group | `halvadm_iplvr_sudo` |
| SSH group | `halvadm_iplvr_ssh` |
| Certificate location | `/root/iplvr.it.ntnu.no.crt` and `/root/iplvr.it.ntnu.no.key` |
| Wiki reference | [Halvadministrerte linux-servere – NTNU](https://www.ntnu.no/wiki/spaces/ntnuitubuntu/pages/251003154/Halvadministrerte+linux-servere) |

---

## ⚙️ 2. SSH Access

Connected using the NTNU username:

```bash
ssh <username>@iplvr01.os.it.ntnu.no
```

## 3. System Verification

Checked running services to confirm this is a standard NTNU base image:
```bash
sudo systemctl list-units --type=service --state=running
```

Found monitoring (Munin, pmcd), networking (chrony, rpcbind), and base daemons running as expected.

Confirmed presence of NTNU management scripts:

```bash
/local/admin/bin/do_pkgsync.sh
/local/admin/bin/install-firewall.sh
```


## 4. Docker Installation (NTNU-compliant)

NTNU servers automatically remove manually installed packages,
so Docker was installed permanently through pkgsync.

```bash
sudo tee /etc/pkgsync/required-packages-docker > /dev/null <<'EOF'
docker.io
EOF
sudo /local/admin/bin/do_pkgsync.sh
```

Installed Docker Compose v2 manually (safe outside package management):
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version  # → v2.40.0
```

## 5. Firewall Configuration (NTNU Rules)

Created custom IPv4 firewall definition for Docker:
```bash
sudo tee /etc/local/firewall.d/ipv4-tobif-docker.conf > /dev/null <<'EOF'
-I DOCKER-USER -p tcp -m conntrack --ctorigdstport 80 -j permit_ntnu
-I DOCKER-USER -p tcp -m conntrack --ctorigdstport 443 -j permit_ntnu
EOF

sudo /local/admin/bin/install-firewall.sh
```

verification
```bash
sudo iptables-save | grep DOCKER-USER
```


## 6. SSL Certificate Verification

```bash
/root/iplvr.it.ntnu.no.crt   (4214 bytes)
 /root/iplvr.it.ntnu.no.key   (241 bytes)
```

## 7. Application Deployment Structure

/home/<user>/iplvr-app/RAGdoll

Created production-ready Compose and Nginx configuration.


## 8. Deployment & Runtime

start services: 
```bash
sudo docker-compose up -d
sudo docker ps
``` 

## 9. Troubleshooting
Port conflicts

Initially, system Nginx occupied port 80 → resolved by:

```bash
sudo systemctl stop nginx
sudo systemctl disable nginx
```

502 Bad Gateway

Indicates Nginx could not reach chat-service.
Confirm FastAPI binds to 0.0.0.0:8000 inside container.

Command to check logs:

```bash
sudo docker logs ragdoll-backend --tail 30
```

## 10. Auto-Start After Reboot

Enabled Docker and added a cron entry

```bash
sudo systemctl enable docker
sudo crontab -e
@reboot cd /home/<user>/iplvr-app/RAGdoll && /usr/local/bin/docker-compose up -d
```

