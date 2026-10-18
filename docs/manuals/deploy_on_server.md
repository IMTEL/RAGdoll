

server name: iplvr01.os.it.ntnu.no
accessed at: https://iplvr.it.ntnu.no

Service DNS: iplvr.it.ntnu.no

Traffic is allowed from NTNU’s internal network to reach your Docker containers on ports 80 (HTTP) and 443 (HTTPS).

## Where the data lives

All persistent data is in named Docker volumes (see `docker-compose.yml`):

| Volume | Contents |
|---|---|
| `ragdoll_keycloak_data` | Keycloak: user accounts, passwords, realm settings |
| `ragdoll_mongodb_data` | MongoDB: agents, RAGdoll user records, documents, chunks, progress |
| `ragdoll_faiss_indices` | FAISS vector indexes (can be rebuilt from MongoDB with `scripts/rebuild_faiss_index.py --all`) |

`docker compose down` and `docker compose up -d --build` (what the deploy does) keep these volumes. **Never run `docker compose down -v` or `docker volume prune` on the server**: that deletes them.

Keycloak imports `keycloak/realm-ragdoll.json` only when the `ragdoll` realm does not exist yet. On an existing server the import is skipped, so users are never overwritten, but changes to that file must be applied by hand in the Keycloak admin console.

## Back up before every deploy

Run from the directory containing `docker-compose.yml` on the server:

```sh
# 1. MongoDB (consistent online dump)
docker compose exec mongodb mongodump --archive=/tmp/pre-deploy.archive \
  -u admin -p "$MONGODB_ROOT_PASSWORD" --authenticationDatabase admin
docker compose cp mongodb:/tmp/pre-deploy.archive ./backup-$(date +%F)-mongo.archive

# 2. Keycloak users and realm (stop it briefly for a clean copy)
docker compose stop keycloak
docker run --rm -v ragdoll_keycloak_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/backup-$(date +%F)-keycloak.tgz -C /data .
docker compose start keycloak
```

Keep the files somewhere other than the server as well.

## Restore

```sh
# MongoDB: replaces the dumped databases
docker compose cp ./backup-YYYY-MM-DD-mongo.archive mongodb:/tmp/restore.archive
docker compose exec mongodb mongorestore --drop --archive=/tmp/restore.archive \
  -u admin -p "$MONGODB_ROOT_PASSWORD" --authenticationDatabase admin

# Keycloak: replaces all Keycloak data
docker compose stop keycloak
docker run --rm -v ragdoll_keycloak_data:/data -v "$PWD":/backup alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/backup-YYYY-MM-DD-keycloak.tgz -C /data"
docker compose start keycloak
```
