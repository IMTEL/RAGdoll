#!/bin/sh
set -eu

mkdir -p /opt/keycloak/data/import
chmod u+w /opt/keycloak/data/import/realm-ragdoll.json 2>/dev/null || true
cp /opt/keycloak/ragdoll-import/realm-ragdoll.json /opt/keycloak/data/import/realm-ragdoll.json
chmod 0444 /opt/keycloak/data/import/realm-ragdoll.json 2>/dev/null || true

exec /opt/keycloak/bin/kc.sh "$@"
