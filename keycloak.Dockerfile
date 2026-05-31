FROM quay.io/keycloak/keycloak:26.0

COPY --chown=keycloak:keycloak keycloak/realm-ragdoll.json /opt/keycloak/ragdoll-import/realm-ragdoll.json
COPY --chown=keycloak:keycloak keycloak-entrypoint.sh /opt/keycloak/bin/ragdoll-entrypoint.sh

USER root
RUN sed -i 's/\r$//' /opt/keycloak/bin/ragdoll-entrypoint.sh \
    && chmod 0555 /opt/keycloak/bin/ragdoll-entrypoint.sh
USER keycloak

ENTRYPOINT ["/opt/keycloak/bin/ragdoll-entrypoint.sh"]
