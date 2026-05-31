param(
    [string]$Username = "test@example.com",
    [string]$Password = "test",
    [string]$FirstName = "Test",
    [string]$LastName = "User",
    [string]$BackendUrl = "http://localhost:8000",
    [string]$Realm = "ragdoll",
    [string]$KeycloakContainer = "ragdoll-keycloak"
)

$ErrorActionPreference = "Stop"

function Invoke-KeycloakAdmin {
    param([string[]]$ArgsList)
    docker exec $KeycloakContainer /opt/keycloak/bin/kcadm.sh @ArgsList
}

Invoke-KeycloakAdmin @(
    "config",
    "credentials",
    "--server",
    "http://localhost:8080",
    "--realm",
    "master",
    "--user",
    "admin",
    "--password",
    "admin"
)

$clientJson = Invoke-KeycloakAdmin @(
    "get",
    "clients",
    "-r",
    $Realm,
    "-q",
    "clientId=ragdoll-config",
    "--fields",
    "id"
)
$clients = $clientJson | ConvertFrom-Json
if ($clients.Count -gt 0) {
    Invoke-KeycloakAdmin @(
        "update",
        "clients/$($clients[0].id)",
        "-r",
        $Realm,
        "-s",
        "publicClient=false",
        "-s",
        "secret=ragdoll-config-secret"
    )
}

$existingUsersJson = Invoke-KeycloakAdmin @(
    "get",
    "users",
    "-r",
    $Realm,
    "-q",
    "username=$Username"
)
$existingUsers = $existingUsersJson | ConvertFrom-Json

if ($existingUsers.Count -gt 0) {
    $userId = $existingUsers[0].id
    Write-Host "Keycloak user already exists: $Username ($userId)"
} else {
    Invoke-KeycloakAdmin @(
        "create",
        "users",
        "-r",
        $Realm,
        "-s",
        "username=$Username",
        "-s",
        "email=$Username",
        "-s",
        "firstName=$FirstName",
        "-s",
        "lastName=$LastName",
        "-s",
        "enabled=true",
        "-s",
        "emailVerified=true"
    )

    $createdUsersJson = Invoke-KeycloakAdmin @(
        "get",
        "users",
        "-r",
        $Realm,
        "-q",
        "username=$Username"
    )
    $createdUsers = $createdUsersJson | ConvertFrom-Json
    $userId = $createdUsers[0].id

    Invoke-KeycloakAdmin @(
        "set-password",
        "-r",
        $Realm,
        "--username",
        $Username,
        "--new-password",
        $Password
    )
    Write-Host "Created Keycloak user: $Username ($userId)"
}

$body = @{
    provider_user_id = $userId
    email = $Username
    name = "$FirstName $LastName"
    attach_all_agents = $true
} | ConvertTo-Json

$bootstrapResult = Invoke-RestMethod `
    -Method Post `
    -Uri "$BackendUrl/api/debug/bootstrap-keycloak-user" `
    -ContentType "application/json" `
    -Body $body

Write-Host "Attached local RAGdoll data to Keycloak user:"
$bootstrapResult | ConvertTo-Json
