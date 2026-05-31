// =============================================================================
// MongoDB Initialization Script for RAGdoll Local Development
// =============================================================================
// This script runs automatically when MongoDB container first starts
// Creates database, collections, and indexes according to RAGdoll schema

// Switch to application database
db = db.getSiblingDB('ragdoll_dev');

print('==========================================');
print('Initializing RAGdoll Local Database');
print('==========================================');

// =============================================================================
// Users Collection
// =============================================================================
print('Creating users collection...');
db.createCollection('users');

// Create indexes for users
db.users.createIndex({ "id": 1 }, { unique: true, sparse: true });
db.users.createIndex({ "email": 1 });
db.users.createIndex({ "auth_provider": 1, "provider_user_id": 1 }, { unique: true });

print('[OK] Users collection created with indexes');

// =============================================================================
// Agents Collection
// =============================================================================
print('Creating agents collection...');
db.createCollection('agents');

// Create indexes for agents
db.agents.createIndex({ "id": 1 }, { unique: true });
db.agents.createIndex({ "name": 1 });
db.agents.createIndex({ "created_at": -1 });

print('[OK] Agents collection created with indexes');

// =============================================================================
// Documents Collection
// =============================================================================
print('Creating documents collection...');
db.createCollection('documents');

// Create indexes for documents
// Note: documents use _id as primary identifier, no separate id field needed
db.documents.createIndex({ "agent_id": 1 });
db.documents.createIndex({ "created_at": -1 });
db.documents.createIndex({ "agent_id": 1, "name": 1 }, { unique: true });

print('[OK] Documents collection created with indexes');

// =============================================================================
// Contexts Collection
// =============================================================================
print('Creating contexts collection...');
db.createCollection('context');

// Create indexes for contexts
// Note: contexts use chunk_id as identifier, not id field
db.context.createIndex({ "chunk_id": 1 }, { unique: true, sparse: true });
db.context.createIndex({ "agent_id": 1 });
db.context.createIndex({ "document_id": 1 });
db.context.createIndex({ "created_at": -1 });

// Text search index for context content
db.context.createIndex({ "text": "text" });

print('[OK] Contexts collection created with indexes');

// =============================================================================
// Development User (for DISABLE_AUTH mode)
// =============================================================================
print('Creating development user...');

// Create dev user with fixed ObjectId
db.users.insertOne({
    _id: ObjectId("000000000000000000000001"),
    id: "000000000000000000000001",
    auth_provider: "dev",
    provider_user_id: "dev-provider-id",
    email: "dev@localhost",
    name: "Development User",
    picture: null,
    owned_agents: [],
    api_keys: []
});

print('[OK] Development user created (ID: 000000000000000000000001)');

// =============================================================================
// Summary
// =============================================================================
print('==========================================');
print('Database Initialization Complete!');
print('==========================================');
print('Database: ragdoll_dev');
print('Collections created:');
print('  - users (with dev user)');
print('  - agents');
print('  - documents');
print('  - context');
print('');
print('Dev user credentials:');
print('  Email: dev@localhost');
print('  ID: 000000000000000000000001');
print('  (Used when DISABLE_AUTH=true)');
print('==========================================');
