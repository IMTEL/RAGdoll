# Git Conventions

## Commit Messages

All commit messages **must** follow the Conventional Commits specification. This ensures clarity and consistency across the project.

### Format
```
<type>(<scope>): <description>
```

- **type**: The type of change (see allowed types below).
- **scope**: The area of the codebase affected (optional).
- **description**: A short, imperative summary of the change.

### Examples
- `feat: add user authentication`
- `fix(auth): resolve login bug`
- `chore(deps): update dependencies`
- `docs: improve README formatting`
- `refactor(api): simplify request handling`

### Breaking Changes
For breaking changes, add a `!` after the type:
- `feat!: introduce new API structure`
- `fix(auth)!: remove deprecated login method`

### Allowed Commit Types
- **feat**: A new feature.
- **fix**: A bug fix.
- **docs**: Documentation-only changes.
- **style**: Code style changes (e.g., formatting, white-space).
- **refactor**: Code changes that neither fix a bug nor add a feature.
- **perf**: Performance improvements.
- **test**: Adding or updating tests.
- **chore**: Maintenance tasks (e.g., updating dependencies).
- **revert**: Reverting a previous commit.

---

## Branching Strategy

### Naming Convention
Use the following format for branch names:
```
<type>/<short-description>
```

- **type**: Same as commit types (e.g., `feat`, `fix`, `chore`).
- **short-description**: A concise description of the branch purpose.

### Examples
- `feat/add-authentication`
- `fix/login-bug`
- `chore/update-dependencies`

### Guidelines
- Keep branches short-lived.
- Regularly merge changes from the main branch to avoid conflicts.

---

## Best Practices

- **Frequent Commits**: Commit small, incremental changes frequently.
- **Descriptive Messages**: Ensure commit messages clearly describe the change.
- **Automated Testing**: Use CI/CD pipelines to verify every commit.
- **Feature Flags**: Use feature flags for incomplete features instead of long-lived branches.

---

By adhering to these conventions, we ensure a clean and maintainable Git history that benefits the entire team.

