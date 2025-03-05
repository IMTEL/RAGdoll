# Quick examples
-----------------------------------------------
feat: new feature
fix(scope): bug in scope
feat!: breaking change / feat(scope)!: rework API
chore(deps): update dependencies

# Commit types
------------------------------------------------
build: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)

ci: Changes to CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)

chore: Changes which doesn't change source code or tests e.g. changes to the build process, auxiliary tools, libraries

docs: Documentation only changes

feat: A new feature

fix: A bug fix

perf: A code change that improves performance

refactor: A code change that neither fixes a bug nor adds a feature

revert: Revert something

style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)

test: Adding missing tests or correcting existing tests


# Branching

Branch by using the following command:
```bash
git checkout <branch-name>
```

When creating a new branch, use:
```bash
git checkout -b branch_type/branch-name
```
Note: *Branch types are the same as the commit types, such as Feat, Fix, or Refactor etc.*


## Git flow
Git Flow is a branching model that structures your repository to support parallel development and systematic releases. Here’s an overview:

**Main Branches**:
- **master/main:** Contains production-ready code.
- **develop:** Contains the latest development changes that are ready for the next release.

**Supporting Branches:**

**Feature branches:** Used to develop new features. Typically branch off from develop and merge back into it.
```bash
git checkout -b Feat/my-new-feature 
```

**Release branches:** Prepare for a new production release. They allow for final polishing before merging into master/main and develop.

```bash
git checkout -b Release/1.0
```
**Hotfix branches:** Quick fixes applied directly to master/main to address issues in production. After the fix, merge back into both master/main and develop.
```bash
git checkout -b Hotfix/urgent-fix-something
```

Note: *A branch should not be alive too long, as it is important that the entire team works on the same source of truth.*

## Trunk Based Development
Trunk Based Development (TBD) is a simpler alternative where developers work on a single branch (often called trunk or main). Key aspects include:

- **Frequent Integration:** Developers commit small, incremental changes frequently to avoid large merge conflicts.
- **Feature Flags:** Instead of long-lived branches, use feature flags to control the release of incomplete features.
- **Continuous Integration:** Automated builds and tests are essential to ensure that the trunk remains in a deployable state at all times.

### Best Practices for TBD

**Short-Lived Branches:** If branches are needed, keep them very short-lived and merge back into the trunk quickly.

**Regular Commits:** Commit changes frequently to the trunk to reduce divergence.

**Automated Testing:** Use CI/CD pipelines to ensure that every commit is verified by automated tests.

