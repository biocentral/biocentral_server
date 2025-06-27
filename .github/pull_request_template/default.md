# Pull Request

## Description

Brief description of what this PR does and why.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes, no api changes)
- [ ] Performance improvement
- [ ] Other (please describe):

## Testing

- [ ] **Tests are passing**: `uv run pytest` passes without errors
- [ ] Added tests for new functionality (if applicable)
- [ ] Existing tests updated for changes (if applicable)
- [ ] Manual testing completed for affected features

## Documentation

- [ ] **Documentation updated** for any user-facing changes
- [ ] Code comments added for complex logic
- [ ] README.md updated (if applicable)
- [ ] CLAUDE.md updated with new commands/workflows (if applicable)

## Changesets

- [ ] **Changeset created**: Used `npx @changesets/cli add` to document changes
- [ ] Appropriate semver level selected (patch/minor/major)
- [ ] Breaking changes clearly documented in changeset

## Code Quality

- [ ] **Code reviewed using LLM** (Claude, ChatGPT, etc.) for quality and best practices
- [ ] Code follows project conventions and patterns
- [ ] No hardcoded values or magic numbers introduced
- [ ] Error handling implemented appropriately
- [ ] Security considerations reviewed (no exposed secrets, proper input validation)

## Dependencies

- [ ] New dependencies added via `uv add` (if applicable)
- [ ] Dependencies are necessary and well-justified
- [ ] No unused dependencies introduced
- [ ] `uv.lock` updated automatically

## Deployment

- [ ] Changes are backwards compatible (or migration path documented)
- [ ] Environment variables documented (if new ones added)
- [ ] Docker configuration updated (if needed)

## Checklist Before Merge

- [ ] All CI checks passing
- [ ] At least one code review completed
- [ ] No merge conflicts
- [ ] Branch is up to date with target branch

## Additional Notes

Any additional information that reviewers should know:

---

**Note**: This PR follows the biocentral-server development workflow. Ensure UV package manager commands are used (`uv run`, `uv add`, etc.) and changesets are created via CLI.