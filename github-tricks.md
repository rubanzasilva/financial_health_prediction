# GitHub & Git Tricks

## .gitignore Best Practices

### Creating a .gitignore file
```bash
touch .gitignore
```

### Common patterns
```
# Ignore all CSV files
*.csv

# Ignore CSVs in specific directory only
data/*.csv

# Ignore CSVs in directory and subdirectories
data/**/*.csv
```

### If files are already tracked
`.gitignore` only affects untracked files. If files were already committed:

```bash
# Untrack files but keep them locally
git rm --cached *.csv

# Commit the changes
git add .gitignore
git commit -m "Remove CSV files from tracking"
git push
```

**Key point**: `--cached` removes files from Git tracking but keeps them on your hard drive.

## Branch Management

### Deleting branches after PR merge

**Best practice**: Delete from GitHub after merging the PR
- Safer (only shows after successful merge)
- One-click deletion button
- Recorded in PR history
- Clear audit trail

**Workflow**:
```bash
# 1. Push your branch
git push -u origin branch-name

# 2. Create PR on GitHub
# 3. Merge the PR
# 4. Click "Delete branch" on GitHub

# 5. Clean up locally
git checkout main
git pull
git branch -d branch-name
```

**Why delete local branch separately?**
- Git treats local and remote branches as separate entities
- Remote deletion doesn't auto-delete local branches (for safety)
- You might have unpushed local commits
- Git prefers explicit control over local work

**No commit/push needed after branch deletion** - it's just removing a pointer, not changing files.

## Push with Upstream Tracking

### Using `-u` flag (set upstream)
```bash
git push -u origin branch-name
```
- Pushes to remote
- **Sets up tracking** between local and remote branch
- Future benefit: Can use `git push` / `git pull` without specifying remote/branch

### Without `-u` flag
```bash
git push origin branch-name
```
- Pushes to remote
- **No tracking setup**
- Must specify remote/branch every time: `git push origin branch-name`

**When to use `-u`**: First time pushing a new branch. After that, simple `git push` works.

## Branch Tracking

**Tracking** = relationship between local branch and remote branch

### Benefits of tracking:
1. Git can compare branches (ahead/behind status)
2. Simplified push/pull (no need to specify remote/branch)
3. `git status` shows helpful info like "Your branch is ahead by 2 commits"

### With tracking:
```bash
git push -u origin branch-name
git status
# Output: "Your branch is up to date with 'origin/branch-name'"

git push    # Works!
git pull    # Works!
```

### Without tracking:
```bash
git push origin branch-name   # No -u
git status
# Output: Generic status, no remote info

git push    # Error: "no upstream branch"
```

## Useful Commands

### Check tracking relationships
```bash
git branch -vv
```

### Prune deleted remote branches
```bash
git fetch --prune
# or
git pull --prune
```

### Safe branch deletion (only if merged)
```bash
git branch -d branch-name
```

### Force branch deletion (even if not merged)
```bash
git branch -D branch-name
```
