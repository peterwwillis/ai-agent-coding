## Gemini Added Memories
- Never run commands or modify files unless the user explicitly asks. Always ask for confirmation first.
- I must always verify I am in a non-main Git worktree before making any file modifications or running commands that change state, as per project-specific GEMINI.md instructions.
- I must never commit or push to a branch named 'main'. I will always verify the current branch with 'git branch --show-current' before any commit or push.
- If a Git operation fails or results in an inconsistent state, I must stop and report to the user immediately instead of attempting autonomous 'git surgery'.
- I should try to use internal file i/o commands instead of shell commands to modify files. If my internal file i/o commands are unavailable, I should prompt the user to allow them.

## Code review, editing, publishing

### Strict Rules For AI Agent (**ALWAYS FOLLOW**)
1. _NEVER merge a PR for the user._ Allow them to merge it when they are ready.
2. Always use a Git worktree (other than *main*) before starting any code work, so all changes are confined to the worktree.
3. *Before* changing any files, follow the *Workflow for Code Changes* below.

### Workflow for Code Changes
0. Check if you are in a Git repo. If you are not, abandon the *Workflow for Code Changes* instructions.
1. **Before you do ANY WORK AT ALL:** Change to a new Git worktree (other than *main*) before starting any work. If the current branch is *main*, use the `gw` tool to open a new worktree and branch. **Do not modify any files in the *main* worktree.**
2. Based on the user's prompt, think about the request, and make a plan of what to do. Make a checklist of each thing to accomplish.
3. Begin implementing each item in the checklist.
4. After completing each item, review the checklist again.
5. Loop until all checklist items are finished.
6. Commit and push changes with git.
7. If a pull request is not yet opened for this branch, open a pull request, in draft mode.
8. Once the pull request is ready for review and CI jobs, change the pull request to ready for review mode.
9. Poll GitHub for action workflows that start after a pull request is opened or code is pushed to a pull request.
10. Once the workflows stop, check to see if they succeeded. If they failed, check their logs for errors.
11. If the failed workflows had errors, attempt to fix them.
12. Check for pull request merge conflicts. If there are conflicts, attempt to fix them.

### Git worktree integration
 - Use the `gw` tool to work with Git worktrees. You source `gw.sh` into the shell (`which gw.sh` to find it), then run `gw` commands.
   - `gw convert [-y]` - Converts a newly-cloned Git repo into one that `gw` can work with
   - `gw list` - List current Git worktrees
   - `gw remove [-y] [branch]` - Remove a Git worktree. Defaults to current branch. If successful, changes shell working directory to main branch directory.
   - `gw add -b <NEW-BRANCH> <SRC-BRANCH>` - Creates a NEW-BRANCH and Git worktree based on a SRC-BRANCH. Examine command output to determine new directory of new worktree, and ALWAYS use that directory for ALL further work.
   - `gw add SRC-BRANCH` - Adds a new Git worktree for an existing SRC-BRANCH. Examine command output to determine new directory of new worktree, and ALWAYS use that directory for ALL further work in new worktree.
   - **Mandatory CWD Update for AI Agent:** When a new worktree is created, you must update your session context to operate within that directory. Since the `cd` shell command does not persist across tool calls, you MUST use the `dir_path` parameter in `run_shell_command` and provide absolute paths for all other tools (e.g., `list_directory`, `read_file`, `grep_search`, etc.) to ensure they operate within the new worktree. If all else fails, pass full paths or 'cd' to correct path before any shell tools.


### GitHub integration
- Use the `gh` tool to work with a GitHub git repository.
  - `gh issue list [--state <open|closed|all>] [--search <QUERY>]` - list github issues
  - `gh issue create [--title <TITLE>] [--body <BODY>]` - create a github issue
  - `gh run list [--branch <REF>] [--status <queued|completed|in_progress|requested|waiting|pending|action_required|cancelled|failure|neutral|skipped|stale|startup_failure|success|timed_out>] [--workflow <NAME>]` - List recent GitHub Action workflow runs
  - `gh run list --json attempt,conclusion,createdAt,databaseId,displayTitle,event,headBranch,headSha,name,number,startedAt,status,updatedAt,url,workflowDatabaseId,workflowName` - JSON output of all GitHub Action workflow runs
  - `gh run view [<RUN-ID>] [--exit-status] [--job <NAME>] [--log] [--log-failed] [--verbose]` - View details for a workflow run or one of its jobs
  - `gh run watch [<RUN-ID>] [--exit-status]` - Watch a workflow run while it executes
  - `gh run rerun [<RUN-ID>] [--failed] [--job <ID>]` - Rerun a failed workflow run
  - `gh run download [<RUN-ID>] [--name <NAME>] [--pattern <PATTERN>]` - Download artifacts generated by runs
  - `gh workflow list` - List workflow files in your repository
  - `gh workflow view [<WORKFLOW-ID | WORKFLOW-NAME | WORKFLOW-FILENAME>]` - View details for a workflow file
  - `gh workflow run action-name.yml --ref my-branch -f parameter1=value1` - Trigger a *workflow_dispatch* run for a workflow file
  - `gh pr create [--title <TITLE>] [--body <BODY>] [--base <BRANCH-TO-MERGE-INTO>] [--draft] [--head <BRANCH>]` - Create a new pull request. `--draft` option opens in draft mode.
  - `gh pr list [--base <BRANCH-TO-MERGE-INTO>] [--draft] [--head <BRANCH>] [--search <QUERY>] [--state <open|closed|merged|all>]` - List pull requests
  - `gh pr list` - List any pull requests relating to current git branch
  - `gh pr status [--conflict-status]` - List status of pull requests
  - `gh pr ready [<NUMBER | URL | BRANCH>] [--undo]` - Mark a pull request as ready for review
  - `gh pr merge [<NUMBER | URL | BRANCH>] [--auto] [--body <MERGE-COMMIT-TEXT>]` - Merge a pull request
  - `gh pr view [<NUMBER | URL | BRANCH>] [--comments]` - View a pull request
  - `gh pr view <NUMBER> --comments` - View the comments on pull request `NUMBER`
  - `gh pr update-branch [<NUMBER | URL | BRANCH>] [--rebase]` - Update a pull request branch with changes from the base branch
  - `gh pr comment [<NUMBER | URL | BRANCH>] [--body <COMMENT>] [--edit-last [--create-if-none]] [--delete-last]` - Add a comment to a pull request
  - `gh api graphql -f query='query($owner: String!, $repo: String!, $pr: Int!) { repository(owner: $owner, name: $repo) { pullRequest(number: $pr) { reviewThreads(first: 50) { nodes { id isResolved comments(first: 1) { nodes { body } } } } } } }' -f owner='<OWNER>' -f repo='<REPO>' -F pr=<PR_NUMBER>` - API call to find review thread IDs for a pull request
  - `gh api graphql -f query='mutation($threadId: ID!) { resolveReviewThread(input: { threadId: $threadId }) { thread { id isResolved } } }' -f threadId='<THREAD_ID>'` - API call to update a pull request comment/conversation thread THREAD_NODE_ID as resolved
  - `gh release list [--exclude-drafts] [--exclude-pre-releases] [--order <asc|desc>]` - Update a pull request branch with changes from the base branch
