# Anonymous code submission for BRACIS 2026

Developer-facing checklist for publishing the `bracis` branch through
`anonymous.4open.science` for double-blind review. Reviewers can ignore this
file — see `README.md` instead.

## How `anonymous.4open.science` works

`anonymous.4open.science` is a free **proxy** that mirrors a *public* GitHub
repository while hiding the owner/organization, scrubbing the commit history of
authoring metadata, and replacing user-supplied identifying terms with `XXXX`.

- It is a **live proxy**, not a snapshot — when you push new commits to the
  source repo, the anonymous mirror updates after a re-fetch.
- The source repository **must be public on github.com**.
- The reviewer URL has the form
  `https://anonymous.4open.science/r/<random-uuid>/`. Reviewers see files
  through the proxy; they cannot clone via `git`.
- File-system size limits are not advertised but the service throttles
  downloads (a few hundred files / 15 min).
- **The anonymize form lets you pick a specific branch.** The form fields
  in `public/partials/anonymize.htm` are:
  - `sourceUrl` — the GitHub repo URL
  - `branch` — a select dropdown populated from the repo's branches
  - `commit` — a text input pinning the anonymized mirror to a specific
    commit SHA (≥ 6 hex chars, required)
  - `repoId` — the user-chosen anonymous slug
  - `conference` — optional metadata
  This means the source repo can have other branches too — only the branch
  you select is exposed through the proxy. **However**, the source repo on
  GitHub is still public, and a curious reviewer who finds the source repo
  can browse the other branches there. So either (a) push `bracis` to a
  dedicated public repo with no other branches, or (b) accept that the
  source repo's existence is at most weak side-channel evidence — the proxy
  view itself is single-branch.

## Preparation steps (already done in this branch)

The `bracis` branch in this worktree has been prepared as follows:

1. **Trimmed to paper-essential code.** Internal scratch (paper drafts,
   audit logs, exploratory shell scripts, fusion-study docs, archived results)
   was removed.
2. **Identity scrubbed at source.** All occurrences of the author's name,
   GitHub handle, university domain, and absolute home-directory paths were
   replaced with placeholders (`<REPO_ROOT>`, `ANONYMIZED`, `/path/to/...`).
3. **Tests pass at baseline.** `pytest tests/` reports ~980 passed,
   ~28 skipped, 7 pre-existing failures — i.e., trimming and scrubbing did
   not regress the suite.
4. **Reviewer README written** (`README.md`) explaining what is in the repo
   and how to reproduce the paper's numbers.
5. **The development repo is private and stays private**;
   the anonymous mirror points at a separate public repo (see Step 1 below).

## Steps to publish anonymously

### 1. Push the `bracis` branch to a public GitHub repo

Two options:

**Option A — dedicated public repo (cleanest).** Create a new public repo on
github.com under any account, no other branches:

```bash
# From this worktree directory:
git remote add anon git@github.com:<account>/poi-mtl-bracis2026.git
git push anon bracis:main         # push as main of the new repo
```

**Option B — push the branch to an existing public repo.** Since the
anonymize form lets you pick a branch, you can push `bracis` to any public
repo and then select it in the form. Branches like
`worktree-check2hgi-mtl` will still be visible at the source repo URL, so
prefer Option A unless the source repo is already public.

Note the commit SHA of the pushed branch — you will need it in step 2.

### 2. Submit the URL to `anonymous.4open.science`

1. Visit `https://anonymous.4open.science/anonymize`.
2. Sign in with the GitHub account that owns the public repo.
3. Fill the form:
   - **GitHub URL** (`sourceUrl`) — the public repo URL.
   - **Branch** — select `bracis` (or `main`, depending on Option A vs B
     in step 1) from the dropdown.
   - **Commit** — paste the SHA of the anonymized commit (≥ 6 hex chars).
     Pinning the commit ensures the proxy keeps showing the curated tree
     even if you push more later.
   - **Anonymized repository ID** (`repoId`) — pick a slug; this becomes
     the URL path segment.
   - **Conference** (optional) — `BRACIS 2026`.
4. **Replacement-terms list** (provide author identifiers as pairs
   `term -> XXXX`):
   - First and last name (every casing variant in use)
   - GitHub handle
   - University name and email domain
   - Any project/codename referenced in old absolute paths
5. The service issues a permanent URL like
   `https://anonymous.4open.science/r/<repoId>/`. **Save this URL** — it is
   the value that goes into the paper.

### 3. Insert the link into the paper

In `articles/[BRACIS]_Beyond_Cross_Task/samplepaper.tex`, replace the
placeholder anonymous URL (search for `anonymous.4open.science`) with the
real URL from Step 2.

### 4. Final manual verification

Before submitting to JEMS3:

- Browse the proxy URL in an **incognito window** and confirm the repo
  renders without the owner name.
- Search the proxy for the author identifiers above (the search bar in the
  proxy navigates inside the repo). Anything found that is not part of an
  external citation must be re-scrubbed in the source repo.
- The bracis branch was committed under `anonymous <anon@example.com>`, so
  no real author appears in `git log`. If you re-commit later, repeat the
  override:
  ```bash
  git -c user.name=anonymous -c user.email=anon@example.com commit -m "..."
  ```

## Sources

- BRACIS 2026 CFP, §3 (Anonymous GitHub or Dropbox required for code/data).
- Anonymous GitHub FAQ — `https://anonymous.4open.science/faq`
- Anonymous GitHub source — `https://github.com/tdurieux/anonymous_github`
