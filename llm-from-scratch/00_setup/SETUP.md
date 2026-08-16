# 00 — Setup

> This document explains the structure and constraints of the course environment.
> It is not a walkthrough. Figure out the mechanics using GitHub's own documentation
> (linked below) and the mental model here. `verify_setup.ipynb` will tell you
> whether your setup is correct. `TROUBLESHOOTING.md` is for when something goes wrong.

---

## Why a fork, not a branch or clone of the original

This course uses a **fork-per-student** model. A fork is your own copy of the
repository, owned by your GitHub account, fully independent of the original and of
every other student's copy. You commit your phase work there. The instructor's repo
stays separate and can receive updates (to scaffold files, NOTES.md improvements,
etc.) without those changes automatically appearing in yours.

This model has one important implication you need to understand before writing any
code: **the relationship between your fork and the upstream (instructor) repository
is not automatic**. Changes in one direction don't flow to the other unless you
explicitly make them. This is a feature, not a limitation — it means your work is
safe from instructor changes, and the instructor can't accidentally overwrite what
you've done.

→ [GitHub documentation: About forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)

---

## What must be true before you start phase 01

These are requirements, not steps. How you satisfy them is up to you.

**1. GPU connected in Colab**
Every phase after 00 assumes a GPU runtime. If you run code on CPU by accident
you'll know because training will be 10–50x slower than expected. The Colab
runtime type must be set to GPU before any phase notebook runs. This resets every
session — it is a per-session configuration, not a one-time setup.

**2. Your fork is cloned, not the original repository**
The URL of what you've cloned matters. If you cloned the instructor's URL, pushes
will fail (you don't have write access to someone else's repository). The clone URL
must contain your GitHub username, not the instructor's.

Colab's filesystem is temporary — it resets when a session ends. You will re-clone
your fork at the start of every session. This is expected; build it into your
workflow.

**3. Git identity is configured**
Git requires a name and email before it will accept a commit. These settings also
reset per session. A commit failing with an "Author identity unknown" error is not
a bug — it means this configuration step was skipped.

**4. You can actually push to your fork**
GitHub removed password authentication for HTTPS pushes in 2021. You need a
**Personal Access Token** with `repo` scope to authenticate from Colab. How you
store and use that token is your decision — the constraint is that `git push` must
succeed before phase 01 begins.

→ [GitHub documentation: Creating a personal access token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)

**5. Hugging Face token with write scope exists**
You won't need it until phase 08. Generate it now, write it down somewhere you can
find it, and don't lose it. The constraint for phase 08 is that the token has
**write** access (not read-only) — this is set at token creation time and cannot be
changed after the fact.

→ [Hugging Face documentation: User access tokens](https://huggingface.co/docs/hub/security-tokens)

---

## Scaffold files: where they are and the update problem

The four scaffold files (`model_config.py`, `gpt.py`, `train.py`, `generate.py`)
and `SCAFFOLD.md` already exist in the repository you forked. They are part of the
repo, not something you download separately.

**The constraint that matters:**

The instructor may update these files during the course — bug fixes, clarifications,
improvements to the verification checks. When that happens, you need a way to get
the updated versions into your fork **without overwriting your own committed work**.

This is a real version-control problem. Think about how Git's remote model works,
what the relationship between your fork and the upstream repository is, and what
operations exist for selectively incorporating changes from one Git history into
another. You have the tools to solve it — figure out the strategy before you need
it, not in the middle of a session.

---

## Colab session model — one constraint worth being explicit about

When a Colab session ends (timeout, disconnect, or you close the tab), the
filesystem is wiped. **Any work you haven't committed and pushed to GitHub is
gone.** There is no recovery.

This is not a setup problem to fix. It is a fundamental property of Colab's free
tier that affects how you work: commit frequently, push before closing anything,
and don't treat Colab's filesystem as permanent storage.

For model weights specifically: checkpoints go to either Google Drive or Hugging
Face Hub (covered in phases 06 and 08). They do not go in the Git repository — the
`.gitignore` already excludes `checkpoints/`.

---

## Checking your setup

Run `00_setup/verify_setup.ipynb` top-to-bottom in Colab. Each cell prints a clear
PASS or FAIL. If everything passes, your environment is ready for phase 01.

If something fails, `TROUBLESHOOTING.md` covers the six most common failure modes
with exact error text and fix steps.
