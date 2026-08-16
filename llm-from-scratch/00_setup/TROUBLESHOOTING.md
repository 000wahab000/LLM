# Troubleshooting — 00 Setup

These are the most common things that go wrong during setup, in order of how often
beginners hit them. Check here before asking for help — the answer is almost certainly
one of these six.

---

## 1. "My GPU check says False / I'm running on CPU"

**What it looks like:**
```
GPU available: False
```
Or your training loop is extremely slow (10–50x slower than expected).

**What caused it:**
You forgot to change the runtime type before running code. Colab starts on CPU by default
every single session. This is the most common mistake — even experienced users do it.

**How to fix it:**
1. In the menu bar at the top, click **Runtime → Change runtime type**
2. Select **T4 GPU** under Hardware accelerator
3. Click **Save**
4. When prompted, click **OK** — this will restart the runtime and clear any code you
   already ran
5. Re-run your setup cells (clone, cd, git config, PAT remote) from the top

> **Prevention:** Make changing the runtime to GPU the very first thing you do every
> time you open a Colab session, before you run any code at all.

---

## 2. "Git push says 'Permission denied' or 'Authentication failed'"

**What it looks like:**
```
remote: Support for password authentication was removed.
fatal: Authentication failed for 'https://github.com/...'
```
Or:
```
ERROR: Permission to INSTRUCTOR-USERNAME/llm-from-scratch.git denied to YOUR-USERNAME.
```

**What caused it:**
Either (a) you're trying to use your GitHub password instead of a Personal Access Token,
or (b) you cloned the instructor's repo URL instead of your fork's URL.

**How to fix it:**

**If you cloned the right URL but are using a password:**
GitHub removed password auth in 2021. You must use a Personal Access Token. See
SETUP.md Part B Step 5 to generate one, then run:
```bash
!git remote set-url origin https://YOUR-USERNAME:YOUR-PAT@github.com/YOUR-USERNAME/llm-from-scratch.git
```

**If the error says "Permission to INSTRUCTOR-USERNAME/...":**
You cloned the instructor's repo, not your fork. You don't have push permission to
someone else's repo. Fix this by:
1. Check what URL git thinks the origin is:
   ```bash
   !git remote -v
   ```
2. If it shows the instructor's URL, update it to your fork:
   ```bash
   !git remote set-url origin https://YOUR-USERNAME:YOUR-PAT@github.com/YOUR-USERNAME/llm-from-scratch.git
   ```
3. Try pushing again.

---

## 3. "I cloned the original repo URL, not my fork"

**What it looks like:**
Your `!git remote -v` shows the instructor's username in the URL, not yours. Or you
tried to push and got a permission error (see above).

**What caused it:**
When you cloned, you copied the URL from the original course repo page instead of
navigating to your own fork first. Your fork has `YOUR-USERNAME` in the URL;
the original has the instructor's username.

**How to fix it:**

If you just want to fix the remote without re-cloning:
```bash
!git remote set-url origin https://YOUR-USERNAME:YOUR-PAT@github.com/YOUR-USERNAME/llm-from-scratch.git
```

If you're not sure which URL is correct, go to github.com, sign in, click your
profile picture, click **Your repositories** — your fork should be listed there.
The URL in the browser bar is the one to use.

---

## 4. "I closed Colab and lost my work — it's not on GitHub"

**What it looks like:**
You come back to Colab the next day, open your notebook, and your code changes from
the previous session are gone. Colab's filesystem is empty again.

**What caused it:**
Colab's filesystem is completely temporary. When a session ends (disconnected, closed,
or idle for too long), everything in it is deleted — including any files you changed
but didn't push to GitHub. This is expected behavior, not a bug.

**How to fix it:**
Unfortunately, if you didn't commit and push, the work is gone. There is no recovery.

**How to prevent it in future sessions:**
Commit and push every time you finish something — even something small. Make it a habit:
```bash
!git add .
!git commit -m "describe what you just did"
!git push
```
Do this:
- After finishing a NOTES.md answer block
- After writing a function that works
- After a successful sabotage-task experiment
- Before walking away from Colab for any reason

You can push dozens of commits in a session. There is no "too many commits."

For model weights specifically: the training loop in `train.py` saves checkpoints to
the `checkpoints/` folder (which is gitignored by default since weights are large).
In phase 06, you'll either push weights to HF Hub or mount Google Drive. That's
covered in phase 06's NOTES.md.

---

## 5. "Hugging Face push fails — 403 or 'not authorized'"

**What it looks like:**
```
huggingface_hub.utils._errors.HfHubHTTPError: 403 Client Error: Forbidden
```
Or:
```
You don't have the rights to push to this repository.
```

**What caused it:**
Almost always one of two things:
- Your HF token was created with **Read** access instead of **Write** access
- Your HF token expired or was revoked

**How to fix it:**
1. Go to **huggingface.co → Settings → Access Tokens**
2. Check if your token shows **read** or **write** under its role
3. If it says **read**, you need to generate a new one with **write** access
   (you cannot change the permission of an existing token — delete it and make a new one)
4. In your Colab session, run:
   ```python
   from huggingface_hub import login
   login(token="hf_YOUR_NEW_TOKEN_HERE")
   ```

---

## 6. "Git commit fails — 'Please tell me who you are'"

**What it looks like:**
```
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
```

**What caused it:**
Git requires a name and email before it will let you commit. These reset every Colab
session because Colab's environment resets. You either forgot to run the config
commands this session, or ran them with a typo.

**How to fix it:**
Run these two commands (they take effect immediately):
```bash
!git config --global user.name "Your Name"
!git config --global user.email "your@email.com"
```

Then retry your commit:
```bash
!git commit -m "your message"
```

> **Tip:** Put these two lines plus the PAT remote command into a single cell at
> the top of every Colab notebook. Make running that cell the first thing you do
> after cloning, so you never forget.

---

## Still stuck?

If your issue isn't in this list, copy the exact error message you're seeing and
share it. Include:
1. The full error text (not a summary — the actual text)
2. What command or action triggered it
3. What you already tried from this document

A raw error message is almost always enough to diagnose the problem immediately.
