# Agent Configuration — Honest Limitations

This document explains what `.agents/AGENTS.md` actually enforces,
what it cannot enforce, and what you need to do separately.

---

## What AGENTS.md actually does in Antigravity

The `.agents/AGENTS.md` file is read by the Antigravity agent and treated as
behavioral instructions. The agent will follow these rules when answering
questions — it will redirect code requests, not fulfill them, for the listed files.

**This is real enforcement, not a suggestion.** The agent reads the rules file
at the start of each session and incorporates them as hard constraints. You are
reading this document right now because of exactly this mechanism working as designed.

---

## What AGENTS.md cannot do — be clear with students about this

### 1. It does not technically block anything

The rules are instructions to an AI, not code that prevents actions. They are as
strong as the AI's ability to recognize and resist bypass attempts. A student who
phrases a request cleverly — e.g., asking for a "math explanation" of the attention
mechanism that coincidentally includes all the code — may extract partial guidance
that wasn't intended.

**What this means in practice:** The rules catch direct requests reliably. They are
less reliable against sophisticated indirect requests. For a beginner course this is
almost certainly fine — beginners aren't trying to jailbreak their IDE, they're just
stuck and asking for help. The rule handles the common case.

### 2. It does not disable inline autocomplete

**This is the bigger risk you identified, and you are correct.**

Antigravity does not have a documented config mechanism to disable inline code
suggestions (ghost-text / tab-complete) on a per-file or per-folder basis. There is
no equivalent of VS Code's `.copilotignore` file in this IDE.

Your options, in order of reliability:

| Option | What it does | Limitation |
|---|---|---|
| **Disable AI suggestions globally** | Turns off all inline completions in Antigravity settings | Students lose completions everywhere, including in helper scripts |
| **Educate students explicitly** | Tell them autocomplete is off-limits for phase files | Depends on student discipline; "it didn't feel like cheating" is a real excuse |
| **Have students write phase files externally** | They write their phase code in a plain text editor or separate notebook, paste only to test | Inconvenient but removes the autocomplete risk entirely |
| **Accept the limitation** | Acknowledge that ghost-text is a gray area and focus the honour requirement on the chat agent | Most practical for a real course |

If Antigravity adds per-file autocomplete scoping in a future release, the config
would go here in `.agents/` — but that feature does not currently exist.

### 3. The chat agent and the code editor are separate systems

The rules in AGENTS.md govern the chat agent's responses. If Antigravity's code
editor has its own AI-powered autocomplete engine, that engine may not read
AGENTS.md at all. They are often separate systems. The practical upshot: a student
could follow the chat agent rules perfectly and still receive ghost-text suggestions
while typing in their file. There is currently no single config file that governs both.

---

## What actually works to prevent copy-paste

In order of effectiveness:

1. **The NOTES.md pre-session questions** — students who filled in detailed answers
   in their own words before the session are much less likely to have copy-pasted,
   because the questions target conceptual understanding that copy-pasted code doesn't give.

2. **The sabotage tasks** — a student who copy-pasted their attention implementation
   will almost certainly fail the "remove the causal mask and describe exactly what changes"
   task, because they don't understand the mechanism well enough to describe the failure mode
   precisely. This is your real verification, not the code check.

3. **The `python gpt.py` causality test** — this test does catch broken mask
   implementations (including "correct-looking" code that doesn't actually mask).
   Passing it with copy-pasted code is possible; passing it AND articulating why
   it works is much harder.

4. **The AGENTS.md rule** — handles direct "write this for me" requests reliably.

---

## Prompt to give another instructor configuring this in Antigravity

> I'm running a course where students build a GPT-2-style transformer from scratch.
> I need the Antigravity agent to answer conceptual/debugging questions but never write
> implementation code for specific student-authored files (embeddings.py, attention.py,
> block.py, data.py, and code cells in phase notebooks 01-08). A set of instructor-provided
> scaffold files (model_config.py, gpt.py, train.py, generate.py) have no restriction.
>
> The rules file is at `.agents/AGENTS.md` in the workspace root. Antigravity reads this
> automatically — no additional configuration is needed for the chat agent.
>
> **Known limitation:** Per-file inline autocomplete scoping is not currently configurable
> in Antigravity. If you need to prevent ghost-text suggestions in specific files, the only
> current option is to disable AI editor suggestions globally in Antigravity's settings,
> or to instruct students explicitly that tab-complete in their phase files is off-limits.
>
> The `.agents/AGENTS.md` file handles chat agent requests. It does not control the
> inline suggestion engine.
