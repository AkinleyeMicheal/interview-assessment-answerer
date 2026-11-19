# Quizbox Phone UI — Complete Control Reference

Every button, toggle, input and gesture in the phone interface, what it does,
and when to use it. The phone is the control surface; the PC only hosts the
capture region and the optional overlay pane.

---

## 1. Header (always visible)

| Control | What it does | When to use |
|---|---|---|
| **▦** (QR) | Opens the QR modal with this page's URL so you can scan it on another device | When you want to open Quizbox on a different phone/tablet |
| **🔒** (Rotate Gemini Key) | Immediately rotates to the next Gemini API key from the configured pool | When Gemini rate-limits you mid-session; no need to open Settings |
| **⚙** (Settings) | Opens the Settings modal (API keys, provider status) | First-time setup, key rotation, checking provider health |

### Settings modal
- **Groq API Key**, **Gemini API Key 1–5** (password fields) — paste keys, then **Save**; **Cancel** discards
- **Provider Status** panel — live validity of each configured provider

### QR modal
- **Close** — dismisses

---

## 2. Mode tabs (top bar)

| Tab | What it does | When to use |
|---|---|---|
| **Groq Vision** | Question → Groq vision answer flow (screenshot capture) | Quizzes on Groq (free, fast) |
| **Gemini Vision** | Same flow answered by Gemini | When Groq is slow/unavailable |
| **Interview** | Live & Mock interview mode (see §5) | Real or practice interviews |
| **Mirror** | Controls for the desktop overlay pane (see §6) | You want a pane floating on the PC screen |

The active tab persists across reloads (except Interview/Mirror, which are
session tabs). Switching vision tabs auto-saves the choice and broadcasts it
so an open Mirror pane follows along.

---

## 3. Vision tabs — Groq Vision / Gemini Vision

### Capture area
| Control | What it does | When to use |
|---|---|---|
| **Select Region** | Opens the full-screen region picker on the PC — drag a rectangle over the question | Once, before answering; re-do whenever the question layout moves |
| Live preview | Shows the selected region streaming from the PC | Visual confirmation that the right area is captured |

### Bottom action bar
| Control | What it does | When to use |
|---|---|---|
| **Quick Answer** | Grabs the region now and sends it to the active AI → answer appears in the panel | The normal "answer this question" button (Groq tab) |
| **Web Verified** | Same capture, but the answer is cross-checked against web search results | When you want a verified/second-opinion answer (Groq tab) |
| **Capture** | Same as Quick Answer (Gemini tab's button) | Answering questions on the Gemini tab |
| **⏱ cooldown** | Opens the Capture Cooldown popup: number input (1–120 s) + **Save** / **Cancel** | To throttle how often captures fire — avoids rate limits |

> Capture buttons show `Wait Ns` while cooling down and disable themselves;
> they also need an active region + WS connection.

### Answer panel
| Control | What it does | When to use |
|---|---|---|
| **Copy** (on code blocks) | Copies the answer's code to the phone clipboard | When you need the code elsewhere |

---

## 4. Screen status bar (informational, not controls)

- **● status dot + text** — server/capture state
- **Provider text** — active AI provider
- **Count** — questions answered this session

---

## 5. Interview tab

### 5.1 Subtabs
| Control | What it does |
|---|---|
| **Live** / **Mock** | Switch between real-interview assist and AI-led practice |
| *(conn indicator)* | `connected` / `reconnecting...` socket status |

### 5.2 Live subtab
| Control | What it does | When to use |
|---|---|---|
| **Start Live Interview** | Captures interviewer speech from your PC speakers, transcribes it, and posts a suggested answer for every question heard | When the real interview begins |
| **Stop** | Ends the live session | When the interview is over |

### 5.3 Mock subtab
| Control | What it does | When to use |
|---|---|---|
| **Mic source** select — *Desktop Mic / Phone Mic* | Where the mock hears your spoken answers | Phone Mic when you must speak quietly / PC mic is occupied |
| **Role** text field | Tells the AI interviewer which role to simulate | Before Start (e.g. `backend engineer`) |
| **Level** select — *Entry / Mid / Senior* | Difficulty of generated questions | Before Start |
| **Start Mock Interview** | AI asks a question aloud, listens, scores you /10, then follows up | Rehearsal runs |
| **Stop** | Ends the mock entirely | Ending practice |
| **Skip** | Abandons the current question and advances | When you're stuck and want the next one |
| **Done** *(while answering)* | Commits your spoken answer immediately — skips the 6-second silence detector | Right when you finish talking |

### 5.4 Screen & context row (both subtabs)

The row is a **3-column button grid** (`Groq | Select Region | Quick Capture`
/ `Solve | Next Step | Auto` / `Problem | Code | + Segment` / `Design`), so
every target stays thumb-sized on a phone and comfortable in the pane.
Start/Stop use 2 columns; mock actions a 2×2 grid.

| Control | What it does | When to use |
|---|---|---|
| **Groq ⇄ Gemini** | Swaps the answering model mid-session — full history carries over; flips itself automatically on provider failure | When Groq is slow/rate-limited (label = active model) |
| **Select Region** | Same region picker as the vision tab | When no region is set yet |
| **Quick Capture** | One screenshot of the region + preview overlay | To eyeball exactly what will be sent |
| **Solve** | Region frame (→ stacked segments → stashed Problem text) + history → spoken answer + code + next steps | You want the answer to the current coding problem |
| **Next Step** | Same inputs, but returns only the *next increment* of the solution | Building an answer step-by-step |
| **Auto: On/Off** | When On, a coding question detected in Live mode auto-triggers Solve in the background, and an active Design session re-reads the panel every ~15 s (see 5.7). Failures surface as a toast (`Auto assist: No screen frame…`) instead of failing silently — throttle/duplicate skips stay quiet | Hands-free solving; Off = manual Solve only |
| **Problem** | After you **physically copy** the problem text with Ctrl+C, tap to stash the clipboard into context | Copy-paste-disabled platforms; text too long to screenshot |
| **Code** | Same, for your starter/partial code | You copied your current code |
| **+ Segment** | Screenshots the region into a stack (up to 3 sent as multi-images) | Long problem that spans multiple screens — scroll, tap, repeat |
| **Design** | Starts a system-design session from the question panel (frame → stashed Problem text). While a session runs the button becomes **Next ▸** (region re-read) | The interviewer posts a "Design a …" question |

### 5.5 Context chips (below the row)
| Control | What it does |
|---|---|
| **Problem N ×** / **Code N ×** / **Segments N ×** | Tap text to preview the stash; tap **×** to clear it |
| **×** on the screenshot preview | Closes the floating preview overlay |

### 5.6 During a live/mock session
- **Status line** — state machine (`listening`, `evaluating`, …)
- **Audio chip** *(live only)* — `Audio: starting / ok / silent / no audio /
  no transcript` beside the status line: green = hearing the interviewer,
  amber/red = capture problem. The first transition into each failure also
  toasts once per session (e.g. `Live audio check: No PC audio — is the
  meeting playing on this PC?`), so a dead capture can't hide behind
  "Auto: On"
- **Rate chip** — amber `Rate limit — retry in Ns` beside the audio chip
  while the rate gate is backing off (all model calls go through it: instead
  of failing, the app waits its turn — exponential backoff, provider
  failover Groq⇄Gemini, Gemini key rotation). It clears itself when a call
  recovers
- **History** — full conversation; answers with code include **Copy**
- **Caption strip** — always visible while a session runs: live partial
  transcript while the interviewer speaks, and a `🎙 Listening for the
  interviewer…` placeholder (mirroring the audio chip's message) between
  utterances. It only hides when the session stops — you never have to
  *assume* what the app is hearing

### 5.7 Design interview mode (system-design questions)

Start it with the **Design** button (live session required, region set over
the question panel — or the Problem text stashed as a fallback). The app
extracts the question **and its instruction ladder** (clarify → high-level
design → deep dive → bottlenecks), then follows that ladder for the whole
multi-turn answer.

| Control / element | What it does |
|---|---|
| **Design bar** (appears while active) | Stage stepper `Clarify → Needs → Design → Deep dive → Bottlenecks → Done`, the extracted question, and a `Next: …` hint from the last turn |
| **Next ▸** (design bar *and* the Design button) | Re-reads the region and continues from the newest instruction/feedback on the panel — for text-only instructions the STT never hears |
| **Auto: On** *(while active)* | Also re-reads the panel automatically every ~15 s (non-overlay client only, server floors it at 12 s, rate gate serializes calls) |
| **📋 Copy image** (design bar + every diagram card) | Renders the current mermaid diagram to PNG on the clipboard — tap the whiteboard, Ctrl+V |
| **Code** on a diagram card | Copies the mermaid source instead (also the automatic fallback if mermaid fails to render) |
| **End** | Leaves the design session (state cleared; also clears automatically on Stop) |

How the session stays consistent for 20+ minutes:

- Every turn returns **state patches** — clarified facts, architecture
  decisions, the full mermaid diagram, current stage — merged server-side.
  Verbatim history is capped at 6 turns because everything older already
  lives in that state, so the prompt size never grows and early
  requirements are never forgotten.
- Answers append to the normal history, so questions *after* the design
  session still reference what was decided.
- Diagrams render inline as **diagram cards** (phone + pane) with
  Copy image / Code buttons.
- Spoken turns are detected by live STT (every committed utterance advances
  the session); region re-reads catch silent on-screen instructions.

---

## 6. Mirror tab — desktop overlay pane controls

The pane is a frameless, always-on-top window on the PC that mirrors this
app (same tabs, follows your active tab). It is **read-only by default**:
the mouse always passes through to whatever is underneath (never to the
pane), and the page itself ignores all pointer events — except while
**Scroll mode** is unlocked (see below). The pane is also **excluded from
screen capture** at the OS level (`WDA_EXCLUDEFROMCAPTURE`): it never
appears in the live feed, the phone view, or any recording — the feed
always shows what's *behind* the pane, so the mirror can never reflect
itself into an infinite-mirror loop.

### What the pane mirrors (everything you can see on the phone)

- **Live region feed** — the pane starts/stops its own frame loop whenever
  capture starts or stops anywhere (even after the pane is already open)
- **Vision answers** — answer panel, badges, explanation, sources, the
  answered count, and the cooldown countdown arrive over the same broadcast
- **Toasts** — every toast the phone shows appears on the pane too
- **Interview state** — Live/Mock subtab, Start/Stop buttons, status dot and
  line, history, transcripts, captions, mock questions/feedback
- **Context & model** — Problem/Code/Segment chips, active provider
  (Groq ⇄ Gemini), Auto: On/Off
- **Tabs & pane controls** — tab switches, plus every Mirror-panel control
  (size/zoom/opacity feedback)

Settings and QR popups and input fields (role, level, mic, keys) stay
phone-only — they are controls, not display content.

| Control | What it does | When to use |
|---|---|---|
| **Open Pane** | Creates/shows the overlay window on the PC (parks at the right edge on first open, fully on-screen; default size **768×800**, clamped to fit the screen — e.g. 768×768 on a 768-high display) | When you want the mirror visible on the PC |
| **Close Pane** | Destroys the window (position/size/zoom/opacity are remembered) | Done with the mirror |
| **Hide / Show** | Instantly hides the pane without closing it (panic button) | The moment someone approaches — or use the PC hotkey below |
| **Drag pad** | Touch-drag inside the pad → the pane follows (deltas flush every 50 ms) | Repositioning the pane without touching the PC |
| **Width −/+** | Resize width by 40 px per tap (clamped to screen) | Getting wider for reading / narrower for stealth |
| **Height −/+** | Resize height by 40 px per tap | Same, vertical axis |
| **Zoom −/+** | Scales the pane's *content* 0.5×–2.0× in 0.1 steps — starts at **0.75×** (compact desktop text/buttons; phone UI unaffected); window size unchanged | Text too small/large without changing the footprint |
| **Opacity −/+** | Window transparency 20 %–100 % in 10 % steps | Blending the pane into the page behind it |
| **Scroll mode** — 🔒 **Locked** / 🔓 **Unlocked** | Unlocks native scrolling **on the pane itself**: put the PC mouse over the answer/conversation panel and wheel-scroll, drag the (always-visible) scrollbar, or click its arrows. Only those two panels become interactive — every other pane area still ignores clicks, and the pane never takes focus. Auto re-locks on **Hide**, **Close pane**, phone disconnect, or tapping Locked again | Reading long solutions directly on the PC with the mouse |
| **Tiny / Compact / Standard** | Instant presets: 240×150 / 420×560 / 768×800 | Rapid size jumps (Tiny = near-invisible corner readout) |
| **Center** | Moves the pane to the center of the screen | Quick reposition after a drag |

**Status line** — `Pane closed`, `Pane hidden`, or `Pane open — 768×800 @
(1112,40) · 100% · read-only` (or `· scroll unlocked` while Scroll mode is
on), turning green while open. Updates live from server state.

### PC-side pane controls
| Control | What it does |
|---|---|
| **Ctrl+Shift+`** (global hotkey) | Show/hide the pane instantly, even from inside a fullscreen browser — works while the phone is in your pocket |

**Behavior notes**
- The pane never steals focus and never appears in the taskbar or Alt+Tab.
  While Scroll mode is **Locked** (the default) mouse and touch pass
  through to the window below; while **Unlocked**, the pane area intercepts
  the mouse (only its two scroll panels respond) until it re-locks
- While unlocked, an amber **SCROLL ON** badge appears on the pane so a
  forgotten unlock is always visible — and Hide, Close, and phone
  disconnect re-lock it automatically
- The pane is invisible to **every** capture API on the PC: the live feed,
  the phone's view, screenshots, and even a full-screen recording show only
  what's behind it (this is what prevents the infinite-mirror feedback loop)
- Tab switches on the phone are mirrored live; the pane reconnects by itself
- Opening the pane costs ~200 MB RAM; Close frees it
- In a *tab-only* screen share the pane is already invisible; a full-screen
  recording is safe too — the OS excludes the pane's pixels from capture
- Caveat: a normal **browser window** showing this app *inside* the captured
  region can still pick up its own preview (Windows only lets our own windows
  opt out) — keep such windows outside the region
- Long answers: unlock **Scroll mode** and scroll the pane natively with the
  mouse — or scroll the answer/conversation on the phone while the pane is
  locked and the pane mirrors the position live (an unlocked pane mirrors
  its scrolls back to the phone too). New answers auto-reveal (solution at
  the top, conversation at the newest entry) on both screens

---

## 7. Gesture & keyboard shortcuts

| Gesture | Where | Effect |
|---|---|---|
| **Drag** | Mirror tab drag pad | Move the overlay pane |
| **Ctrl+C** (physical keyboard on PC) | Anywhere | Copies problem/code → tap **Problem**/**Code** to stash |
| **Ctrl+Shift+`** | PC global | Toggle pane visibility |
| **Escape** | Region picker (PC) | Cancel selection |

---

## 8. Quick recipes

1. **Coding interview, text problem:** copy text (Ctrl+C) → tap **Problem**
   → **Solve** (works with no screenshot at all).
2. **Long problem on screen:** Select Region → scroll + **+ Segment** ×2–3 →
   **Solve** (multi-image).
3. **Hands-free during live interview:** Start Live + **Auto: On** — coding
   questions auto-solve; watch the phone or the pane.
4. **Stealth mirror:** Open Pane → **Tiny** preset → **Opacity −** a few
   times (mouse already passes through); hide with **Ctrl+Shift+`**.
5. **Groq rate-limited:** tap **Groq ⇄ Gemini** — conversation and stashes
   carry over automatically.
