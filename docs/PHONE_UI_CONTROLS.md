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
| Control | What it does | When to use |
|---|---|---|
| **Groq ⇄ Gemini** | Swaps the answering model mid-session — full history carries over; flips itself automatically on provider failure | When Groq is slow/rate-limited (label = active model) |
| **Select Region** | Same region picker as the vision tab | When no region is set yet |
| **Quick Capture** | One screenshot of the region + preview overlay | To eyeball exactly what will be sent |
| **Solve** | Region frame (→ stacked segments → stashed Problem text) + history → spoken answer + code + next steps | You want the answer to the current coding problem |
| **Next Step** | Same inputs, but returns only the *next increment* of the solution | Building an answer step-by-step |
| **Auto: On/Off** | When On, a coding question detected in Live mode auto-triggers Solve in the background | Hands-free solving; Off = manual Solve only |
| **Problem** | After you **physically copy** the problem text with Ctrl+C, tap to stash the clipboard into context | Copy-paste-disabled platforms; text too long to screenshot |
| **Code** | Same, for your starter/partial code | You copied your current code |
| **+ Segment** | Screenshots the region into a stack (up to 3 sent as multi-images) | Long problem that spans multiple screens — scroll, tap, repeat |

### 5.5 Context chips (below the row)
| Control | What it does |
|---|---|
| **Problem N ×** / **Code N ×** / **Segments N ×** | Tap text to preview the stash; tap **×** to clear it |
| **×** on the screenshot preview | Closes the floating preview overlay |

### 5.6 During a live/mock session
- **Status line** — state machine (`listening`, `evaluating`, …)
- **History** — full conversation; answers with code include **Copy**
- **Caption strip** — live partial transcript while the interviewer speaks

---

## 6. Mirror tab — desktop overlay pane controls

The pane is a frameless, always-on-top window on the PC that mirrors this
app (same tabs, follows your active tab). Default mode is **click-through**
(mouse passes straight through to the browser underneath).

| Control | What it does | When to use |
|---|---|---|
| **Open Pane** | Creates/shows the overlay window on the PC (parks at the right edge on first open) | When you want the mirror visible on the PC |
| **Close Pane** | Destroys the window (position/size/zoom/opacity are remembered) | Done with the mirror |
| **Hide / Show** | Instantly hides the pane without closing it (panic button) | The moment someone approaches — or use the PC hotkey below |
| **Drag pad** | Touch-drag inside the pad → the pane follows (deltas flush every 50 ms) | Repositioning the pane without touching the PC |
| **Width −/+** | Resize width by 40 px per tap (clamped to screen) | Getting wider for reading / narrower for stealth |
| **Height −/+** | Resize height by 40 px per tap | Same, vertical axis |
| **Zoom −/+** | Scales the pane's *content* 0.5×–2.0× in 0.1 steps (window size unchanged) | Text too small/large without changing the footprint |
| **Opacity −/+** | Window transparency 20 %–100 % in 10 % steps | Blending the pane into the page behind it |
| **Tiny / Compact / Standard** | Instant presets: 240×150 / 420×560 / 640×800 | Rapid size jumps (Tiny = near-invisible corner readout) |
| **Center** | Moves the pane to the center of the screen | Quick reposition after a drag |
| **Click-through: On/Off** | On = mouse passes through (stealth); Off = you can tap the mirrored UI on the PC | Only turn Off when you intentionally want to interact with the pane |

**Status line** — `Pane closed` or `Pane open — 640×800 @ (1240,40) · 100% ·
click-through`, turning green while open. Updates live from server state.

### PC-side pane controls
| Control | What it does |
|---|---|
| **Ctrl+Shift+`** (global hotkey) | Show/hide the pane instantly, even from inside a fullscreen browser — works while the phone is in your pocket |

**Behavior notes**
- The pane never steals focus and never appears in the taskbar or Alt+Tab
- Tab switches on the phone are mirrored live; the pane reconnects by itself
- Opening the pane costs ~200 MB RAM; Close frees it
- In a *tab-only* screen share the pane is invisible; a full-screen recording
  would include its pixels (share the exam tab, not the whole screen)

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
   times → leave **Click-through: On**; hide with **Ctrl+Shift+`**.
5. **Groq rate-limited:** tap **Groq ⇄ Gemini** — conversation and stashes
   carry over automatically.
