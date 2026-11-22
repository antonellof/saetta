# Themes and settings

## Themes

At the first start, before the first prompt, the agent asks for a theme. `/theme` changes it later, with a live preview as you move through the list. The choice is saved as `theme` in settings.json.

| theme | look |
|---|---|
| `saetta` | the default: macOS's system colors, light or dark to match the terminal's background |
| `system` | the terminal's own palette |
| `dark`, `light` | pi's built-in themes |

Colors use 24-bit color when the terminal supports it; `PI_TRUE_COLOR=1` forces it and `PI_TRUE_COLOR=0` forces 256 colors. `PI_HYPERLINKS=1` or `0` turns clickable links on or off when the terminal is not detected correctly.

## /settings

`/settings` opens pi's settings menu, filtered as you type. Changes are saved to settings.json in the agent folder. Among its entries: Auto-compact, Skill commands, Hide thinking, Steering mode and Follow-up mode, Double-escape action, Tree filter mode, Default project trust, and the theme.

## settings.json

Saetta Code reads pi's `settings.json` from the agent folder (`~/.pi/agent`, or `~/.saetta/pi` when started with `saetta start code`) and, in a trusted project, `.pi/settings.json` for project skills. These are the keys it uses:

| key | default | effect |
|---|---|---|
| `defaultProvider`, `defaultModel` | | the model a session starts with; `/model` updates them |
| `defaultThinkingLevel` | | the thinking level a session starts with |
| `enabledModels` | | the models Ctrl+P cycles through (`/scoped-models`) |
| `theme` | asked at first start | `saetta`, `system`, `dark` or `light` |
| `hideThinkingBlock` | `false` | hide thinking blocks (Ctrl+T toggles) |
| `compaction` | `enabled` true, `reserveTokens` 16384, `keepRecentTokens` 20000 | automatic compaction ([Sessions](sessions.md#compaction)) |
| `retry` | `enabled` true, `maxRetries` 3, `baseDelayMs` 2000, `maxAgentDelayMs` 60000 | retries of failed model requests |
| `skills` | | extra skill paths and patterns ([Context and skills](context-and-skills.md#skills)) |
| `enableSkillCommands` | `true` | register `/skill:name` commands |
| `defaultProjectTrust` | ask | `always` or `never` to answer the trust prompt without asking |
| `steeringMode`, `followUpMode` | `one-at-a-time` | deliver queued messages one at a time or `all` together |
| `doubleEscapeAction` | `tree` | Escape twice in an empty editor: `tree`, `fork` or `none` |
| `treeFilterMode` | `default` | the filter `/tree` opens with |
| `externalEditor` | `$VISUAL`, `$EDITOR`, nano | the editor Ctrl+G opens |
| `fullscreenCopyOnSelect` | `true` | copy text as soon as it is selected |
| `collapseChangelog` | `false` | show a condensed changelog after an update |

An example:

```json
{
  "defaultProvider": "saetta",
  "defaultModel": "qwen3.8-flash-next",
  "defaultThinkingLevel": "medium",
  "theme": "saetta",
  "compaction": { "enabled": true },
  "skills": ["~/skills"]
}
```

`/settings` also lists pi's other entries (images, transport, share destination, cache warming and more). Keys not in the table above are pi's: Saetta Code keeps them in the file, so a settings.json shared with pi loses nothing.

## Keybindings

The keys are pi's defaults on macOS; `/hotkeys` lists them and [Interactive mode](interactive-mode.md#keys) has the table.
