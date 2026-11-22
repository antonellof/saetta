# Sessions

Every conversation is a session: a JSONL file of entries (messages, tool results, model changes, compactions) that form a tree. Saetta Code writes them in pi's format and folder, so pi and Saetta Code continue each other's sessions, compacted ones included.

Sessions live in the agent folder, one subfolder per working directory:

```text
~/.pi/agent/sessions/--Users-me-src-my-project--/<timestamp>_<id>.jsonl
```

(`~/.saetta/pi/sessions/...` when the agent was started with `saetta start code`; see [Saetta Code](saetta-code.md#the-configuration-folder).)

## Continue and resume

```sh
saetta code -c                            # the most recent session in this folder
saetta code --session path/to/file.jsonl  # a given session file
saetta code --no-session                  # do not save this one
```

In the agent, `/resume` opens a picker of earlier sessions, searchable by their messages and names, and `/new` starts a fresh one.

![/resume: the session picker](../web/assets/code/resume.png)

## The session tree

A session is a tree, not a list: going back to an earlier message and continuing from there adds a branch, and the old branch stays in the file.

- `/tree` shows the tree and moves to any entry. Its filter (default, no tools, user messages only, labeled only, all) starts from `treeFilterMode` in settings.json. Escape twice in an empty editor opens it too.
- `/fork` picks an earlier user message and starts a new session file from it, with that message in the editor.
- `/clone` copies the session at the current position into a new session file.
- `/name` sets the display name shown in `/resume`; `/session` shows the file, the message and token counts and the cost.

## Compaction

When the context gets close to the model's window, the agent compacts it: older messages are summarized by the model into one compaction entry, and the recent ones are kept as they are. The full history stays in the session file.

`/compact` compacts now. Text after it guides the summary:

```text
/compact keep the details of the parser bug and the failing test names
```

Automatic compaction is on by default and is toggled in `/settings` (Auto-compact). Its settings in settings.json are pi's:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

`reserveTokens` is the room kept free for the answer, `keepRecentTokens` how much recent context stays uncompacted. `compaction.modelOverrides`, keyed by `provider/model`, sets either value for one model.

## Export, import and share

```text
/export                        pi-session-<id>.html in the current folder
/export notes/session.html     an HTML page at that path
/export session.jsonl          the session file itself
/import ~/Downloads/session.jsonl
/share
/copy
```

The HTML export is pi's page: the conversation with its tool calls, thinking blocks and diffs, readable in any browser. `/import` loads a JSONL session and keeps its id and working directory. `/share` uploads the HTML export as a secret GitHub gist and prints its link; it uses the GitHub CLI (`gh`), signed in. `/copy` copies the last answer to the clipboard.

![/export: the exported session page](../web/assets/code/export.png)
