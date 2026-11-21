// `saetta` (no arguments, in a terminal) and `saetta auto`: a guided start. Shows this Mac
// (chip, memory, free disk), lists the model packs best first with the ones that do not fit
// greyed out, downloads the chosen one if needed, finds the coding agents installed (Claude
// Code, pi, opencode) and offers the official installer for a missing one, then runs
// `saetta start <agent> --pack <name>`. Arrow keys (or j / k) move, Enter selects, q quits.
#ifndef SAETTA_WIZARD_H
#define SAETTA_WIZARD_H

#include "base.h"

// Sets *code to the process exit code (the agent's, or 1 / 0 when stopped early). Errors:
// "NotATerminal" when stdin or stdout is not a terminal.
Err wizard_run(int *code);

#endif
