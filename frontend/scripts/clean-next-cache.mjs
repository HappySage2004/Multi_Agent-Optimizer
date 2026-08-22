/**
 * Delete `.next` before `next dev` and `next build`.
 *
 * The problem: this repo lives under a OneDrive-synced path, and Next reads manifests out
 * of `.next` with calls that end in `readlink`. On a path OneDrive is managing that throws
 * `EINVAL`, and the command dies:
 *
 *   [Error: EINVAL: invalid argument, readlink '...\.next\cache']
 *   [Error: EINVAL: invalid argument, readlink '...\.next\server\next-font-manifest.json']
 *
 * Two things worth knowing, both established by measurement rather than assumption,
 * because each contradicts the obvious guess:
 *
 *   1. The `ReparsePoint` attribute is NOT the signal. Every single entry under a synced
 *      path carries it — all ~18k of `node_modules` does, permanently, and that never
 *      breaks. So "is it a reparse point" cannot be used to detect the bad state.
 *   2. The pinned attribute (`attrib +P`, OneDrive's "always keep on this device") does
 *      NOT help. A pinned entry keeps its content local but stays a reparse point
 *      (`.next` reads `0x80410` = Directory + ReparsePoint + Pinned), and a build over a
 *      fully pinned `.next` still failed.
 *
 * What is left is that the failure tracks how long `.next` has been sitting in the synced
 * tree: a directory just created has not been enrolled by the sync engine yet, and every
 * build over a freshly deleted `.next` has succeeded. So the mitigation is to never let
 * one persist. This costs the incremental cache — a cold build here is ~6s against ~2s
 * warm, which is the right trade for a command that otherwise fails outright.
 *
 * This is a mitigation, not a guarantee. The failure is intermittent, so a long `next dev`
 * session can still be swept after startup; re-running is the workaround. The only actual
 * fix is to keep the project outside OneDrive, which would also spare the Python venv,
 * DuckDB and `node_modules`. Delete this script and its two hooks if the repo moves.
 *
 * Relocating `distDir` outside OneDrive was tried and rejected: the build output lands
 * clean (0 reparse points), but Next generates `.next/types/*` that cannot resolve
 * `next/dist/...` from outside the project, so typechecking breaks.
 */

import { rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// `force` makes a missing directory a no-op, so a first-ever run is not an error.
rmSync(join(projectRoot, ".next"), { recursive: true, force: true });
