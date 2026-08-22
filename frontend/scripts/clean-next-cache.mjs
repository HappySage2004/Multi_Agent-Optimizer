/**
 * Delete `.next` before a production build.
 *
 * Why this exists: the repo lives under a OneDrive-synced path. OneDrive dehydrates
 * synced files into cloud placeholders, which carry the Windows `ReparsePoint` attribute
 * but are not symlinks — and Node's `readlink` throws `EINVAL` on one. Next.js calls
 * `readlink` while reading `.next/diagnostics/*.json`, so the *second* build after
 * OneDrive has swept the directory fails with:
 *
 *   [Error: EINVAL: invalid argument, readlink '...\.next\diagnostics\framework.json']
 *
 * A build output that never survives between builds can never be dehydrated, so clearing
 * it first sidesteps the whole class of problem. Builds here take ~7s, so losing the
 * incremental cache costs nothing.
 *
 * This is a workaround, not a fix. The fix is to keep the project outside OneDrive, which
 * would also spare the Python venv, DuckDB and the 18k files in `node_modules`. Delete
 * this script and its `prebuild` hook if the repo ever moves.
 *
 * Note `next dev` writes `.next` too and can hit the same error. It is deliberately not
 * cleaned here — dev leans on that cache for fast restarts. Run this by hand if it bites:
 *   node scripts/clean-next-cache.mjs
 */

import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(projectRoot, ".next");

// `force` makes a missing directory a no-op, so a first-ever build is not an error.
rmSync(target, { recursive: true, force: true });
