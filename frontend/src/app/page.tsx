import { Workspace } from "@/components/layout/Workspace";

/**
 * Server component by convention; the workspace itself is interactive (streaming,
 * resizing, tab state) so it opts into the client.
 */
export default function Home() {
  return <Workspace />;
}
