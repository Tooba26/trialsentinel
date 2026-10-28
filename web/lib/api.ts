import type { Overview } from "./types";
import snapshot from "@/data/overview.json";

/** Server-side base URL of the FastAPI model service. */
export const API_URL = (process.env.API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** Live overview from the API, falling back to the snapshot bundled at build time. */
export async function getOverview(): Promise<{ data: Overview; live: boolean }> {
  try {
    const res = await fetch(`${API_URL}/api/v1/analytics/overview`, {
      next: { revalidate: 3600 },
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) throw new Error(String(res.status));
    return { data: (await res.json()) as Overview, live: true };
  } catch {
    return { data: snapshot as unknown as Overview, live: false };
  }
}
