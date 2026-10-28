import { NextResponse } from "next/server";
import { API_URL } from "@/lib/api";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get("q") ?? "";
  if (q.trim().length < 2) return NextResponse.json([]);
  try {
    const res = await fetch(`${API_URL}/api/v1/sponsors?q=${encodeURIComponent(q)}`, {
      signal: AbortSignal.timeout(8000),
    });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch {
    return NextResponse.json([], { status: 503 });
  }
}
