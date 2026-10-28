import { NextResponse } from "next/server";
import { API_URL } from "@/lib/api";

// Serverless proxy: the browser talks to this route, so the model service URL
// stays server-side and no CORS setup is needed.
export async function POST(req: Request) {
  const body = await req.text();
  try {
    const res = await fetch(`${API_URL}/api/v1/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      signal: AbortSignal.timeout(15000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "The model service did not respond. It may be waking up; try again in a few seconds." },
      { status: 503 },
    );
  }
}
