"use client";

import { RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_DESIGN, PRESETS } from "@/lib/presets";
import type { Overview, Prediction, TrialDesign } from "@/lib/types";
import { pct, TIER_COLOR } from "@/lib/utils";
import { DesignForm } from "./design-form";
import { Verdict } from "./verdict";

export function StressTest({ overview }: { overview: Overview }) {
  const [design, setDesign] = useState<TrialDesign>(DEFAULT_DESIGN);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preset, setPreset] = useState<string | null>(null);
  const seq = useRef(0);

  const set = useCallback((patch: Partial<TrialDesign>) => {
    setPreset(null);
    setDesign((d) => ({ ...d, ...patch }));
  }, []);

  useEffect(() => {
    if (design.primary_condition.trim().length < 2) return;
    const id = ++seq.current;
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(design),
        });
        const body = await res.json();
        if (id !== seq.current) return;
        if (!res.ok) {
          const detail = Array.isArray(body.detail) ? body.detail.map((d: { msg: string }) => d.msg).join("; ") : body.detail;
          setError(detail ?? "The model could not score this design.");
        } else {
          setPrediction(body as Prediction);
          setError(null);
        }
      } catch {
        if (id === seq.current) setError("Could not reach the model service. Check your connection and try again.");
      } finally {
        if (id === seq.current) setLoading(false);
      }
    }, 280);
    return () => clearTimeout(t);
  }, [design]);

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,440px)] lg:gap-12">
      <div>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Example designs">
          <span className="mr-1 text-[13px] text-ink-soft">Start from</span>
          {PRESETS.map((p) => (
            <button
              key={p.name} type="button" title={p.description} aria-pressed={preset === p.name}
              onClick={() => { setDesign(p.design); setPreset(p.name); }}
              className="rounded-full border border-line px-3 py-1 text-[13px] text-ink-soft hover:border-ink hover:text-ink aria-pressed:border-ink aria-pressed:bg-ink aria-pressed:text-paper"
            >
              {p.name}
            </button>
          ))}
          <button type="button" onClick={() => { setDesign(DEFAULT_DESIGN); setPreset(null); }}
            className="inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-[13px] text-ink-faint hover:text-ink">
            <RotateCcw size={13} aria-hidden /> Reset
          </button>
        </div>
        <div className="mt-6">
          <DesignForm design={design} set={set} inferredArea={prediction?.therapeutic_area} />
        </div>
      </div>
      <div id="verdict" className="scroll-mt-6 lg:sticky lg:top-6 lg:self-start">
        <Verdict prediction={prediction} overview={overview} loading={loading} error={error} />
      </div>
      {prediction && (
        <a
          href="#verdict"
          className="glass fixed inset-x-3 bottom-3 z-30 flex items-center justify-between gap-3 rounded-xl px-4 py-3 shadow-lg lg:hidden"
          aria-label={`Current score ${pct(prediction.probability)}, ${prediction.risk_tier} risk. Jump to the explanation.`}
        >
          <span className="flex items-baseline gap-2">
            <span className="tabular font-display text-2xl font-semibold" style={{ color: TIER_COLOR[prediction.risk_tier] }}>
              {pct(prediction.probability)}
            </span>
            <span className="text-[13px] text-ink-soft">{prediction.risk_tier} risk of termination</span>
          </span>
          <span className="text-[13px] font-medium text-ink underline underline-offset-4">See why</span>
        </a>
      )}
    </div>
  );
}
