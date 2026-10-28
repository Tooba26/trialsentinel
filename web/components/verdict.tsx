"use client";

import { AlertTriangle, Loader2 } from "lucide-react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Overview, Prediction } from "@/lib/types";
import { cn, humanize, MODEL_NAMES, pct, TIER_COLOR } from "@/lib/utils";

/** Where this trial sits among held-out historical trials. */
function RiskStrip({ p, histogram }: { p: number; histogram: Overview["probability_histogram"] }) {
  const bins = histogram.filter((b) => b.lo < 0.8);
  const total = histogram.reduce((s, b) => s + b.count, 0);
  const below = histogram.reduce((s, b) => s + (b.hi <= p ? b.count : b.lo < p ? b.count * ((p - b.lo) / (b.hi - b.lo)) : 0), 0);
  const percentile = total ? below / total : 0;
  const max = Math.max(...bins.map((b) => b.count), 1);
  const markerLeft = Math.min(p / 0.8, 1) * 100;
  return (
    <div>
      <p className="text-[14px] text-ink-soft">
        Riskier than <strong className="tabular font-semibold text-ink">{pct(percentile)}</strong> of{" "}
        {total.toLocaleString()} past trials the model had never seen.
      </p>
      <div className="relative mt-3 h-16" aria-hidden>
        <div className="flex h-full items-end gap-[2px]">
          {bins.map((b) => (
            <div
              key={b.lo}
              className="flex-1 rounded-t-[2px]"
              style={{
                height: `${Math.max(2, (b.count / max) * 100)}%`,
                background: b.hi <= p ? "var(--bar-muted)" : "color-mix(in srgb, var(--bar-muted) 45%, transparent)",
              }}
            />
          ))}
        </div>
        <div className="absolute inset-y-[-6px] w-[2px] bg-ink transition-[left] duration-500" style={{ left: `calc(${markerLeft}% - 1px)` }} />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-ink-faint tabular">
        <span>0%</span><span>20%</span><span>40%</span><span>60%</span><span>80%+</span>
      </div>
    </div>
  );
}

export function Verdict({
  prediction, overview, loading, error,
}: { prediction: Prediction | null; overview: Overview; loading: boolean; error: string | null }) {
  if (!prediction) {
    return (
      <div className="glass flex min-h-[420px] flex-col items-center justify-center gap-3 rounded-2xl p-8 text-center">
        {error ? (
          <>
            <AlertTriangle aria-hidden className="text-high" />
            <p className="max-w-xs text-[14px] text-ink-soft">{error}</p>
          </>
        ) : (
          <>
            <Loader2 aria-hidden className="animate-spin text-ink-faint" />
            <p className="text-[14px] text-ink-soft">Scoring the design…</p>
          </>
        )}
      </div>
    );
  }
  const p = prediction;
  const color = TIER_COLOR[p.risk_tier];
  const chart = p.contributions.slice(0, 10).map((c) => ({ ...c, name: c.label }));
  const rr = p.relative_risk;
  return (
    <section aria-label="Risk verdict" aria-live="polite" className={cn("glass rounded-2xl p-6 transition-opacity", loading && "opacity-70")}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[14px] text-ink-soft">Chance this trial is terminated early</p>
          <p className="tabular font-display text-[76px] font-semibold leading-none tracking-tight" style={{ color }}>
            {pct(p.probability)}
          </p>
        </div>
        <span className="mt-1 rounded-full px-3 py-1 text-[13px] font-semibold text-paper" style={{ background: color }}>
          {p.risk_tier} risk
        </span>
      </div>
      <p className="mt-3 text-[14px] text-ink-soft">
        {rr >= 1.05
          ? `${rr.toFixed(1)}× the rate of a typical Phase 2/3 trial (${pct(p.base_rate)}).`
          : rr <= 0.95
            ? `${(1 / rr).toFixed(1)}× lower than a typical Phase 2/3 trial (${pct(p.base_rate)}).`
            : `About the same as a typical Phase 2/3 trial (${pct(p.base_rate)}).`}
      </p>

      <div className="mt-6 border-t border-line pt-5">
        <RiskStrip p={p.probability} histogram={overview.probability_histogram} />
      </div>

      <div className="mt-6 border-t border-line pt-5">
        <h3 className="font-display text-lg font-semibold">What drives this score</h3>
        <ul className="mt-3 flex flex-col gap-2">
          {p.top_drivers.map((d) => (
            <li key={d.feature} className="flex items-baseline gap-3 text-[14px]">
              <span
                className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                style={{ background: d.contribution > 0 ? "var(--risk-critical)" : "var(--risk-low)" }}
                aria-hidden
              />
              <span>
                <span className="font-medium">{d.label}: {humanize(d.value)}</span>
                <span className="text-ink-faint"> {d.direction}</span>
              </span>
            </li>
          ))}
        </ul>
        <div className="mt-5 h-[300px]" role="img" aria-label="Contribution of each factor to the risk score">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
              <XAxis type="number" hide domain={["dataMin", "dataMax"]} />
              <YAxis type="category" dataKey="name" width={170} tickLine={false} axisLine={false}
                tick={{ fill: "var(--ink-soft)", fontSize: 12 }} />
              <ReferenceLine x={0} stroke="var(--line)" />
              <Tooltip
                cursor={{ fill: "var(--line)" }}
                contentStyle={{ background: "var(--field)", border: "1px solid var(--line)", borderRadius: 8, fontSize: 13 }}
                formatter={(v, _n, item) => [`${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(3)} log-odds`, (item?.payload as { value: string }).value]}
              />
              <Bar dataKey="contribution" radius={3} isAnimationActive={false}>
                {chart.map((c) => (
                  <Cell key={c.feature} fill={c.contribution > 0 ? "var(--risk-critical)" : "var(--risk-low)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-ink-faint">
          SHAP contributions from the {MODEL_NAMES[p.model] ?? p.model} model. Red pushes risk up, teal pulls it down.
        </p>
        {p.warnings.map((w) => (
          <p key={w} className="mt-2 flex gap-2 text-xs text-ink-soft"><AlertTriangle size={14} aria-hidden className="shrink-0 text-mid" />{w}</p>
        ))}
        {error && <p className="mt-2 text-xs text-critical">{error}</p>}
      </div>
    </section>
  );
}
