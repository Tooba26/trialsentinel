"use client";

import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Overview, Rate } from "@/lib/types";
import { humanize, pct, TIER_COLOR } from "@/lib/utils";

const tip = {
  contentStyle: { background: "var(--field)", border: "1px solid var(--line)", borderRadius: 8, fontSize: 13 },
  cursor: { fill: "var(--line)" },
};

function RateBars({ title, note, data, base, height }: {
  title: string; note: string; data: Rate[]; base: number; height: number;
}) {
  const rows = data.map((r) => ({ ...r, name: humanize(r.group) }));
  return (
    <figure className="flex flex-col gap-3">
      <figcaption>
        <h3 className="font-display text-lg font-semibold">{title}</h3>
        <p className="text-[13px] text-ink-soft">{note}</p>
      </figcaption>
      <div style={{ height }} role="img" aria-label={`${title}: ${rows.map((r) => `${r.name} ${pct(r.termination_rate)}`).join(", ")}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
            <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={(v) => pct(v)} tick={{ fill: "var(--ink-faint)", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={170} tick={{ fill: "var(--ink-soft)", fontSize: 12 }} axisLine={false} tickLine={false} />
            <ReferenceLine x={base} stroke="var(--ink-faint)" strokeDasharray="3 3" />
            <Tooltip {...tip} formatter={(v, _n, item) => [`${pct(Number(v), 1)} of ${(item?.payload as Rate).trials.toLocaleString()} trials`, "Terminated"]} />
            <Bar dataKey="termination_rate" fill="var(--ink)" radius={3} barSize={14} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function Evidence({ overview }: { overview: Overview }) {
  const base = overview.headline.termination_rate;
  const s = overview.segments;
  return (
    <div className="grid gap-12 lg:grid-cols-2">
      <RateBars title="Therapeutic area" note="Dashed line marks the overall rate." data={s.therapeutic_area} base={base} height={420} />
      <div className="flex flex-col gap-12">
        <RateBars title="Sponsor's track record"
          note="Share of the sponsor's earlier trials that were stopped, measured before each new trial began."
          data={s.sponsor_track_record} base={base} height={180} />
        <figure className="flex flex-col gap-3">
          <figcaption>
            <h3 className="font-display text-lg font-semibold">By start year</h3>
            <p className="text-[13px] text-ink-soft">Rates climb for recent start years. Part of the rise is real; part is because long trials started recently are still running and not yet counted.</p>
          </figcaption>
          <div className="h-[180px]" role="img" aria-label="Termination rate by trial start year">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={overview.by_start_year} margin={{ top: 4, right: 16, bottom: 0, left: -12 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="year" tick={{ fill: "var(--ink-faint)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tickFormatter={(v) => pct(v)} tick={{ fill: "var(--ink-faint)", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip {...tip} cursor={{ stroke: "var(--line)" }} formatter={(v) => [pct(Number(v), 1), "Terminated"]} />
                <Line dataKey="termination_rate" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </figure>
      </div>
    </div>
  );
}

export function TierBreakdown({ overview }: { overview: Overview }) {
  const tiers = overview.risk_tiers;
  return (
    <div>
      <div className="flex h-3 overflow-hidden rounded-full" role="img"
        aria-label={tiers.map((t) => `${t.tier} ${pct(t.share)}`).join(", ")}>
        {tiers.map((t) => (
          <div key={t.tier} style={{ width: `${t.share * 100}%`, background: TIER_COLOR[t.tier] }} />
        ))}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
        {tiers.map((t) => (
          <div key={t.tier}>
            <dt className="flex items-center gap-1.5 text-[13px] font-medium">
              <span className="h-2 w-2 rounded-full" style={{ background: TIER_COLOR[t.tier] }} aria-hidden />
              {t.tier}
            </dt>
            <dd className="tabular text-[13px] text-ink-soft">{pct(t.share)} of trials</dd>
            <dd className="tabular text-[13px] text-ink-soft">
              {t.observed_termination_rate === null ? "none" : pct(t.observed_termination_rate)} were stopped
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
