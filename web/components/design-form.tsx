"use client";

import { Search } from "lucide-react";
import { useEffect, useState } from "react";
import type { AgeGroup, InterventionType, Masking, Phase, SponsorHit, TrialDesign } from "@/lib/types";
import { humanize, pct } from "@/lib/utils";
import { Field, MultiToggle, NumberSlider, Segmented, SelectField, SwitchField } from "./ui/controls";

const opt = <T extends string>(values: readonly T[]) => values.map((v) => ({ value: v, label: humanize(v) }));

const PHASES = opt(["PHASE1|PHASE2", "PHASE2", "PHASE2|PHASE3", "PHASE3"] as Phase[]);
const FUNDERS = opt(["INDUSTRY", "OTHER", "NIH", "OTHER_GOV", "FED", "NETWORK", "INDIV"]);
const PURPOSES = opt(["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "DIAGNOSTIC", "BASIC_SCIENCE",
  "HEALTH_SERVICES_RESEARCH", "SCREENING", "OTHER"]);
const MODELS = opt(["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL", "SEQUENTIAL"]);
const MASKING: { value: Masking; label: string }[] = [
  { value: "NONE", label: "Open" }, { value: "SINGLE", label: "Single" }, { value: "DOUBLE", label: "Double" },
  { value: "TRIPLE", label: "Triple" }, { value: "QUADRUPLE", label: "Quadruple" },
];
const AGES: { value: AgeGroup; label: string }[] = [
  { value: "CHILD", label: "Children" }, { value: "ADULT", label: "Adults" }, { value: "OLDER_ADULT", label: "Older adults" },
];
const TYPES: { value: InterventionType; label: string }[] = [
  { value: "DRUG", label: "Drug" }, { value: "BIOLOGICAL", label: "Biological" }, { value: "DEVICE", label: "Device" },
  { value: "PROCEDURE", label: "Procedure" }, { value: "BEHAVIORAL", label: "Behavioral" },
  { value: "RADIATION", label: "Radiation" }, { value: "DIETARY_SUPPLEMENT", label: "Supplement" },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="grid gap-5 border-t border-line pt-5 sm:grid-cols-2">
      <legend className="float-left mb-1 w-full font-display text-lg font-semibold text-ink sm:col-span-2">{title}</legend>
      {children}
    </fieldset>
  );
}

function SponsorSearch({ onPick }: { onPick: (s: SponsorHit) => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SponsorHit[]>([]);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => {
    if (q.trim().length < 2 || q === picked) { setHits([]); return; }
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`/api/sponsors?q=${encodeURIComponent(q)}`);
        setHits(r.ok ? await r.json() : []);
      } catch { setHits([]); }
    }, 250);
    return () => clearTimeout(t);
  }, [q, picked]);
  return (
    <Field label="Look up a real sponsor" hint="Fills in experience and track record from the registry." className="relative sm:col-span-2">
      <div className="relative">
        <Search size={15} aria-hidden className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint" />
        <input
          value={q} onChange={(e) => setQ(e.target.value)} placeholder="Pfizer, Mayo Clinic, NCI…"
          aria-label="Sponsor name" role="combobox" aria-expanded={hits.length > 0} aria-controls="sponsor-hits"
          className="h-9 w-full rounded-lg border border-line bg-field pl-9 pr-3 text-[14px] text-ink placeholder:text-ink-faint"
        />
      </div>
      {hits.length > 0 && (
        <ul id="sponsor-hits" role="listbox" className="absolute top-[68px] z-20 w-full overflow-hidden rounded-lg border border-line bg-field shadow-lg">
          {hits.map((h) => (
            <li key={h.sponsor} role="option" aria-selected={false}>
              <button
                type="button"
                onClick={() => { onPick(h); setPicked(h.sponsor); setQ(h.sponsor); setHits([]); }}
                className="flex w-full items-baseline justify-between gap-4 px-3 py-2 text-left text-[14px] hover:bg-line"
              >
                <span className="truncate">{h.sponsor}</span>
                <span className="tabular shrink-0 text-xs text-ink-faint">{h.trials} trials, {pct(h.termination_rate)} stopped</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Field>
  );
}

export function DesignForm({
  design, set, inferredArea,
}: { design: TrialDesign; set: (patch: Partial<TrialDesign>) => void; inferredArea?: string }) {
  const d = design;
  const sitesListed = d.n_sites !== null;
  return (
    <form className="flex flex-col gap-8" onSubmit={(e) => e.preventDefault()} aria-label="Trial design">
      <Section title="The study">
        <Field label="Primary condition" htmlFor="condition"
          hint={inferredArea ? `Therapeutic area: ${inferredArea}` : "Free text, as you would register it."}
          className="sm:col-span-2">
          <input
            id="condition" value={d.primary_condition} maxLength={200}
            onChange={(e) => set({ primary_condition: e.target.value })}
            className="h-9 rounded-lg border border-line bg-field px-3 text-[14px] text-ink"
          />
        </Field>
        <div className="sm:col-span-2">
          <Segmented label="Phase" value={d.phase} options={PHASES} onChange={(phase) => set({ phase })} />
        </div>
        <SelectField label="Primary purpose" value={d.primary_purpose} options={PURPOSES}
          onChange={(primary_purpose) => set({ primary_purpose })} />
        <SelectField label="Funder" value={d.funder_type} options={FUNDERS}
          onChange={(funder_type) => set({ funder_type })} />
        <NumberSlider label="Start year" value={d.start_year} min={2000} max={2030}
          onChange={(start_year) => set({ start_year })} />
        <SwitchField label="Pilot or feasibility study" checked={d.is_pilot} onChange={(is_pilot) => set({ is_pilot })} />
      </Section>

      <Section title="Design and blinding">
        <div className="sm:col-span-2">
          <Segmented label="Allocation" value={d.allocation}
            options={[{ value: "RANDOMIZED", label: "Randomized" }, { value: "NON_RANDOMIZED", label: "Non-randomized" },
              { value: "NA", label: "Single arm" }]}
            onChange={(allocation) => set({ allocation, ...(allocation !== "RANDOMIZED" ? { has_placebo: false } : {}) })} />
        </div>
        <div className="sm:col-span-2">
          <Segmented label="Blinding" value={d.masking} options={MASKING} onChange={(masking) => set({ masking })} />
        </div>
        <SelectField label="Intervention model" value={d.intervention_model} options={MODELS}
          onChange={(intervention_model) => set({ intervention_model })} />
        <SwitchField label="Placebo arm" checked={d.has_placebo} disabled={d.allocation !== "RANDOMIZED"}
          hint={d.allocation !== "RANDOMIZED" ? "Needs randomized allocation." : undefined}
          onChange={(has_placebo) => set({ has_placebo })} />
        <div className="sm:col-span-2">
          <MultiToggle label="Intervention types" value={d.intervention_types} options={TYPES}
            onChange={(intervention_types) => set({ intervention_types })} />
        </div>
        <NumberSlider label="Interventions listed" value={d.n_interventions} min={1} max={12}
          onChange={(n_interventions) => set({ n_interventions })} />
      </Section>

      <Section title="Who can enroll">
        <Segmented label="Sex" value={d.sex}
          options={[{ value: "ALL", label: "All" }, { value: "FEMALE", label: "Female" }, { value: "MALE", label: "Male" }]}
          onChange={(sex) => set({ sex })} />
        <MultiToggle label="Age groups" value={d.age_groups} options={AGES} onChange={(age_groups) => set({ age_groups })} />
      </Section>

      <Section title="Sites and sponsor">
        <SwitchField label="Sites are listed" checked={sitesListed}
          hint="Turn off if sites are not chosen yet."
          onChange={(on) => set(on ? { n_sites: 10, n_countries: 1 } : { n_sites: null, n_countries: null, has_us_site: false })} />
        <SwitchField label="Includes a US site" checked={d.has_us_site} disabled={!sitesListed}
          onChange={(has_us_site) => set({ has_us_site })} />
        <NumberSlider label="Sites" value={d.n_sites ?? 1} min={1} max={300} log disabled={!sitesListed}
          onChange={(n_sites) => set({ n_sites, n_countries: Math.min(d.n_countries ?? 1, n_sites) })} />
        <NumberSlider label="Countries" value={d.n_countries ?? 1} min={1} max={Math.max(1, Math.min(40, d.n_sites ?? 1))}
          disabled={!sitesListed || (d.n_sites ?? 1) <= 1} onChange={(n_countries) => set({ n_countries })} />
        <NumberSlider label="Collaborators" value={d.n_collaborators} min={0} max={15}
          onChange={(n_collaborators) => set({ n_collaborators })} />
        <SponsorSearch onPick={(s) => set({ sponsor_prior_trials: s.trials, sponsor_prior_termination_rate: s.trials ? s.termination_rate : null })} />
        <NumberSlider label="Sponsor's previous trials" value={d.sponsor_prior_trials} min={0} max={1500} log
          onChange={(v) => set({ sponsor_prior_trials: v, ...(v === 0 ? { sponsor_prior_termination_rate: null } : d.sponsor_prior_termination_rate === null ? { sponsor_prior_termination_rate: 0.1 } : {}) })} />
        <NumberSlider label="Share of those trials stopped early" step={0.01}
          value={d.sponsor_prior_termination_rate ?? 0} min={0} max={1}
          disabled={d.sponsor_prior_termination_rate === null}
          format={(v) => (d.sponsor_prior_termination_rate === null ? "no history" : pct(v))}
          onChange={(v) => set({ sponsor_prior_termination_rate: v })} />
      </Section>

      <Section title="Outcomes and registration">
        <NumberSlider label="Primary outcomes" value={d.n_primary_outcomes} min={1} max={10}
          onChange={(n_primary_outcomes) => set({ n_primary_outcomes })} />
        <NumberSlider label="Secondary outcomes" value={d.n_secondary_outcomes} min={0} max={30}
          onChange={(n_secondary_outcomes) => set({ n_secondary_outcomes })} />
        <NumberSlider label="Conditions listed" value={d.n_conditions} min={1} max={10}
          onChange={(n_conditions) => set({ n_conditions })} />
        <NumberSlider label="Registered relative to start" value={d.registration_lag_days} min={-365} max={1095} step={5}
          format={(v) => (v === 0 ? "on start day" : v < 0 ? `${-v} days before` : `${v} days after`)}
          onChange={(registration_lag_days) => set({ registration_lag_days })} />
      </Section>
    </form>
  );
}
