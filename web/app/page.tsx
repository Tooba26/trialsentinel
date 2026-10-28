import { Evidence, TierBreakdown } from "@/components/evidence";
import { StressTest } from "@/components/stress-test";
import { ThemeToggle } from "@/components/theme";
import { getOverview } from "@/lib/api";
import { MODEL_NAMES, pct } from "@/lib/utils";

export const revalidate = 3600;

const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/your-username/trialsentinel";

export default async function Home() {
  const { data: o } = await getOverview();
  const h = o.headline;
  const m = o.model;
  return (
    <div className="mx-auto max-w-[1240px] px-4 pb-28 sm:px-8 lg:pb-24">
      <header className="flex items-center justify-between gap-4 py-5">
        <a href="#" className="font-display text-xl font-semibold tracking-tight">TrialSentinel</a>
        <nav className="flex items-center gap-5 text-[14px] text-ink-soft">
          <a href="#evidence" className="hidden hover:text-ink sm:inline">Evidence</a>
          <a href="#model" className="hidden hover:text-ink sm:inline">How the model works</a>
          <a href={REPO_URL} className="hidden hover:text-ink sm:inline">Code</a>
          <ThemeToggle />
        </nav>
      </header>

      <section className="pb-10 pt-8 sm:pt-12">
        <h1 className="max-w-[16ch] font-display text-[44px] font-semibold leading-[1.02] tracking-tight sm:text-[68px]">
          Will this clinical trial finish?
        </h1>
        <p className="mt-5 max-w-[62ch] text-[17px] text-ink-soft">
          About one in six Phase 2 and 3 trials is terminated before it ends. Describe a trial the way you would
          register it, and see how its design compares with {h.trials_analyzed.toLocaleString()} trials on
          ClinicalTrials.gov that started between {h.start_year_window[0]} and {h.start_year_window[1]}.
        </p>
        <dl className="mt-8 grid max-w-[880px] grid-cols-1 gap-6 border-t border-line pt-6 sm:grid-cols-[auto_auto_1fr] sm:gap-12">
          <div>
            <dt className="text-[13px] text-ink-soft">Terminated early</dt>
            <dd className="tabular font-display text-3xl font-semibold">{pct(h.termination_rate, 1)}</dd>
            <dd className="text-xs text-ink-faint">{h.terminated_trials.toLocaleString()} trials</dd>
          </div>
          <div>
            <dt className="text-[13px] text-ink-soft">Model ROC-AUC</dt>
            <dd className="tabular font-display text-3xl font-semibold">{m.roc_auc.toFixed(2)}</dd>
            <dd className="text-xs text-ink-faint">on {(o.risk_tiers.reduce((s, t) => s + t.count, 0)).toLocaleString()} held-out trials</dd>
          </div>
          <div>
            <dt className="mb-2 text-[13px] text-ink-soft">How held-out trials split by predicted risk</dt>
            <dd><TierBreakdown overview={o} /></dd>
          </div>
        </dl>
      </section>

      <section aria-label="Trial design stress test" className="border-t border-line pt-10">
        <StressTest overview={o} />
      </section>

      <section id="evidence" className="mt-24 border-t border-line pt-10">
        <h2 className="font-display text-[32px] font-semibold tracking-tight">What the registry shows</h2>
        <p className="mt-2 max-w-[62ch] text-ink-soft">
          Raw termination rates across all {h.trials_analyzed.toLocaleString()} trials, before any modeling.
        </p>
        <div className="mt-8"><Evidence overview={o} /></div>
      </section>

      <section id="model" className="mt-24 border-t border-line pt-10">
        <h2 className="font-display text-[32px] font-semibold tracking-tight">How the model works</h2>
        <div className="mt-6 grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="flex max-w-[62ch] flex-col gap-4 text-[15px] text-ink-soft">
            <p>
              The model only sees what is known when a trial is registered: phase, design, blinding, sites,
              sponsor history and outcomes. Fields that reveal how a trial ended are excluded, including
              enrollment, which the registry overwrites with the actual number of patients once a trial closes.
            </p>
            <p>
              The selected model is <strong className="text-ink">{MODEL_NAMES[m.selected] ?? m.selected}</strong>,
              tuned with Bayesian optimization over 5-fold cross-validation and calibrated so its percentages
              match observed rates. The riskiest tenth of held-out trials were stopped{" "}
              <strong className="text-ink">{m.top_decile_lift.toFixed(1)}×</strong> as often as average, catching{" "}
              {pct(m.top_decile_capture)} of all terminations.
            </p>
            <p>
              A tougher test: trained only on trials started up to 2014 and scored on trials started 2017 onward,
              the model reaches a ROC-AUC of {m.temporal_roc_auc.toFixed(2)}, against {m.roc_auc.toFixed(2)} on a
              random split.{" "}
              {m.temporal_roc_auc / m.roc_auc >= 0.95
                ? "The patterns hold up over time."
                : "Most of the signal carries forward, but newer trials terminate more often and behave somewhat differently, so read scores for recent designs with that in mind."}
            </p>
            <p className="text-[13px] text-ink-faint">
              A high score is not a verdict on the science. Trials stop for many reasons, including slow
              recruitment, funding and business decisions. Use the score to question a design, not to judge one.
            </p>
          </div>
          <div className="flex flex-col gap-8">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[14px]">
                <caption className="mb-2 text-left font-display text-lg font-semibold text-ink">Candidate models</caption>
                <thead className="text-[13px] text-ink-soft">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 font-medium">CV PR-AUC</th>
                    <th className="py-2 pr-4 font-medium">Validation PR-AUC</th>
                    <th className="py-2 font-medium">Validation ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {m.comparison.map((c) => (
                    <tr key={c.model} className="border-b border-line">
                      <td className="py-2 pr-4">
                        {MODEL_NAMES[c.model] ?? c.model}
                        {c.model === m.selected && <span className="ml-2 text-xs text-low">selected</span>}
                      </td>
                      <td className="py-2 pr-4">{c.cv_pr_auc.toFixed(3)}</td>
                      <td className="py-2 pr-4">{c.val_pr_auc.toFixed(3)}</td>
                      <td className="py-2">{c.val_roc_auc.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-ink-faint">
                PR-AUC is the main metric because terminations are the minority class; random guessing scores {pct(h.termination_rate, 0)}.
                Test set: PR-AUC {m.pr_auc.toFixed(3)}, macro F1 {m.f1_macro.toFixed(3)}, accuracy {pct(m.accuracy, 1)}, Brier {m.brier.toFixed(3)}.
              </p>
            </div>
            <div>
              <h3 className="font-display text-lg font-semibold">Most influential factors overall</h3>
              <ol className="mt-3 flex flex-col gap-1.5">
                {o.global_importance.slice(0, 8).map((g, i) => {
                  const max = o.global_importance[0].mean_abs_shap;
                  return (
                    <li key={g.feature} className="grid grid-cols-[1.5rem_minmax(0,1fr)_minmax(0,40%)] items-center gap-3 text-[14px]">
                      <span className="tabular text-ink-faint">{i + 1}</span>
                      <span className="truncate">{g.label}</span>
                      <span className="h-1.5 rounded-full bg-ink" style={{ width: `${(g.mean_abs_shap / max) * 100}%` }} aria-hidden />
                    </li>
                  );
                })}
              </ol>
              <p className="mt-2 text-xs text-ink-faint">Mean absolute SHAP value on held-out trials.</p>
            </div>
          </div>
        </div>
      </section>

      <footer className="mt-24 border-t border-line pt-6 text-[13px] text-ink-faint">
        Data from ClinicalTrials.gov (U.S. National Library of Medicine). Withdrawn trials ({h.withdrawn_excluded.toLocaleString()})
        never enrolled a patient and are excluded. This tool is for research and education, not regulatory or medical decisions.
      </footer>
    </div>
  );
}
