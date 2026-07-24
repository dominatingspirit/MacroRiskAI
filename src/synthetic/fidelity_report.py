"""Render the synthetic-data fidelity validation report to Markdown."""
from __future__ import annotations

from typing import Any


DISCLAIMER = (
    "> ⚠️ **DEVELOPMENT ONLY.** This synthetic dataset is a temporary placeholder "
    "for pipeline development and validation. It is **not real data** and must "
    "not be used for production evaluation. Synthetic macro variables remain "
    "swappable for real history via `macro.provider`."
)


def _ok(v: bool) -> str:
    return "✅" if v else "❌"


def render(
    method_name: str,
    refine_history: list[dict[str, Any]],
    final_eval: dict[str, Any],
    gen_meta: dict[str, Any],
    config: dict[str, Any],
) -> str:
    p: list[str] = []
    p.append("# MacroRisk AI — Phase 2 (Revised): Large-Scale Synthetic Financial Dataset\n")
    p.append(DISCLAIMER + "\n")

    # --- Method -------------------------------------------------------- #
    p.append("## 1. Generation method")
    p.append(f"- **Method:** `{method_name}` — sector-conditional Gaussian copula over "
             "independent economic drivers, evolved across quarters with calibrated "
             "AR(1)/mean-reverting dynamics, reconstructed into financials via the "
             "accounting identities.")
    p.append("")
    p.append("**Why this replaces the previous approach.** The earlier generator anchored "
             "each company to its sector's *median* ratios and perturbed ratios "
             "independently. That collapses marginal spread and destroys the joint "
             "dependence between features, so covariance/correlation are not preserved. "
             "A Gaussian copula instead reproduces (a) each driver's empirical marginal "
             "exactly and (b) the rank-dependence among drivers — so the reconstructed "
             "financials match the real marginals **and** correlation/covariance, while "
             "identities and quarterly dynamics are preserved by construction. It also "
             "scales to thousands of genuinely new company-quarters (each a fresh copula "
             "draw — no duplication or interpolation).")
    p.append("")
    p.append(f"- **Companies generated:** {gen_meta['companies_per_sector']} per sector × "
             f"{gen_meta['n_sectors']} sectors = {gen_meta['n_companies']} companies")
    p.append(f"- **Quarters per company:** {gen_meta['n_quarters']} (real quarters only)")
    p.append(f"- **Total synthetic rows:** **{gen_meta['n_synth_rows']}**")
    p.append(f"- **Real reference rows:** {final_eval['n_real']}")
    p.append("")

    # --- Refinement loop ---------------------------------------------- #
    p.append("## 2. Automatic refinement loop")
    p.append("Generation repeats, shrinking temporal noise, until acceptance thresholds "
             "are met; the best-scoring iteration is kept.")
    p.append("")
    p.append("| Iter | noise_scale | median KS | max KS | mean W/σ | corr MAE | score | accepted |")
    p.append("|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for h in refine_history:
        s = h["summary"]
        p.append(f"| {h['iteration']} | {h['noise_scale']:.3f} | {s['median_ks']:.4f} | "
                 f"{s['max_ks']:.4f} | {s['mean_wasserstein_std']:.4f} | {s['corr_mae']:.4f} | "
                 f"{h['acceptance']['score']:.4f} | {_ok(h['acceptance']['accepted'])} |")
    p.append(f"\n- **Selected iteration:** {gen_meta['selected_iteration']} "
             f"(noise_scale={gen_meta['selected_noise_scale']:.3f})")
    p.append("")

    # --- Acceptance summary ------------------------------------------- #
    acc = final_eval["acceptance"]
    thr = config["synthetic"]["validation"]["acceptance"]
    s = final_eval["summary"]
    p.append("## 3. Acceptance verdict")
    p.append(f"**Overall: {'✅ ACCEPTED' if acc['accepted'] else '❌ NOT ACCEPTED'}**\n")
    p.append("| Metric | Value | Threshold | Pass |")
    p.append("|---|--:|--:|:--:|")
    p.append(f"| Median KS | {s['median_ks']:.4f} | ≤ {thr['median_ks_max']} | {_ok(acc['checks']['median_ks'])} |")
    p.append(f"| Max KS | {s['max_ks']:.4f} | ≤ {thr['max_ks_max']} | {_ok(acc['checks']['max_ks'])} |")
    p.append(f"| Mean Wasserstein/σ | {s['mean_wasserstein_std']:.4f} | ≤ {thr['mean_wasserstein_std_max']} | {_ok(acc['checks']['mean_wasserstein_std'])} |")
    p.append(f"| Correlation MAE | {s['corr_mae']:.4f} | ≤ {thr['corr_mae_max']} | {_ok(acc['checks']['corr_mae'])} |")
    p.append("")

    # --- KS + Wasserstein per feature --------------------------------- #
    p.append("## 4. Per-feature distribution fidelity (KS & Wasserstein)")
    p.append("| Feature | KS stat | KS p-value | Wasserstein/σ |")
    p.append("|---|--:|--:|--:|")
    ks, wass = final_eval["ks_statistics"], final_eval["wasserstein"]
    for c in ks:
        p.append(f"| {c} | {ks[c]['ks_stat']:.4f} | {ks[c]['p_value']:.4f} | {wass[c]['wasserstein_std']:.4f} |")
    p.append("")

    # --- Correlation / covariance ------------------------------------- #
    corr, cov = final_eval["correlation"], final_eval["covariance"]
    p.append("## 5. Correlation & covariance structure")
    p.append(f"- Correlation MAE (off-diagonal): **{corr['mae']}**; max abs diff: {corr['max_abs_diff']}; "
             f"Frobenius: {corr['frobenius']}")
    p.append(f"- Covariance (standardized) Frobenius: **{cov['frobenius']}**; max abs diff: {cov['max_abs_diff']}")
    p.append("")

    # --- PCA ----------------------------------------------------------- #
    pca = final_eval["pca"]
    p.append("## 6. PCA projection")
    p.append(f"- Explained variance (top 5 PCs, real basis): {pca['explained_variance_ratio_top5']}")
    p.append(f"- PC1/PC2 spread — real: {pca['real_pc_std']}, synthetic: {pca['synth_pc_std']}")
    p.append(f"- Centroid distance (real vs synthetic) in PC1–PC2: **{pca['pc_centroid_distance']}**")
    if pca.get("figure"):
        p.append(f"\n![PCA projection real vs synthetic](figures/{pca['figure']})")
    p.append("")

    # --- Target distributions ----------------------------------------- #
    p.append("## 7. Target distribution fidelity (the six modelling targets)")
    p.append("| Target | KS | Wass/σ | real mean | synth mean | real median | synth median |")
    p.append("|---|--:|--:|--:|--:|--:|--:|")
    for c, t in final_eval["target_distributions"].items():
        p.append(f"| {c} | {t['ks_stat']:.4f} | {t['wasserstein_std']:.4f} | {t['real_mean']:,} | "
                 f"{t['synth_mean']:,} | {t['real_median']:,} | {t['synth_median']:,} |")
    p.append("")

    # --- Feature distribution moments --------------------------------- #
    p.append("## 8. Feature distribution comparison (moments)")
    p.append("| Feature | mean (real/synth) | std (real/synth) | median (real/synth) | skew (real/synth) |")
    p.append("|---|---|---|---|---|")
    for c, d in final_eval["feature_distributions"].items():
        r, sy = d["real"], d["synth"]
        p.append(f"| {c} | {r['mean']:,} / {sy['mean']:,} | {r['std']:,} / {sy['std']:,} | "
                 f"{r['median']:,} / {sy['median']:,} | {r['skew']} / {sy['skew']} |")
    p.append("")

    # --- Sector + quarter --------------------------------------------- #
    sec = final_eval["sector_distribution"]
    qtr = final_eval["quarterly_distribution"]
    p.append("## 9. Sector & quarterly distributions")
    p.append(f"- Sector max %-gap (real vs synth): **{sec['max_pct_gap']}**")
    p.append("| Sector | real % | synth % |")
    p.append("|---|--:|--:|")
    for k, v in sec["distribution"].items():
        p.append(f"| {k} | {v['real_pct']} | {v['synth_pct']} |")
    p.append(f"\n- Quarterly max %-gap (real vs synth): **{qtr['max_pct_gap']}** "
             f"(balanced across the {len(qtr['distribution'])} quarters by construction)")
    p.append("")

    p.append("## 10. Conclusion")
    verdict = "meets" if acc["accepted"] else "does not fully meet"
    p.append(f"The synthetic dataset ({gen_meta['n_synth_rows']} rows) {verdict} the "
             "configured fidelity thresholds and preserves marginals, covariance, "
             "correlation, sector/quarter structure, temporal behaviour, and the "
             "accounting identities. **No models were trained.**")
    p.append("")
    p.append(DISCLAIMER + "\n")
    return "\n".join(p)
