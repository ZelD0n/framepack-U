# FramePack-U project page

Static academic project page for **Elephant in the Fridge: Constant-Memory Frame Packing for Long Video Understanding**.

## Preview

```bash
npm run dev
```

Open http://localhost:5173. Alternatively, run `python3 -m http.server 5173` or open `index.html` directly. The site requires no build step or third-party runtime dependencies.

## Contents

- Paper title, authors, affiliations, and NeurIPS 2026 venue label
- Paper PDF, release/v1.0 code, Hugging Face, ModelScope links
- Method overview
- Compact benchmark table with five proprietary model references, two open-weight backbones, and three budgets
- Memory/latency figures, two qualitative examples, and three t-SNE visualizations
- Vertically stacked qualitative and t-SNE examples
- Concise limitation and discussion summaries from Section 6 of the paper
- Figure enlargement and an inline BibTeX copy button with a checkmark confirmation
- Responsive layout and keyboard-accessible controls

## Updating the page

- `index.html`: paper metadata, prose, resource links, and displayed citation
- `app.js`: Table 1 benchmark values and figure/citation interactions
- `styles.css`: layout, typography, and responsive styles
- `assets/citation.bib`: downloadable citation; keep it in sync with `index.html`
- `assets/`: optimized figures and the supplied paper PDF

The supplied PDF labels itself **Preprint** and has no arXiv identifier or finalized proceedings citation. The paper button therefore serves the local PDF, and the BibTeX uses `@misc` with the known author/title/year metadata. Replace its note/URL and add the arXiv identifier or proceedings metadata when those are available. The venue label follows the requested NeurIPS 2026 framing; it does not assert acceptance or an award.

The abstract reproduces the supplied manuscript verbatim, with only the `[Shuo]:` marker removed and PDF line wraps normalized. Performance scores use **Table 1**, including **overall** Video-MME accuracy, and improvements are percentage points. The qualitative examples use Figure 3's original artwork. t-SNE exports correspond to Figure 4. Runtime charts use the supplied original exports for Figures 5 and 6. External release/download URLs were copied from the repository's `release/v1.0` README.

## Regenerate optimized images

This is optional; exported assets are already included.

```bash
python3 -m venv .venv
.venv/bin/pip install pymupdf pillow
.venv/bin/python scripts/export_assets.py /Users/shuogao/Desktop/FramePack-U
```

## Deploy

Serve the directory as a static site. For GitHub Pages, publish the repository root from the desired branch; no framework configuration, package installation, or absolute asset paths are required. Publication has not been performed.

## Font

Lato matches the font used on https://zeld0n.github.io/shuo_homepage/. Regular and bold faces are hosted locally in `assets/fonts/`. Its SIL Open Font License is included in `assets/fonts/Lato-OFL.txt`. The page does not depend on an external font CDN.
