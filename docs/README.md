# DTC-TRUS GitHub Project Page

Static GitHub Pages template for the DTC-TRUS manuscript:

**Distilling Temporal Coherence into 2D Networks for Transrectal Ultrasound Prostate Video Segmentation**

The page is intentionally prepared for a pre-publication state. Paper, code, dataset, author list, and BibTeX entries are placeholders and should be updated after publication or repository release.

## Files

```text
.
├── index.html
├── _config.yml
├── .nojekyll
├── robots.txt
└── assets
    ├── css/style.css
    ├── js/site.js
    └── img/
```

## Deployment on GitHub Pages

1. Copy this folder's contents to either the repository root or a `docs/` directory.
2. Commit and push to GitHub.
3. Open **Settings -> Pages**.
4. Select **Deploy from a branch**.
5. Select the target branch and either `/` or `/docs` depending on where you placed the files.
6. Save and wait for GitHub Pages to build.

## Publication checklist

Before making the page public, update the following items in `index.html`:

- Replace `Anonymous Authors` with the final author list and affiliations.
- Replace the disabled paper button with the official paper or preprint URL.
- Replace the code placeholder with the public repository URL.
- Replace the dataset placeholder with the release/request URL.
- Replace the BibTeX placeholder with the official citation.
- Re-check all figures for de-identification and publication permissions.

## Indexing checklist

This version contains pre-publication indexing protection:

- `index.html` includes `<meta name="robots" content="noindex, nofollow">`.
- `robots.txt` disallows crawling.

Remove both after the page is intended to be indexed publicly.

## Editing notes

The page is dependency-free: no build step, no npm package, and no Jekyll theme are required. Styling lives in `assets/css/style.css`; navigation and BibTeX copy behavior live in `assets/js/site.js`.
