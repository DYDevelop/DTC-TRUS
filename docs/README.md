# DTC-TRUS GitHub Project Page

Static GitHub Pages template for:

**Distilling Temporal Coherence into 2D Networks for Transrectal Ultrasound Prostate Video Segmentation**

This version includes the public TRUS-V dataset link on KHDP:

<https://khdp.net/database/data-search-detail/TRUS-V>

The MICCAI paper URL is still intentionally left as a placeholder until publication.

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

## Update notes

- Dataset CTA now points to the TRUS-V KHDP page.
- Dataset section now states that TRUS-V is available through KHDP.
- Materials section now marks Dataset as `Available` and includes the direct KHDP record link.
- The broken Dataset CTA anchor in the uploaded `index.html` has been closed correctly.
- Paper button remains `MICCAI paper coming soon` until an official paper/preprint URL is ready.

## Indexing note

`robots.txt` currently disallows crawling. Remove or update `robots.txt` when the project page should be indexed publicly.

## Editing notes

The page is dependency-free: no build step, no npm package, and no Jekyll theme are required. Styling lives in `assets/css/style.css`; navigation and BibTeX copy behavior live in `assets/js/site.js`.
