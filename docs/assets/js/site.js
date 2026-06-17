const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const copyButton = document.getElementById('copy-bibtex');
const bibtex = document.getElementById('bibtex');

if (copyButton && bibtex) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibtex.innerText.trim());
      const original = copyButton.textContent;
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch (error) {
      copyButton.textContent = 'Copy manually';
    }
  });
}
