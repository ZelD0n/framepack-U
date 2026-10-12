'use strict';
// Main results transcribed from Table 1 of the supplied paper.
// Columns: LongVideoBench, MLVU, Video-MME (overall).
const results = {
  moe: {
    128: [[59.4,71.8,66.7],[50.3,66.7,61.5],[52.2,65.6,60.7],[56.5,63.9,61.3],[72.3,82.4,76.4]],
    256: [[60.3,73.0,67.1],[51.2,68.4,63.9],[56.9,67.5,64.6],[57.5,67.7,64.8],[72.6,82.9,75.5]],
    512: [[59.8,74.3,67.3],[57.9,68.4,64.5],[55.2,69.4,64.4],[59.9,69.3,65.4],[73.0,84.7,75.8]]
  },
  dense: {
    128: [[68.0,75.9,73.6],[59.9,67.3,61.0],[57.7,67.4,63.2],[57.8,65.4,63.5],[74.4,83.9,76.3]],
    256: [[70.9,76.7,75.9],[60.8,68.9,62.2],[62.8,70.7,65.2],[61.6,67.7,65.4],[74.3,83.2,77.9]],
    512: [[71.8,78.0,74.4],[58.5,68.4,61.1],[63.7,69.9,67.2],[62.8,70.5,66.9],[74.9,82.1,76.3]]
  }
};
const methods = ['Vanilla','FastVID','VisionZip','FrameFusion','FramePack-U'];
const proprietaryResults = [
  ['GPT-5', [72.6, 77.7, 83.3]],
  ['GPT-5 mini', [69.7, 69.1, 77.3]],
  ['Gemini 3 Pro', [75.9, 75.7, 88.6]],
  ['Gemini 2.5 Flash', [73.1, 75.1, 84.2]],
  ['Claude Sonnet 4.5', [65.1, 64.0, 74.2]]
];
document.getElementById('proprietary-results').innerHTML =
  '<tr class="table-group"><th scope="rowgroup" colspan="4">Proprietary models</th></tr>' +
  proprietaryResults.map(([name, scores]) => `<tr><th scope="row">${name}</th>${scores.map(value => `<td>${value.toFixed(1)}</td>`).join('')}</tr>`).join('');
const modelSelect = document.getElementById('model-select');
let selectedBudget = 128;
function renderResults() {
  const rows = results[modelSelect.value][selectedBudget];
  document.getElementById('results-body').innerHTML = `<tr class="table-group"><th scope="rowgroup" colspan="4">${modelSelect.selectedOptions[0].textContent} · ${selectedBudget}-frame-equivalent budget</th></tr>` + rows.map((scores,i) => `<tr class="${i===4?'ours-row':''}"><th scope="row"${i===4?' style="background:inherit;color:inherit"':''}>${methods[i]}${i===4?'<span class="ours-mark">Ours</span>':''}</th>${scores.map(v=>`<td>${v.toFixed(1)}</td>`).join('')}</tr>`).join('');

}
modelSelect.addEventListener('change',renderResults);
document.querySelectorAll('[data-budget]').forEach(button=>button.addEventListener('click',()=>{
  selectedBudget=Number(button.dataset.budget);
  document.querySelectorAll('[data-budget]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  renderResults();
}));
renderResults();

const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('.figure-open').forEach(button=>button.addEventListener('click',()=>{
  const img=document.getElementById('dialog-image');img.src=button.dataset.image;img.alt=button.querySelector('img').alt;
  document.getElementById('dialog-caption').textContent=button.dataset.caption;
  document.getElementById('dialog-original').href=button.dataset.image;
  dialog.showModal();document.body.classList.add('dialog-open');
}));
document.getElementById('dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('close',()=>document.body.classList.remove('dialog-open'));
dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}});

const copyButton = document.getElementById('copy-citation');
let copyResetTimer;
function resetCopyButton() {
  copyButton.classList.remove('is-copied');
  copyButton.querySelector('use').setAttribute('href', '#i-copy');
  copyButton.setAttribute('aria-label', 'Copy citation');
  copyButton.title = 'Copy citation';
}
copyButton.addEventListener('click', async () => {
  clearTimeout(copyResetTimer);
  const text = document.getElementById('bibtex').textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.append(area);
      area.select();
      const success = document.execCommand('copy');
      area.remove();
      copyButton.focus({preventScroll: true});
      if (!success) throw new Error('Clipboard unavailable');
    }
    copyButton.classList.add('is-copied');
    copyButton.querySelector('use').setAttribute('href', '#i-check');
    copyButton.setAttribute('aria-label', 'Citation copied');
    copyButton.title = 'Copied';
    copyResetTimer = setTimeout(resetCopyButton, 2500);
  } catch {
    resetCopyButton();
    copyButton.setAttribute('aria-label', 'Copy failed; select the BibTeX text to copy');
    copyButton.title = 'Select the BibTeX text to copy';
  }
});

const sections=[...document.querySelectorAll('main section[id]')];
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('.site-header nav a').forEach(link=>link.classList.toggle('active',link.getAttribute('href')===`#${entry.target.id}`));}});},{rootMargin:'-15% 0px -65% 0px',threshold:0});sections.forEach(section=>observer.observe(section));}

document.getElementById('back-to-top').addEventListener('click', () => {
  window.scrollTo({
    top: 0,
    behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'
  });
});

// Reveal the navigation when scrolling up; hide it when scrolling down.
const siteHeader = document.querySelector('.site-header');
let headerScrollAnchor = Math.max(0, window.scrollY);
let headerScrollScheduled = false;
window.addEventListener('scroll', () => {
  if (headerScrollScheduled) return;
  headerScrollScheduled = true;
  requestAnimationFrame(() => {
    const currentY = Math.max(0, window.scrollY);
    const delta = currentY - headerScrollAnchor;
    if (currentY <= siteHeader.offsetHeight) {
      siteHeader.classList.remove('is-hidden');
      headerScrollAnchor = currentY;
    } else if (Math.abs(delta) >= 8) {
      siteHeader.classList.toggle('is-hidden', delta > 0);
      headerScrollAnchor = currentY;
    }
    headerScrollScheduled = false;
  });
}, { passive: true });
