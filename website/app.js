(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('result-data').textContent);
  const metrics = {macro_f1: 'Macro-F1', accuracy: 'Accuracy', kappa: 'Cohen’s κ'};
  const buttons = document.querySelectorAll('[data-metric]');
  function showMetric(key) {
    const minimum = key === 'kappa' ? -0.2 : 0;
    const range = 1 - minimum;
    document.querySelectorAll('.chart-row').forEach((row, i) => {
      const value = data[i][key];
      const bar = row.querySelector('.bar');
      row.querySelector('.track').style.setProperty('--zero', `${-minimum / range * 100}%`);
      bar.style.left = `${(Math.min(value, 0) - minimum) / range * 100}%`;
      bar.style.width = `${Math.abs(value) / range * 100}%`;
      bar.classList.toggle('negative', value < 0);
      row.querySelector('.value').textContent = value.toFixed(3);
      row.setAttribute('aria-label', `${data[i].dataset}, ${metrics[key]} ${value.toFixed(3)}, ${data[i].n} test clips`);
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === key)));
    document.getElementById('metric-label').textContent = metrics[key];
    document.getElementById('metric-status').textContent = `Showing ${metrics[key]} for the six native tasks.`;
  }
  buttons.forEach(button => button.addEventListener('click', () => showMetric(button.dataset.metric)));
  const toggle = document.getElementById('table-toggle');
  const table = document.getElementById('result-table');
  toggle.addEventListener('click', () => {
    const expanded = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.textContent = expanded ? 'Hide data table ↑' : 'View data table ↓';
    table.hidden = !expanded;
  });
  document.getElementById('copy-command').addEventListener('click', async event => {
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(document.getElementById('quickstart').textContent.trim());
      button.textContent = 'Copied';
    } catch (_) {
      button.textContent = 'Select to copy';
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('quickstart'));
      selection.removeAllRanges();
      selection.addRange(range);
    }
  });
  showMetric('macro_f1');
})();
