const records = [...document.querySelectorAll('.record')];
const search = document.querySelector('#search');
const filter = document.querySelector('#filter');
const expand = document.querySelector('#expand');
function applyFilter() {
  let visible = 0;
  const query = search.value.toLowerCase();
  for (const record of records) {
    record.hidden = !(record.dataset.id.includes(query) &&
      (filter.value === 'all' || record.dataset.status === filter.value));
    if (!record.hidden) visible++;
  }
  document.querySelector('#no-results').hidden = visible > 0;
}
search.addEventListener('input', applyFilter);
filter.addEventListener('change', applyFilter);
expand.addEventListener('click', () => {
  const visible = records.filter(record => !record.hidden);
  const open = visible.some(record => !record.open);
  for (const record of visible) record.open = open;
  expand.textContent = open ? 'Collapse all' : 'Expand all';
});
