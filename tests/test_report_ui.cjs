const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup() {
  const element = extra => ({hidden: false, value: '', open: false, events: {},
    addEventListener(name, listener) { this.events[name] = listener; }, ...extra});
  const records = [
    element({dataset: {id: 'sample-a', status: 'changed'}}),
    element({dataset: {id: 'sample-b', status: 'unchanged'}}),
    element({dataset: {id: 'sample-c', status: 'added'}}),
  ];
  const controls = {'#search': element(), '#filter': element({value: 'all'}),
    '#expand': element(), '#no-results': element({hidden: true})};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../sftdiff/report.js'), 'utf8'), {
    document: {querySelectorAll: () => records, querySelector: id => controls[id]},
  });
  return {records, controls};
}

test('search and status filters combine without changing record contents', () => {
  const {records, controls} = setup();
  controls['#search'].value = 'SAMPLE-B';
  controls['#search'].events.input();
  assert.deepEqual(records.map(r => r.hidden), [true, false, true]);
  controls['#filter'].value = 'changed';
  controls['#filter'].events.change();
  assert.ok(records.every(r => r.hidden));
  assert.equal(controls['#no-results'].hidden, false);
  controls['#search'].value = '';
  controls['#search'].events.input();
  assert.deepEqual(records.map(r => r.hidden), [false, true, true]);
  assert.equal(controls['#no-results'].hidden, true);
});

test('expand and collapse operate only on visible records', () => {
  const {records, controls} = setup();
  controls['#filter'].value = 'changed';
  controls['#filter'].events.change();
  controls['#expand'].events.click();
  assert.deepEqual(records.map(r => r.open), [true, false, false]);
  assert.equal(controls['#expand'].textContent, 'Collapse all');
  controls['#expand'].events.click();
  assert.ok(records.every(r => !r.open));
  assert.equal(controls['#expand'].textContent, 'Expand all');
});
