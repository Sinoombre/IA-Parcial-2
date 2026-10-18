const form = document.querySelector('#predict-form');
const reset = document.querySelector('#reset');
const emptyState = document.querySelector('#empty-state');
const result = document.querySelector('#result');
const state = document.querySelector('#result-state');
const seedForm = document.querySelector('#seed-form');
const sampleArea = document.querySelector('#sample-area');
const sampleGrid = document.querySelector('#sample-grid');
const seedError = document.querySelector('#seed-error');
const groupResults = document.querySelector('#group-results');

function setFormValues(values) {
  for (const [key, value] of Object.entries(values)) {
    const field = form.elements.namedItem(key);
    if (field) field.value = value;
  }
  document.querySelector('#profile-form').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderSamples(data) {
  sampleGrid.replaceChildren();
  document.querySelector('#seed-tag').textContent = `SEMILLA ${data.seed}`;
  document.querySelector('#sample-count').textContent =
    `El resumen y la gráfica incluyen las ${data.count} personas. Las tarjetas siguientes solo cargan una persona en el formulario individual.`;
  for (const profile of data.profiles.slice(0, 3)) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sample-card';
    button.setAttribute('aria-label', `Cargar ${profile.name}`);

    const title = document.createElement('span');
    title.className = 'sample-card-index';
    title.textContent = profile.name;
    const contract = document.createElement('strong');
    contract.textContent = profile.values.Contract;
    const details = document.createElement('small');
    details.textContent = `${profile.values['Tenure Months']} meses · ${profile.values['Internet Service']}`;
    const action = document.createElement('span');
    action.className = 'sample-card-action';
    action.textContent = 'Probar individualmente ↗';

    button.append(title, contract, details, action);
    button.addEventListener('click', () => setFormValues(profile.values));
    sampleGrid.append(button);
  }
  sampleArea.hidden = false;
}

function renderGroup(data) {
  document.querySelector('#group-count').textContent = String(data.count);
  document.querySelector('#group-average').textContent =
    `${(data.mean_p_churn * 100).toFixed(1)}%`;
  document.querySelector('#group-high-risk').textContent =
    `${data.high_risk_count} de ${data.count}`;
  document.querySelector('#group-utility').textContent =
    data.mean_utility.toFixed(3);
  document.querySelector('#group-disclaimer').textContent = data.disclaimer;

  const bins = Array.from({ length: 5 }, (_, index) => ({
    label: `${index * 20}–${(index + 1) * 20}%`,
    count: 0,
  }));
  for (const prediction of data.predictions) {
    const bin = Math.min(4, Math.floor(prediction.p_churn * 5));
    bins[bin].count += 1;
  }
  const chart = document.querySelector('#probability-chart');
  chart.replaceChildren();
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  for (const bin of bins) {
    const item = document.createElement('div');
    item.className = 'chart-bin';
    const barTrack = document.createElement('div');
    barTrack.className = 'chart-bar-track';
    const bar = document.createElement('span');
    bar.className = 'chart-bar';
    bar.style.height = `${Math.max(3, (bin.count / maxCount) * 100)}%`;
    barTrack.append(bar);
    const count = document.createElement('b');
    count.textContent = String(bin.count);
    const label = document.createElement('small');
    label.textContent = bin.label;
    item.append(count, barTrack, label);
    chart.append(item);
  }

  const tableBody = document.querySelector('#group-predictions');
  tableBody.replaceChildren();
  for (const prediction of data.predictions) {
    const row = document.createElement('tr');
    for (const value of [
      prediction.name,
      `${(prediction.p_churn * 100).toFixed(1)}%`,
      prediction.u_act.toFixed(3),
      prediction.recommendation,
    ]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    tableBody.append(row);
  }
  groupResults.classList.remove('hidden');
  groupResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function predictGroup(profiles) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profiles }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'No se pudo calcular el grupo.');
  renderGroup(data);
}

function payloadFromForm() {
  const data = {};
  new FormData(form).forEach((value, key) => { data[key] = value; });
  return data;
}

function render(data) {
  const probability = Number(data.p_churn);
  document.querySelector('#p-churn').textContent = `${(probability * 100).toFixed(1)}%`;
  document.querySelector('#meter-fill').style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  document.querySelector('.meter').setAttribute('aria-valuenow', String(Math.round(probability * 100)));
  document.querySelector('#risk-label').textContent = probability >= .7 ? 'RIESGO ALTO' : probability >= .4 ? 'RIESGO MEDIO' : 'RIESGO BAJO';
  document.querySelector('#u-act').textContent = Number(data.u_act).toFixed(3);
  document.querySelector('#v-client').textContent = Number(data.v_cliente).toFixed(2);
  document.querySelector('#recommendation').textContent = data.recommendation;
  document.querySelector('#disclaimer').textContent = data.disclaimer;
  emptyState.classList.add('hidden');
  result.classList.remove('hidden');
  state.textContent = 'CALCULADO';
}

seedForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const seed = document.querySelector('#seed-input').value;
  const count = document.querySelector('#group-size').value;
  const button = seedForm.querySelector('button');
  button.disabled = true;
  seedError.textContent = '';
  button.firstChild.textContent = 'Generando y calculando… ';
  try {
    const response = await fetch(
      `/api/sample?seed=${encodeURIComponent(seed)}&count=${encodeURIComponent(count)}`,
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudieron crear los perfiles.');
    renderSamples(data);
    await predictGroup(data.profiles);
  } catch (error) {
    seedError.textContent = error.message;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = 'Generar y calcular grupo ';
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  state.textContent = 'CALCULANDO';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payloadFromForm()) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudo calcular la predicción');
    render(data);
  } catch (error) {
    state.textContent = 'ERROR';
    emptyState.classList.remove('hidden');
    result.classList.add('hidden');
    emptyState.querySelector('p').textContent = error.message;
  } finally {
    state.textContent = state.textContent === 'CALCULANDO' ? 'EN ESPERA' : state.textContent;
  }
});

reset.addEventListener('click', () => {
  form.reset();
  state.textContent = 'EN ESPERA';
  result.classList.add('hidden');
  emptyState.classList.remove('hidden');
  emptyState.querySelector('p').textContent = 'Tu siguiente prueba empieza aquí.';
});
