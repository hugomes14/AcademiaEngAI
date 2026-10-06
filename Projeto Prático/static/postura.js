(() => {
  const form = document.getElementById('postura-form');
  if (!form) return;
  const get = id => document.getElementById(`postura-${id}`);
  let timer, statusUrl, polling = false;
  const stages = {queued: 'Em espera', pose: 'Pontos do corpo', classification: 'Classificação', encoding: 'A preparar vídeo', completed: 'Concluído', failed: 'Não concluído'};
  const error = message => { get('error').textContent = message; get('error').hidden = false; };
  function render(state) {
    const done = state.status === 'completed', failed = state.status === 'failed';
    get('title').textContent = state.name;
    get('stage').textContent = stages[state.stage] || 'A analisar';
    get('message').textContent = state.message;
    get('progress-panel').hidden = false;
    get('progress').value = state.percent;
    get('percent').textContent = `${state.percent}%`;
    get('submit').disabled = !done && !failed;
    get('submit').textContent = done || failed ? 'Analisar outro vídeo →' : 'Análise em curso…';
    get('empty').hidden = true;
    get('result-panel').hidden = !done;
    const progress = state.progress;
    get('frame-count').textContent = done ? `${state.result.frames} frames analisados` :
      progress ? `${progress.frames} ${progress.total_frames > 0 ? `de ${progress.total_frames} ` : ''}frames · ${progress.video_seconds.toFixed(1)} s de vídeo` : state.message;
    get('preview-panel').hidden = done || !state.has_preview;
    if (!done && state.has_preview) get('preview').src = `${state.preview_url}?t=${Date.now()}`;
    if (done) {
      const video = get('result');
      if (video.getAttribute('src') !== state.video_url) { video.src = state.video_url; video.load(); }
      get('download').href = state.video_url;
      const r = state.result;
      const metrics = [
        ['Frames com pessoa', `${r.frames_with_people} / ${r.frames}`],
        ['Previsão disponível', `${Math.round(100*r.frames_with_prediction/r.frames)}% dos frames`],
        ['Previsão: boa postura', `${r.predicted_frames.boa_postura} frames`],
        ['Previsão: má postura', `${r.predicted_frames.ma_postura} frames`],
        ['Previsão inconclusiva', `${r.inconclusive_frames} frames`]
      ];
      get('metrics').replaceChildren(...metrics.map(([label, value]) => {
        const card = document.createElement('div'), title = document.createElement('small'), number = document.createElement('strong');
        title.textContent = label; number.textContent = value; card.append(title, number); return card;
      }));
    }
    if (failed) error(state.message);
    return !done && !failed;
  }
  async function poll() {
    if (polling) return;
    polling = true;
    let retry = false;
    try {
      const response = await fetch(statusUrl, {cache: 'no-store'}), state = await response.json();
      if (!response.ok) { error(state.erro || 'Não foi possível consultar a análise.'); get('submit').disabled = false; return; }
      get('error').hidden = true;
      retry = render(state);
    } catch (_) { error('Ligação interrompida. A tentar acompanhar a análise novamente…'); retry = true; }
    finally { polling = false; if (retry) timer = setTimeout(poll, 1200); }
  }
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    clearTimeout(timer);
    get('error').hidden = true;
    get('submit').disabled = true;
    get('submit').textContent = 'A enviar vídeo…';
    const xhr = new XMLHttpRequest();
    xhr.open('POST', form.action);
    xhr.responseType = 'json';
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) get('submit').textContent = `A enviar vídeo… ${Math.round(event.loaded/event.total*100)}%`;
    };
    xhr.onload = () => {
      if (xhr.status !== 202) {
        error(xhr.response?.erro || 'Não foi possível enviar o vídeo.');
        get('submit').disabled = false; get('submit').textContent = 'Analisar vídeo →'; return;
      }
      const state = xhr.response;
      statusUrl = state.status_url;
      history.replaceState(null, '', `?job=${state.id}`);
      get('result').removeAttribute('src'); get('result').load();
      if (render(state)) poll();
    };
    xhr.onerror = () => { error('Não foi possível enviar o vídeo. Verifica a ligação.'); get('submit').disabled = false; get('submit').textContent = 'Analisar vídeo →'; };
    xhr.send(new FormData(form));
  });
  const jobId = new URLSearchParams(location.search).get('job');
  if (jobId && /^[a-f0-9]{32}$/.test(jobId)) { statusUrl = `/api/postura/jobs/${jobId}`; poll(); }
})();
