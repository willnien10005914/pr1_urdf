import { createViewer } from './viewer.js?v=3';

const $ = (id) => document.getElementById(id);
const viewer = createViewer($('viewer'));
let catalog = null;
let joints = [];
let motion = null;
let frame = 0;
let playing = false;
let activeJob = null;
let robotId = 'rp1_24dof';
let armAutoplay = false;
const seenComplete = new Set();
const playedJobs = new Set();

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function say(message) { $('status').textContent = message; }

function renderJoints() {
  const query = $('joint-search').value.trim().toLowerCase();
  const list = $('joints');
  list.replaceChildren();
  const shown = joints.filter((joint) => joint.name.toLowerCase().includes(query));
  $('joint-count').textContent = `${shown.length} / ${joints.length}`;
  for (const joint of shown) {
    const row = document.createElement('label');
    row.className = 'joint';
    const name = document.createElement('b');
    name.textContent = joint.name;
    const readout = document.createElement('small');
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(joint.lower);
    slider.max = String(joint.upper);
    slider.step = '0.001';
    const live = motion ? motion.joint_names.indexOf(joint.name) : -1;
    if (live >= 0) joint.value = motion.q[frame][live];
    slider.value = String(joint.value || 0);
    const paint = () => { readout.textContent = `${Number(slider.value).toFixed(3)} rad`; };
    paint();
    slider.addEventListener('input', () => {
      playing = false;
      viewer.setJoints({ [joint.name]: Number(slider.value) });
      joint.value = Number(slider.value);
      paint();
      if (selectedStep) correctEditing();
    });
    row.append(name, readout, slider);
    list.append(row);
  }
}

async function loadRobot(id) {
  robotId = id;
  const robot = catalog.robots.find((item) => item.id === id);
  document.querySelectorAll('#models button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.robot === id));
  });
  joints = await viewer.load(robot.urdf_url);
  playing = false;
  renderJoints();
  if (motion) viewer.showFrame(motion, frame);
}

function showMotion(next) {
  motion = next;
  frame = 0;
  playing = true;
  $('scrub').disabled = false;
  $('scrub').max = String(Math.max(0, next.frames - 1));
  $('scrub').value = '0';
  $('frame-label').textContent = `${next.title} · 0 / ${next.frames}`;
  if (robotId !== 'rp1_24dof') loadRobot('rp1_24dof');
  else viewer.showFrame(motion, 0);
}

function tick() {
  if (playing && motion) {
    frame = (frame + 1) % motion.frames;
    viewer.showFrame(motion, frame);
    $('scrub').value = String(frame);
    $('frame-label').textContent = `${motion.title} · ${frame + 1} / ${motion.frames}`;
    if (frame % 6 === 0) syncSliders();
  }
  setTimeout(tick, motion ? 1000 / motion.fps : 200);
}

function syncSliders() {
  if (!motion) return;
  const values = Object.fromEntries(motion.joint_names.map((name, index) => [name, motion.q[frame][index]]));
  for (const input of document.querySelectorAll('#joints input')) {
    const name = input.parentElement.querySelector('b').textContent;
    if (name in values && document.activeElement !== input) {
      input.value = String(values[name]);
      input.parentElement.querySelector('small').textContent = `${Number(values[name]).toFixed(3)} rad`;
    }
  }
}

function fillTask(source) {
  for (const key of ['num_envs', 'max_iterations', 'learning_rate', 'entropy_coef', 'num_steps_per_env', 'clip_param', 'gamma', 'seed']) {
    $(key).value = source[key];
  }
}

function currentTask() {
  return catalog.tasks.find((task) => task.id === $('task').value);
}

async function playRecorded(job) {
  playedJobs.add(job.id);
  showMotion(await api(`/api/jobs/${job.id}/motion`));
  say(`Playing ${job.title}`);
}

function paintJob(job) {
  if (!job) return;
  activeJob = job;
  $('job-stage').textContent = job.stage_label || job.state;
  $('job-eta').textContent = job.eta || '—';
  const metrics = job.metrics || {};
  $('job-iter').textContent = metrics.iteration ? `${metrics.iteration}/${metrics.iteration_total || job.config.max_iterations || ''}` : '—';
  $('job-reward').textContent = metrics.mean_reward === undefined ? '—' : Number(metrics.mean_reward).toFixed(3);
  $('job-progress').value = job.percent || 0;
  if (job.state === 'complete' && job.has_motion && job.kind === 'train') {
    $('job-log').textContent = `${job.title} 可以播放。檢視器正在播 Isaac Lab 裡訓好的 PPO 策略。`;
  } else if (job.state === 'complete' && job.has_motion) {
    $('job-log').textContent = `${job.title} 可以在檢視器播放。這是重定向後的關節動作，不是 PPO 策略。`;
  } else {
    $('job-log').textContent = job.log_tail || '';
    if (job.error) say(job.error);
  }
}

async function refreshJobs() {
  const { jobs } = await api('/api/jobs');
  const list = $('job-list');
  list.replaceChildren();
  for (const job of jobs.slice(0, 8)) {
    const button = document.createElement('button');
    button.className = 'ghost';
    button.type = 'button';
    button.textContent = `${job.title} · ${job.state}`;
    button.addEventListener('click', async () => {
      paintJob(job);
      if (job.state === 'complete' && job.has_motion) {
        await playRecorded(job);
      } else if (job.kind === 'train' && job.state === 'complete') {
        say('訓練已完成，正在把策略錄成上面可以播的動作。');
      }
    });
    list.append(button);
  }
  const clipSelect = $('track-clip');
  const chosen = clipSelect.value;
  const clips = jobs.filter((job) => job.kind === 'retarget' && job.state === 'complete' && job.has_motion);
  clipSelect.replaceChildren();
  for (const job of clips) {
    const option = document.createElement('option');
    option.value = job.id;
    option.textContent = job.title;
    clipSelect.append(option);
  }
  if (clips.some((job) => job.id === chosen)) clipSelect.value = chosen;
  const live = jobs.find((job) => ['queued', 'download', 'pose', 'retarget', 'starting_isaac', 'training'].includes(job.state));
  if (live) paintJob(live);
  else if (activeJob) {
    const fresh = jobs.find((job) => job.id === activeJob.id);
    if (fresh) {
      paintJob(fresh);
      if (fresh.kind === 'train' && fresh.state === 'complete' && fresh.has_motion && !playedJobs.has(fresh.id)) {
        await playRecorded(fresh);
      }
    }
  }
  showLearning(live && live.kind === 'train' ? live : null);
  if (!armAutoplay) {
    for (const job of jobs) if (job.state === 'complete') seenComplete.add(job.id);
    armAutoplay = true;
    return;
  }
  for (const job of jobs) {
    if (job.state === 'complete' && job.has_motion && !seenComplete.has(job.id)) {
      seenComplete.add(job.id);
      showMotion(await api(`/api/jobs/${job.id}/motion`));
      say(`Playing ${job.title}`);
    }
  }
}

async function boot() {
  catalog = await api('/api/catalog');
  const caps = await api('/api/capabilities');
  const missing = [];
  if (!caps.hhtools) missing.push('HHTools venv');
  if (!caps.smplx) missing.push('SMPL-X weights');
  if (!caps.train_python) missing.push('roboparty_train environment');
  if (!caps.gpu) missing.push('NVIDIA GPU');
  if (missing.length) {
    $('banner').hidden = false;
    $('banner').textContent = `檢視器和參考片段可以用。這些還沒備好，不能開新的訓練：${missing.join('、')}。`;
  }
  const models = $('models');
  for (const robot of catalog.robots) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.robot = robot.id;
    button.textContent = `${robot.title} · ${robot.joints}`;
    button.addEventListener('click', () => loadRobot(robot.id).catch((error) => say(error.message)));
    models.append(button);
  }
  const cards = $('motion-cards');
  const index = await api('/api/motions');
  for (const item of index.motions) {
    const button = document.createElement('button');
    button.className = 'card';
    button.type = 'button';
    button.innerHTML = `<strong>${item.title}</strong><em>${item.task}</em><p>${item.summary}</p>`;
    button.addEventListener('click', async () => {
      const clip = await api(`/api/motions/${item.id}`);
      showMotion(clip);
    });
    cards.append(button);
  }
  const flip = document.createElement('article');
  flip.className = 'card';
  flip.innerHTML = `<strong>${catalog.flip.title}</strong><em>不在公開資料包</em><p>${catalog.flip.summary}</p>`;
  cards.append(flip);
  const task = $('task');
  for (const item of catalog.tasks) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${item.title} · ${item.id}`;
    task.append(option);
  }
  const describe = () => {
    const selected = currentTask();
    const tracking = selected.id === 'RP1-BeyondMimic' || selected.id === 'RP1-Ballet';
    $('track-clip-label').hidden = !tracking;
    const clipNote = tracking
      ? ' 開始訓練會追上面選的片段。通道關掉之後，這次訓練仍在這台電腦上繼續。'
      : ' 這個任務在學行走，不是在追 YouTube 片段。';
    $('task-note').textContent = `${selected.learns} 公開預算：${selected.defaults.max_iterations} 次迭代、${selected.defaults.num_envs} 個環境、學習率 ${selected.defaults.learning_rate}。${clipNote}`;
  };
  task.addEventListener('change', () => { fillTask(currentTask().practice); describe(); });
  fillTask(currentTask().practice);
  describe();
  $('use-practice').addEventListener('click', () => fillTask(currentTask().practice));
  $('use-published').addEventListener('click', () => fillTask(currentTask().defaults));
  for (const step of catalog.stack) {
    const article = document.createElement('article');
    article.innerHTML = `<strong>0${step.step}</strong><h3>${step.title}</h3><p>${step.body}</p>`;
    $('stack-steps').append(article);
  }
  await loadRobot('rp1_24dof');
  await loadSteps();
  tick();
  setInterval(() => { refreshJobs().catch(() => {}); }, 2000);
  refreshJobs().catch(() => {});
}

$('joint-search').addEventListener('input', renderJoints);

let ballet = null;
let selectedStep = null;

async function loadSteps() {
  ballet = await api('/api/ballet-steps');
  const box = $('step-buttons');
  box.replaceChildren();
  for (const step of ballet.steps) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.step = step.id;
    button.textContent = step.title;
    button.classList.toggle('saved', Boolean(step.saved));
    button.addEventListener('click', () => selectStep(step.id));
    box.append(button);
  }
}

function selectStep(id) {
  const step = ballet.steps.find((item) => item.id === id);
  selectedStep = id;
  playing = false;
  motion = null;
  $('scrub').disabled = true;
  $('frame-label').textContent = step.title;
  for (const button of $('step-buttons').querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.step === id));
  }
  $('com-x').value = String(step.com?.x || 0);
  $('com-y').value = String(step.com?.y || 0);
  viewer.setJoints(step.joints || {});
  const live = Object.fromEntries(viewer.jointStates().map((joint) => [joint.name, joint.value]));
  for (const joint of joints) {
    if (joint.name in live) joint.value = live[joint.name];
  }
  renderJoints();
  correctEditing();
  $('step-note').textContent = step.saved
    ? `${step.note} 這一格已儲存。再拖關節的話，要再按一次儲存。`
    : `${step.note} 這是目前的預設，拖完關節後按儲存。`;
}

function currentCom() {
  return { x: Number($('com-x').value) || 0, y: Number($('com-y').value) || 0 };
}

function correctEditing() {
  const stance = viewer.correctStance(currentCom());
  const label = { left: '左腳著地', right: '右腳著地', both: '雙腳著地' }[stance.support] || '著地校正';
  $('support-note').textContent = `${label}。重心可再前後、左右微調。`;
}

for (const id of ['com-x', 'com-y']) {
  $(id).addEventListener('input', () => {
    if (!selectedStep) {
      say('先按一個步驟，再調重心。');
      return;
    }
    playing = false;
    correctEditing();
  });
}

async function storeStep(step, values, com) {
  const saved = await api(`/api/ballet-steps/${step.id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ joints: values, com }),
  });
  step.joints = saved.joints;
  step.com = saved.com;
  step.saved = true;
  $('step-buttons').querySelector(`[data-step="${step.id}"]`).classList.add('saved');
  return saved;
}

$('step-save').addEventListener('click', async () => {
  if (!selectedStep) {
    say('先按一個步驟。');
    return;
  }
  const index = ballet.steps.findIndex((item) => item.id === selectedStep);
  const step = ballet.steps[index];
  const next = ballet.steps[index + 1];
  const copy = next ? window.confirm(`要把現在的姿勢複製到「${next.title}」嗎？`) : false;
  const values = Object.fromEntries(viewer.jointStates().map((joint) => [joint.name, joint.value]));
  const com = currentCom();
  try {
    correctEditing();
    await storeStep(step, values, com);
    if (copy) await storeStep(next, values, com);
    $('step-note').textContent = copy
      ? `${step.title} 已儲存，並複製到${next.title}。`
      : `${step.title} 已儲存。重播會從始站依序播到收尾。`;
    say($('step-note').textContent);
  } catch (error) {
    say(error.message);
  }
});

$('step-play').addEventListener('click', async () => {
  try {
    const clip = await api('/api/ballet-steps/replay');
    selectedStep = null;
    for (const button of $('step-buttons').querySelectorAll('button')) button.setAttribute('aria-pressed', 'false');
    showMotion(clip);
    say('從始站依序播到收尾。沒按過儲存的步驟用目前的預設。');
  } catch (error) {
    say(error.message);
  }
});
$('play').addEventListener('click', () => { if (motion) playing = !playing; });
$('stop').addEventListener('click', () => { playing = false; viewer.resetPose(); joints = viewer.jointStates(); renderJoints(); });
$('scrub').addEventListener('input', () => {
  if (!motion) return;
  playing = false;
  frame = Number($('scrub').value);
  viewer.showFrame(motion, frame);
  syncSliders();
});

function previewYouTube() {
  const url = $('youtube-url').value.trim();
  const match = url.match(/(?:v=|youtu\.be\/|shorts\/)([A-Za-z0-9_-]{6,})/);
  const frame = $('youtube-preview');
  if (!match) { frame.hidden = true; return; }
  const start = Math.max(0, Math.floor(Number($('youtube-start').value) || 0));
  const end = Math.max(start + 1, Math.floor(Number($('youtube-end').value) || start + 1));
  frame.hidden = false;
  frame.src = `https://www.youtube-nocookie.com/embed/${match[1]}?start=${start}&end=${end}`;
}
$('youtube-url').addEventListener('change', previewYouTube);
$('youtube-start').addEventListener('change', previewYouTube);
$('youtube-end').addEventListener('change', previewYouTube);

$('youtube-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  say('Queued. The worker will download, estimate the pose, then retarget.');
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: 'retarget',
        title: $('youtube-title').value,
        youtube_url: $('youtube-url').value,
        start: Number($('youtube-start').value),
        end: Number($('youtube-end').value),
        reference: $('youtube-reference').value,
      }),
    });
    paintJob(job);
  } catch (error) {
    say(error.message);
  }
});

$('csv-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const result = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'csv', title: 'HHTools CSV', csv: $('csv-text').value }),
    });
    showMotion(result.motion);
    say(`Playing ${result.motion.frames} retargeted frames.`);
  } catch (error) {
    say(error.message);
  }
});

let shownPreview = '';

function learningCaption(meta) {
  return `第 ${meta.iteration} / ${meta.total} 次。橘、金是左右手要學的軌跡，藍、綠是左右腳。約每 50 次換一張。`;
}

async function showLearning(job) {
  const figures = [$('learn-monitor'), $('train-preview')];
  if (!job) {
    for (const figure of figures) figure.hidden = true;
    return;
  }
  try {
    const meta = await api(`/api/jobs/${job.id}/preview.json`);
    const key = `${job.id}:${meta.iteration}`;
    for (const figure of figures) {
      if (key !== shownPreview) {
        figure.querySelector('img').src = `/api/jobs/${job.id}/preview.gif?v=${meta.iteration}`;
      }
      figure.querySelector('figcaption').textContent = learningCaption(meta);
      figure.hidden = false;
    }
    shownPreview = key;
  } catch (error) {
    for (const figure of figures) {
      figure.hidden = false;
      figure.querySelector('figcaption').textContent = '訓練進行中，第一張畫面約在第 50 次迭代出現。';
    }
  }
}

$('watch-train').addEventListener('click', () => {
  $('learn-monitor').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

$('train-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = { task: $('task').value, physics: $('physics').value };
  for (const key of ['num_envs', 'max_iterations', 'learning_rate', 'entropy_coef', 'num_steps_per_env', 'clip_param', 'gamma', 'seed']) {
    config[key] = Number($(key).value);
  }
  if (config.task === 'RP1-BeyondMimic') config.clip_id = $('track-clip').value;
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'train', title: config.task, config }),
    });
    paintJob(job);
    say('Training is running on this computer. Closing this page or the tunnel does not stop it. Open the site again to see the same job.');
  } catch (error) {
    say(error.message);
  }
});

const studio = $('studio');
const workspace = document.querySelector('.workspace');
function studioBox() {
  return studio.getBoundingClientRect();
}
function pinStudio(rect = studioBox()) {
  studio.style.left = `${rect.left}px`;
  studio.style.top = `${rect.top}px`;
  studio.style.width = `${rect.width}px`;
  studio.style.height = `${rect.height}px`;
  studio.style.right = 'auto';
  studio.style.bottom = 'auto';
  studio.style.marginLeft = '0';
  studio.style.marginTop = '0';
}
function readShare() {
  const raw = workspace.style.getPropertyValue('--studio-share')
    || getComputedStyle(workspace).getPropertyValue('--studio-share');
  const value = parseFloat(raw);
  return Number.isFinite(value) ? value : 72;
}
function setStudioShare(percent) {
  const bounds = workspace.getBoundingClientRect().height || window.innerHeight || 1;
  const minPct = (96 / bounds) * 100;
  const maxPct = ((bounds - 28) / bounds) * 100;
  const next = Math.max(minPct, Math.min(maxPct, percent));
  workspace.style.setProperty('--studio-share', `${Math.round(next * 10) / 10}%`);
}
function releaseStudio() {
  studio.classList.remove('floating');
  workspace.classList.remove('studio-away');
  $('viewer-float').textContent = '浮動視窗';
  for (const name of ['left', 'top', 'right', 'bottom', 'width', 'height', 'marginLeft', 'marginTop']) {
    studio.style[name] = '';
  }
}
function isIphone() {
  return /iPhone|iPod/.test(navigator.userAgent || '');
}
let layoutChosen = localStorage.getItem('pr1-layout-chosen') === '1';
function preferredLayout() {
  if (isIphone()) return 'portrait';
  if (layoutChosen) return localStorage.getItem('pr1-layout') === 'portrait' ? 'portrait' : 'landscape';
  return window.matchMedia('(max-width: 800px)').matches ? 'portrait' : 'landscape';
}
function setIntro(open) {
  const intro = $('stage-intro');
  intro.hidden = !open;
  $('toggle-intro').textContent = open ? '收合說明' : '展開說明';
  $('toggle-intro').setAttribute('aria-expanded', String(open));
}
function applyLayout(mode, reset = false) {
  const portrait = mode === 'portrait';
  const entering = portrait && !workspace.classList.contains('portrait');
  if (!entering && !reset && workspace.classList.contains('portrait') === portrait) return;
  releaseStudio();
  workspace.classList.toggle('portrait', portrait);
  const button = $('viewer-layout');
  button.textContent = portrait ? '橫向' : '直向';
  button.title = portrait
    ? '改成橫向：URDF 在左，選項在右'
    : '改成直向：URDF 在上，Published references 在下';
  button.setAttribute('aria-pressed', String(portrait));
  if (portrait && !workspace.style.getPropertyValue('--studio-share')) setStudioShare(isIphone() ? 78 : 72);
  if (portrait && (entering || reset)) setIntro(false);
}
function nudgeStudio(_dw, dh) {
  if (workspace.classList.contains('portrait') && !studio.classList.contains('floating')) {
    setStudioShare(readShare() + (dh > 0 ? 6 : -6));
    return;
  }
  const box = studioBox();
  const limit = studio.classList.contains('floating')
    ? window.innerHeight - 16
    : workspace.getBoundingClientRect().height;
  const height = Math.max(280, Math.min(limit, box.height + dh));
  if (studio.classList.contains('floating')) {
    pinStudio(box);
    studio.style.height = `${height}px`;
    return;
  }
  studio.style.height = `${height}px`;
}
$('viewer-smaller').addEventListener('click', () => nudgeStudio(0, -80));
$('viewer-larger').addEventListener('click', () => nudgeStudio(0, 80));
$('viewer-layout').addEventListener('click', () => {
  layoutChosen = true;
  const next = workspace.classList.contains('portrait') ? 'landscape' : 'portrait';
  localStorage.setItem('pr1-layout-chosen', '1');
  localStorage.setItem('pr1-layout', next);
  applyLayout(next, true);
});
applyLayout(preferredLayout(), true);
window.matchMedia('(max-width: 800px)').addEventListener('change', () => {
  if (isIphone() || layoutChosen) return;
  applyLayout(preferredLayout(), true);
});
$('toggle-intro').addEventListener('click', () => setIntro($('stage-intro').hidden));
$('viewer-float').addEventListener('click', () => {
  const floating = studio.classList.toggle('floating');
  workspace.classList.toggle('studio-away', floating);
  $('viewer-float').textContent = floating ? '固定回去' : '浮動視窗';
  if (floating) {
    const rect = studioBox();
    const width = Math.min(rect.width, window.innerWidth - 24);
    const height = Math.min(rect.height, window.innerHeight - 24);
    pinStudio({
      left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
      top: Math.max(8, Math.min(rect.top, window.innerHeight - height - 8)),
      width,
      height,
    });
  } else {
    for (const name of ['left', 'top', 'right', 'bottom', 'width', 'height', 'marginLeft', 'marginTop']) {
      studio.style[name] = '';
    }
  }
});
function dragStudio(event) {
  if (!studio.classList.contains('floating') || event.button !== 0 || event.target.closest('button')) return;
  event.preventDefault();
  const origin = studioBox();
  const offsetX = event.clientX - origin.left;
  const offsetY = event.clientY - origin.top;
  pinStudio(origin);
  const move = (moveEvent) => {
    const width = studioBox().width;
    const height = studioBox().height;
    const left = Math.max(0, Math.min(window.innerWidth - Math.min(width, 120), moveEvent.clientX - offsetX));
    const top = Math.max(0, Math.min(window.innerHeight - 48, moveEvent.clientY - offsetY));
    studio.style.left = `${left}px`;
    studio.style.top = `${top}px`;
    void height;
  };
  const stop = () => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', stop);
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', stop);
}
studio.querySelector('.studio-bar').addEventListener('pointerdown', dragStudio);
function dragPointer(target, setup) {
  target.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    try { target.setPointerCapture(event.pointerId); } catch (error) { /* the window listeners still follow the pointer */ }
    const startX = event.clientX;
    const startY = event.clientY;
    const apply = setup();
    const move = (moveEvent) => apply(moveEvent.clientX - startX, moveEvent.clientY - startY);
    const stop = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
  });
}
for (const handle of studio.querySelectorAll('.studio-edge')) {
  dragPointer(handle, () => {
    const edge = handle.dataset.edge;
    const box = studioBox();
    const floating = studio.classList.contains('floating');
    if (floating && edge === 'n') {
      pinStudio(box);
      return (dx, dy) => {
        studio.style.left = `${Math.max(0, box.left + dx)}px`;
        studio.style.top = `${Math.max(0, box.top + dy)}px`;
      };
    }
    if (floating) pinStudio(box);
    if (!floating && workspace.classList.contains('portrait')) {
      if (edge === 's' || edge === 'se' || edge === 'sw') {
        const bounds = workspace.getBoundingClientRect();
        const start = readShare();
        return (_dx, dy) => setStudioShare(start + (dy / bounds.height) * 100);
      }
      return () => {};
    }
    return (dx, dy) => {
      let width = box.width;
      let height = box.height;
      let left = box.left;
      let top = box.top;
      if (edge.includes('e')) width = box.width + dx;
      if (edge.includes('s')) height = box.height + dy;
      if (edge.includes('w')) width = box.width - dx;
      if (edge.includes('n')) height = box.height - dy;
      width = Math.max(320, Math.min(window.innerWidth - 12, width));
      height = Math.max(280, Math.min(window.innerHeight - 12, height));
      if (edge.includes('w')) left = box.left + (box.width - width);
      if (edge.includes('n')) top = box.top + (box.height - height);
      if (floating) {
        studio.style.width = `${width}px`;
        studio.style.height = `${height}px`;
        studio.style.left = `${Math.max(0, left)}px`;
        studio.style.top = `${Math.max(0, top)}px`;
        return;
      }
      if (edge.includes('e') || edge.includes('w')) {
        workspace.style.setProperty('--pane', `${width}px`);
      }
      if (edge.includes('s') || edge.includes('n')) {
        const limit = workspace.getBoundingClientRect().height;
        studio.style.height = `${Math.max(280, Math.min(limit, height))}px`;
      }
    };
  });
}
dragPointer($('portrait-split'), () => {
  const bounds = workspace.getBoundingClientRect();
  const start = readShare();
  return (_dx, dy) => setStudioShare(start + (dy / bounds.height) * 100);
});
dragPointer($('studio-split'), () => {
  const stage = studio.querySelector('.stage').getBoundingClientRect().width;
  const panel = studio.querySelector('.panel').getBoundingClientRect().width;
  return (dx) => {
    const next = Math.max(280, Math.min(stage + panel - 220, stage + dx));
    studio.style.setProperty('--stage-col', `${next}px`);
    studio.style.setProperty('--panel-col', `${stage + panel - next}px`);
  };
});

boot().catch((error) => { say(error.message); console.error(error); });
