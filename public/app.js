import { createViewer } from './viewer.js?v=2';

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

function paintJob(job) {
  if (!job) return;
  activeJob = job;
  $('job-stage').textContent = job.stage_label || job.state;
  $('job-eta').textContent = job.eta || '—';
  const metrics = job.metrics || {};
  $('job-iter').textContent = metrics.iteration ? `${metrics.iteration}/${metrics.iteration_total || job.config.max_iterations || ''}` : '—';
  $('job-reward').textContent = metrics.mean_reward === undefined ? '—' : Number(metrics.mean_reward).toFixed(3);
  $('job-progress').value = job.percent || 0;
  $('job-log').textContent = job.log_tail || '';
  if (job.error) say(job.error);
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
    button.addEventListener('click', () => paintJob(job));
    list.append(button);
  }
  const live = jobs.find((job) => ['queued', 'download', 'pose', 'retarget', 'starting_isaac', 'training'].includes(job.state));
  if (live) paintJob(live);
  else if (activeJob) {
    const fresh = jobs.find((job) => job.id === activeJob.id);
    if (fresh) paintJob(fresh);
  }
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
    $('banner').textContent = `Viewer and reference clips work. These are not ready for a new run: ${missing.join(', ')}.`;
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
  flip.innerHTML = `<strong>${catalog.flip.title}</strong><em>not in the open pack</em><p>${catalog.flip.summary}</p>`;
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
    $('task-note').textContent = `${selected.learns} Published run: ${selected.defaults.max_iterations} iterations, ${selected.defaults.num_envs} envs, lr ${selected.defaults.learning_rate}.`;
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
  tick();
  setInterval(() => { refreshJobs().catch(() => {}); }, 2000);
  refreshJobs().catch(() => {});
}

$('joint-search').addEventListener('input', renderJoints);
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

$('train-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = { task: $('task').value, physics: $('physics').value };
  for (const key of ['num_envs', 'max_iterations', 'learning_rate', 'entropy_coef', 'num_steps_per_env', 'clip_param', 'gamma', 'seed']) {
    config[key] = Number($(key).value);
  }
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'train', title: config.task, config }),
    });
    paintJob(job);
    say('Training queued. ETA stays blank until the first finished iteration.');
  } catch (error) {
    say(error.message);
  }
});

boot().catch((error) => { say(error.message); console.error(error); });
