// Method demo: clickable overview figure + 2D point-robot animation of the three pipeline steps.
//   1. reachability scoring of WGRs   2. diffusion denoising of path samples   3. tree expansion (Alg. 1)
// The scene is generated from a fixed seed so every visitor sees the same run.
(function () {
  'use strict';

  var root = document.getElementById('wgr-demo');
  if (!root) return;

  /* ================= scene parameters ================= */

  var SEED = 11;
  var START = { x: 45, y: 46 };
  var OBSTACLES = [{x:8,y:10,w:8,h:24},{x:30,y:10,w:8,h:24},{x:8,y:33,w:30,h:5},{x:64.5,y:32.4,w:29.5,h:8.1},{x:14.6,y:80.4,w:23.6,h:8.1},{x:58.7,y:62,w:31.8,h:5.5},{x:8.8,y:56.5,w:29.3,h:5.2},{x:66.9,y:82.3,w:23.6,h:6.3},{x:72.1,y:10.1,w:21.9,h:4.8},{x:64.5,y:8.5,w:6,h:16.7},{x:30.7,y:49.7,w:14.4,h:5.9},{x:8.8,y:64.6,w:6.4,h:12.7},{x:88,y:20.4,w:6,h:12},{x:66.9,y:69.4,w:4.5,h:12.9},{x:73.8,y:42.7,w:6.4,h:8.5},{x:8.8,y:61.7,w:37.9,h:1.3},{x:51.2,y:28.7,w:6.4,h:7.5},{x:53.1,y:48,w:16.6,h:2.8},{x:54,y:37.2,w:7.3,h:6.3},{x:32.3,y:62.9,w:14.4,h:3.1},{x:32.3,y:69,w:6.4,h:6.3},{x:53.1,y:51.3,w:25.8,h:1.5},{x:56.3,y:56,w:7.1,h:5},{x:32.3,y:75.3,w:5.8,h:5.2},{x:64.5,y:25.2,w:6.9,h:4.2},{x:58.7,y:67.5,w:12.7,h:1.8},{x:53.6,y:53.7,w:25.2,h:0.9},{x:51.2,y:65.2,w:2.6,h:8.7},{x:65.6,y:56,w:13.3,h:1.7},{x:53.1,y:45.8,w:9.5,h:2.2},{x:60.4,y:54.7,w:18.5,h:1.1},{x:82.1,y:68.5,w:2.4,h:8.1},{x:32.3,y:66.1,w:5.8,h:2.9},{x:53.5,y:22.8,w:8.4,h:2},{x:64.5,y:29.4,w:8.8,h:1.7},{x:47.5,y:66.6,w:3.6,h:3.9},{x:88,y:76.4,w:2.4,h:5.7},{x:73.5,y:29.4,w:6.2,h:2},{x:65.6,y:52.8,w:13.3,h:0.9},{x:65.6,y:57.6,w:9.2,h:1.3},{x:69.3,y:58.9,w:5.4,h:2.2},{x:8.8,y:62.9,w:6.5,h:1.7},{x:35.7,y:48.6,w:9.3,h:1.1},{x:35.7,y:55.6,w:9.3,h:0.9},{x:81.3,y:67.5,w:9.2,h:0.9},{x:64.5,y:31.1,w:6,h:1.3}];
  // Four WGRs boxed into the obstacle frames, with fixed (illustrative) reachability scores.
  var REGIONS = [
    { x: 15.3, y: 19.9, w: 16.1, h: 9.2,  score: 0.90, color: '#f472b6' },
    { x: 71.8, y: 19.5, w: 14.2, h: 9.9,  score: 0.30, color: '#fb923c' },
    { x: 15.0, y: 65.2, w: 16.4, h: 15.1, score: 0.60, color: '#eab308' },
    { x: 71.8, y: 68.5, w: 9.0,  h: 13.3, score: 0.05, color: '#38bdf8' }
  ];
  var TAU = 0.35;            // softmax temperature
  var ALPHA = 0.06;          // uniform mixing
  var TOTAL_GOALS = 36;      // goal configurations drawn across all regions
  var PATHS_PER_GOAL = 4;    // diffusion batch per goal
  var PATH_LEN = 12;         // waypoints per path sample
  var ROUTE_GAP = 7;         // min distance between two routes to the same goal to count as distinct
  var DDIM_STEPS = 50;
  var GOAL_BIAS = 0.6;
  var STEP = 4.5;            // tree extension step
  var DURATION = { 1: 7, 2: 6.5 };   // seconds per step (step 3 is paced by iterations)
  var HOLD = { 1: 1.4, 2: 1.4, 3: 4 }; // pause at the end of a step while touring

  /* ================= seeded RNG + geometry ================= */

  var rngState = 0;
  function seedRng(s) { rngState = s >>> 0; }
  function rand() { // mulberry32
    rngState = (rngState + 0x6D2B79F5) >>> 0;
    var t = rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function gauss() { return (rand() + rand() + rand() - 1.5) * 0.9; }
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function ease(x) { return x * x * x * (x * (x * 6 - 15) + 10); }

  function collision(p, env, m) {
    m = m == null ? 1.2 : m;
    if (p.x < m || p.x > 100 - m || p.y < m || p.y > 100 - m) return true;
    for (var i = 0; i < env.obs.length; i++) {
      var o = env.obs[i];
      if (p.x > o.x - m && p.x < o.x + o.w + m && p.y > o.y - m && p.y < o.y + o.h + m) return true;
    }
    return false;
  }
  function collSeg(a, b, env) {
    var n = Math.max(3, Math.ceil(dist(a, b) / 0.8));
    for (var i = 0; i <= n; i++) {
      var t = i / n;
      if (collision({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }, env)) return true;
    }
    return false;
  }
  function randFree(env) {
    for (var k = 0; k < 300; k++) {
      var p = { x: 3 + rand() * 94, y: 3 + rand() * 94 };
      if (!collision(p, env)) return p;
    }
    return { x: 50, y: 50 };
  }
  function inBox(p, r) { return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h; }

  /* ================= planners used to fake the learned samples ================= */

  function planRRT(env, goal, opt) {
    var from = opt.from || env.start;
    var nodes = [{ x: from.x, y: from.y, parent: -1 }], goalIndex = -1;
    for (var it = 0; it < opt.maxIter; it++) {
      var q = rand() < opt.goalBias ? goal : randFree(env);
      var ni = 0, nd = Infinity;
      for (var j = 0; j < nodes.length; j++) {
        var dx = nodes[j].x - q.x, dy = nodes[j].y - q.y, d = dx * dx + dy * dy;
        if (d < nd) { nd = d; ni = j; }
      }
      var nr = nodes[ni], ang = Math.atan2(q.y - nr.y, q.x - nr.x);
      var np = { x: nr.x + Math.cos(ang) * STEP, y: nr.y + Math.sin(ang) * STEP, parent: ni };
      if (collision(np, env) || collSeg(nr, np, env)) continue;
      nodes.push(np);
      if (dist(np, goal) < opt.goalDist && !collSeg(np, goal, env)) {
        nodes.push({ x: goal.x, y: goal.y, parent: nodes.length - 1 });
        goalIndex = nodes.length - 1;
        break;
      }
    }
    return { nodes: nodes, goalIndex: goalIndex };
  }
  function extract(nodes, idx) {
    var p = [];
    for (var i = idx; i >= 0; i = nodes[i].parent) p.push({ x: nodes[i].x, y: nodes[i].y });
    return p.reverse();
  }
  // Never fabricate a straight segment through obstacles: if no attempt reached the goal,
  // stop at the node nearest to it and only append the goal if that hop is collision-free.
  function pathTo(env, goal, from) {
    var best = null;
    for (var a = 0; a < 6; a++) {
      var r = planRRT(env, goal, { goalBias: 0.3, maxIter: 1400, goalDist: 4, from: from });
      if (r.goalIndex >= 0) return extract(r.nodes, r.goalIndex);
      if (!best) best = r;
    }
    var bi = 0, bd = Infinity;
    for (var i = 0; i < best.nodes.length; i++) {
      var d = dist(best.nodes[i], goal);
      if (d < bd) { bd = d; bi = i; }
    }
    var path = extract(best.nodes, bi);
    if (!collSeg(path[path.length - 1], goal, env)) path.push({ x: goal.x, y: goal.y });
    return path;
  }
  function shortcut(path, env) {
    path = path.slice();
    for (var k = 0; k < 140 && path.length >= 3; k++) {
      var i = Math.floor(rand() * (path.length - 2));
      var j = i + 2 + Math.floor(rand() * (path.length - i - 2));
      if (j >= path.length) continue;
      if (!collSeg(path[i], path[j], env)) path.splice(i + 1, j - i - 1);
    }
    return path;
  }
  function resample(path, n) {
    var dense = [path[0]];
    for (var i = 1; i < path.length; i++) {
      var a = path[i - 1], b = path[i], steps = Math.max(1, Math.ceil(dist(a, b)));
      for (var s = 1; s <= steps; s++) dense.push({ x: a.x + (b.x - a.x) * s / steps, y: a.y + (b.y - a.y) * s / steps });
    }
    var seg = [], tot = 0;
    for (i = 1; i < dense.length; i++) { var l = dist(dense[i - 1], dense[i]); seg.push(l); tot += l; }
    var out = [{ x: dense[0].x, y: dense[0].y }];
    for (var k = 1; k < n - 1; k++) {
      var target = tot * k / (n - 1), acc = 0, m = 0;
      while (m < seg.length && acc + seg[m] < target) { acc += seg[m]; m++; }
      var r = seg[m] ? (target - acc) / seg[m] : 0;
      var p0 = dense[m], p1 = dense[Math.min(m + 1, dense.length - 1)];
      out.push({ x: p0.x + (p1.x - p0.x) * r, y: p0.y + (p1.y - p0.y) * r });
    }
    out.push({ x: dense[dense.length - 1].x, y: dense[dense.length - 1].y });
    return out;
  }

  function pathLength(path) {
    var l = 0;
    for (var i = 1; i < path.length; i++) l += dist(path[i - 1], path[i]);
    return l;
  }
  // Discrete Hausdorff distance between two routes.
  function routeGap(a, b) {
    a = resample(a, 24); b = resample(b, 24);
    function far(p, q) {
      var m = 0;
      p.forEach(function (u) {
        var d = Infinity;
        q.forEach(function (v) { d = Math.min(d, dist(u, v)); });
        m = Math.max(m, d);
      });
      return m;
    }
    return Math.max(far(a, b), far(b, a));
  }
  // One short RRT attempt; null if it does not reach the target.
  function quickPath(env, goal, from) {
    var r = planRRT(env, goal, { goalBias: 0.3, maxIter: 500, goalDist: 4, from: from });
    return r.goalIndex >= 0 ? extract(r.nodes, r.goalIndex) : null;
  }
  // The diffusion model is multimodal: samples for the same goal follow different routes.
  // The first sample is the shortest route found; the others are forced through a random
  // via-point and kept only if they reach the goal, stay reasonably short, and differ from
  // every route already in the batch.
  function samplePaths(env, goal, n) {
    var routes = [shortcut(pathTo(env, goal), env)];
    var maxLen = 1.8 * pathLength(routes[0]) + 10;
    for (var t = 0; t < 40 && routes.length < n; t++) {
      var via = randFree(env);
      if (dist(env.start, via) + dist(via, goal) > maxLen) continue;   // detour too long anyway
      var a = quickPath(env, via, env.start), b = quickPath(env, goal, via);
      if (!a || !b) continue;
      var r = shortcut(a.concat(b.slice(1)), env);
      if (pathLength(r) > maxLen) continue;
      if (routes.some(function (q) { return routeGap(q, r) < ROUTE_GAP; })) continue;
      routes.push(r);
    }
    for (var i = 0; routes.length < n; i++) routes.push(routes[i]);   // too few distinct routes: repeat
    return routes;
  }

  /* ================= scene ================= */

  function buildScene() {
    seedRng(SEED);
    var env = { obs: OBSTACLES, start: START };
    var regions = REGIONS.map(function (r) { return Object.assign({}, r); });

    // Scores -> temperature softmax -> uniform mixing -> goal allocation.
    var maxS = Math.max.apply(null, regions.map(function (r) { return r.score; }));
    var exps = regions.map(function (r) { return Math.exp((r.score - maxS) / TAU); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    var maxP = 0;
    regions.forEach(function (r, i) {
      r.p = exps[i] / sum;
      r.pMix = (1 - ALPHA) * r.p + ALPHA / regions.length;
      r.nGoals = Math.max(1, Math.round(r.pMix * TOTAL_GOALS));
      maxP = Math.max(maxP, r.pMix);
    });

    var goals = [];
    regions.forEach(function (r, ri) {
      r.goals = [];
      for (var t = 0; t < 250 && r.goals.length < r.nGoals; t++) {
        var p = { x: r.x + rand() * r.w, y: r.y + rand() * r.h };
        if (!collision(p, env)) r.goals.push(p);
      }
      if (!r.goals.length) r.goals.push({ x: r.x + r.w / 2, y: r.y + r.h / 2 });
      r.goals.forEach(function (g) {
        g.region = ri;
        g.index = goals.length;
        g.weight0 = r.pMix / maxP;   // ComputeWeights: queue starts from the mixed probabilities
        goals.push(g);
      });
    });

    // "Diffusion" samples per goal: feasible paths, plus the noise they are denoised from.
    goals.forEach(function (g) {
      g.paths = []; g.noise = []; g.jit = []; g.cloud = [];
      samplePaths(env, g, PATHS_PER_GOAL).forEach(function (route) {
        var path = resample(route, PATH_LEN);
        g.paths.push(path);
        g.noise.push(path.map(function () { return { x: 8 + rand() * 84, y: 8 + rand() * 84 }; }));
        g.jit.push(path.map(function () { return { x: gauss() * 7, y: gauss() * 7 }; }));
        // bias toward the goal end so a goal-biased X step actually pulls toward this goal
        for (var i = Math.ceil(PATH_LEN / 2); i < PATH_LEN; i++) g.cloud.push(path[i]);
      });
    });

    return { env: env, regions: regions, goals: goals, tree: simulateTree(env, regions, goals) };
  }

  // Algorithm 1 on the 2D scene. Every iteration is recorded so it can be replayed / scrubbed.
  function simulateTree(env, regions, goals) {
    var weights = goals.map(function (g) { return g.weight0; });
    var nodes = [{ x: env.start.x, y: env.start.y, parent: -1 }];
    var iters = [], solvedNode = -1;

    function top() {
      var b = 0;
      for (var i = 1; i < weights.length; i++) if (weights[i] > weights[b]) b = i;
      return b;
    }
    function nearest(q) {
      var ni = 0, nd = Infinity;
      for (var j = 0; j < nodes.length; j++) {
        var dx = nodes[j].x - q.x, dy = nodes[j].y - q.y, d = dx * dx + dy * dy;
        if (d < nd) { nd = d; ni = j; }
      }
      return ni;
    }
    function extend(ni, target) {
      var nr = nodes[ni], d = dist(nr, target);
      if (d < 1e-6) return null;
      var s = Math.min(STEP, d);
      var p = { x: nr.x + (target.x - nr.x) / d * s, y: nr.y + (target.y - nr.y) / d * s };
      if (collision(p, env) || collSeg(nr, p, env)) return null;
      nodes.push({ x: p.x, y: p.y, parent: ni });
      return nodes.length - 1;
    }
    function solves(idx) {
      for (var r = 0; r < regions.length; r++) if (inBox(nodes[idx], regions[r])) return true;
      return false;
    }

    for (var it = 0; it < 900 && solvedNode < 0; it++) {
      var biased = rand() < GOAL_BIAS, gi = -1, target;
      if (biased) {
        gi = top();
        var cloud = goals[gi].cloud;
        target = cloud[(rand() * cloud.length) | 0];       // q^X from the diffusion samples
      } else {
        target = randFree(env);                           // GenRandomConf
      }
      var ni = nearest(target);
      var rec = { biased: biased, goal: gi, added: [], xOk: false, yOk: false };

      var xi = extend(ni, target);
      if (xi != null) {
        rec.xOk = true; rec.added.push(xi);
        if (solves(xi)) solvedNode = xi;
      }
      if (biased) {
        var yi = extend(ni, goals[gi]);                   // Y reuses the same q_near
        var w = weights[gi];
        if (yi != null) {
          rec.yOk = true; rec.added.push(yi);
          if (solvedNode < 0 && solves(yi)) solvedNode = yi;
          weights[gi] = Math.min(1, w < 1 ? w / (1 - w) : 1);   // Reward
        } else {
          weights[gi] = w / (w + 1);                            // Penalize
        }
      }
      rec.weights = weights.slice();
      rec.top = top();
      rec.nodeCount = nodes.length;
      iters.push(rec);
    }
    return {
      nodes: nodes, iters: iters, solvedNode: solvedNode,
      path: solvedNode >= 0 ? extract(nodes, solvedNode) : null
    };
  }

  /* ================= DOM ================= */

  var canvas = root.querySelector('.demo-canvas');
  var ctx = canvas.getContext('2d');
  var fig = root.querySelector('svg.ov');
  var tabs = [].slice.call(root.querySelectorAll('.demo-tabs [data-step]'));
  var clickables = [].slice.call(fig.querySelectorAll('.ov-click'));
  var heads = [].slice.call(root.querySelectorAll('.demo-head'));
  var algos = [].slice.call(root.querySelectorAll('.algo'));
  var playBtn = root.querySelector('.demo-play');
  var replayBtn = root.querySelector('.demo-replay');
  var backBtn = root.querySelector('.demo-back');
  var fwdBtn = root.querySelector('.demo-fwd');
  var scrub = root.querySelector('.demo-scrub');
  var readout = root.querySelector('.demo-readout');

  var scene = null;
  var size = 100, S = 1, k = 1;       // canvas px, px per world unit, visual scale
  var pal = {};
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var state = {
    step: 1,
    t: 0,            // steps 1–2: progress 0..1; step 3: iteration count
    playing: !reduceMotion,
    tour: true,      // auto-advance through the steps until the visitor picks one
    hold: 0,
    visible: false
  };
  if (reduceMotion) state.t = 1;

  function readPalette() {
    var cs = getComputedStyle(root);
    ['bg', 'obs', 'edge', 'path', 'region', 'start'].forEach(function (n) {
      pal[n] = cs.getPropertyValue('--demo-' + n).trim();
    });
    pal.font = getComputedStyle(document.body).fontFamily;
  }
  function resize() {
    var w = canvas.clientWidth;
    if (!w) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(w * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    size = w; S = w / 100; k = w / 500;
    draw();
  }

  /* ================= drawing ================= */

  function reachColor(s, a) { return 'hsla(' + Math.round(clamp01(s) * 120) + ',70%,45%,' + (a == null ? 1 : a) + ')'; }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function drawEnv() {
    ctx.fillStyle = pal.bg;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = pal.obs;
    OBSTACLES.forEach(function (o) {
      var w = o.w * S, h = o.h * S;
      roundRect(o.x * S, o.y * S, w, h, Math.min(3 * k, w / 3, h / 3));
      ctx.fill();
    });
  }
  function dot(p, r, color, alpha) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x * S, p.y * S, r, 0, 7);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  function square(p, r, color, alpha) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fillRect(p.x * S - r, p.y * S - r, r * 2, r * 2);
    ctx.globalAlpha = 1;
  }
  function polyline(pts, color, width, alpha) {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0].x * S, pts[0].y * S);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * S, pts[i].y * S);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  // a = box visibility, scoreA = how far the score is revealed (0..1)
  function drawRegion(r, a, scoreA) {
    if (a <= 0) return;
    var x = r.x * S, y = r.y * S, w = r.w * S, h = r.h * S, rad = Math.min(5 * k, w * 0.1, h * 0.1);
    roundRect(x, y, w, h, rad);
    ctx.fillStyle = scoreA > 0 ? reachColor(r.score, 0.22 * scoreA) : 'rgba(34,197,94,' + 0.08 * a + ')';
    ctx.fill();
    ctx.globalAlpha = a;
    ctx.strokeStyle = pal.region;
    ctx.lineWidth = 2 * k;
    ctx.stroke();
    if (scoreA > 0) {
      ctx.font = '600 ' + Math.max(10, 12.5 * k) + 'px "IBM Plex Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillStyle = reachColor(r.score);
      ctx.globalAlpha = scoreA;
      ctx.fillText((r.score * ease(scoreA)).toFixed(2), x + w - 3 * k, y + 3 * k);
    }
    ctx.globalAlpha = 1;
  }
  function drawStart() {
    var p = scene.env.start;
    ctx.save();
    ctx.shadowColor = pal.start;
    ctx.shadowBlur = 12 * k;
    dot(p, 6 * k, pal.start, 1);
    ctx.restore();
    dot(p, 2.2 * k, '#ffffff', 1);
  }

  function drawStep1(p) {
    var regs = scene.regions, n = regs.length;
    var boxA = clamp01(p / 0.15);
    regs.forEach(function (r, i) {
      var scoreA = clamp01(((p - 0.2) / 0.3) * n - i - 1e-6);
      drawRegion(r, boxA, scoreA);
      var shown = Math.round(r.goals.length * clamp01((p - 0.7) / 0.25));
      for (var g = 0; g < shown; g++) square(r.goals[g], 3 * k, r.color, 0.95);
    });
    drawStart();
  }

  // Path samples at denoising progress e (0 = noise, 1 = x_0). Step 3 redraws the finished
  // samples (e = 1) faded, so the tree is visibly guided by the same samples as step 2.
  function drawSamples(e, fade) {
    scene.goals.forEach(function (g) {
      var color = scene.regions[g.region].color;
      g.paths.forEach(function (path, pi) {
        var pts = path.map(function (tgt, i) {
          if (i === 0 || i === path.length - 1) return tgt;   // q_s and q_g are conditioning, not denoised
          var nz = g.noise[pi][i], jt = g.jit[pi][i];
          return { x: nz.x + (tgt.x - nz.x) * e + jt.x * (1 - e), y: nz.y + (tgt.y - nz.y) * e + jt.y * (1 - e) };
        });
        for (var i = 1; i < pts.length - 1; i++) dot(pts[i], 2.2 * k, color, fade * (0.35 + 0.55 * e));
      });
    });
  }

  function drawStep2(p) {
    var e = ease(clamp01((p - 0.12) / 0.8));
    scene.regions.forEach(function (r) { drawRegion(r, 1, 1); });
    drawSamples(e, 1);
    scene.goals.forEach(function (g) { square(g, 3 * k, scene.regions[g.region].color, 1); });
    drawStart();
    return Math.round(DDIM_STEPS * (1 - e));
  }

  function drawStep3(t) {
    var tree = scene.tree, n = Math.min(tree.iters.length, Math.floor(t));
    var cur = n > 0 ? tree.iters[n - 1] : null;
    var weights = cur ? cur.weights : scene.goals.map(function (g) { return g.weight0; });
    var topIdx = cur ? cur.top : 0;

    scene.regions.forEach(function (r) { drawRegion(r, 1, 1); });
    drawSamples(1, 0.35);                                    // the step-2 samples, faded

    ctx.lineWidth = 1.8 * k;
    ctx.lineCap = 'round';
    for (var i = 0; i < n; i++) {
      var it = tree.iters[i];
      ctx.strokeStyle = it.biased ? scene.regions[scene.goals[it.goal].region].color : pal.edge;
      ctx.globalAlpha = it.biased ? 0.9 : 0.6;
      ctx.beginPath();
      it.added.forEach(function (idx) {
        var a = tree.nodes[idx], b = tree.nodes[a.parent];
        ctx.moveTo(b.x * S, b.y * S);
        ctx.lineTo(a.x * S, a.y * S);
      });
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    var done = n >= tree.iters.length && tree.path;
    if (done) polyline(tree.path, pal.path, 3 * k, 0.95);

    scene.goals.forEach(function (g) {
      var w = weights[g.index], isTop = g.index === topIdx && !done;
      square(g, isTop ? 4.4 * k : (1.8 + 1.8 * w) * k, scene.regions[g.region].color, isTop ? 1 : 0.35 + 0.6 * w);
      if (isTop) {
        ctx.save();
        ctx.strokeStyle = pal.path;
        ctx.lineWidth = 1.6 * k;
        ctx.setLineDash([3 * k, 2 * k]);
        ctx.strokeRect(g.x * S - 9 * k, g.y * S - 9 * k, 18 * k, 18 * k);
        ctx.restore();
      }
    });
    drawStart();
    return { n: n, cur: cur, weights: weights, topIdx: topIdx, done: !!done };
  }

  /* ================= side UI ================= */

  var last = { lines: null, readout: null, live: null };

  function setLines(list) {
    var key = list.join(',');
    if (key === last.lines) return;
    last.lines = key;
    var on = {};
    list.forEach(function (l) { on[l] = true; });
    var algo = algos[state.step - 1];
    [].forEach.call(algo.querySelectorAll('[data-l]'), function (li) {
      li.classList.toggle('is-on', !!on[li.getAttribute('data-l')]);
    });
  }
  function setReadout(html) {
    if (html === last.readout) return;
    last.readout = html;
    readout.innerHTML = html;
  }
  function setLive(v) {
    if (v === last.live) return;
    last.live = v;
    fig.setAttribute('data-live', v);
  }
  function stat(label, value) {
    return '<div class="stat"><div class="stat-l">' + label + '</div><div class="stat-v">' + value + '</div></div>';
  }
  function swatch(c) { return '<span class="sw" style="background:' + c + '"></span>'; }

  function sideStep1(p) {
    var lines = p < 0.2 ? [1] : p < 0.5 ? [2, 3, 4] : p < 0.6 ? [5] : p < 0.7 ? [6] : [7];
    setLines(lines);
    var n = scene.regions.length;
    var rows = scene.regions.map(function (r, i) {
      var sa = clamp01(((p - 0.2) / 0.3) * n - i - 1e-6);
      var shown = Math.round(r.goals.length * clamp01((p - 0.7) / 0.25));
      return '<tr><td>' + swatch(r.color) + 'WGR<sub>' + (i + 1) + '</sub></td>' +
        '<td><span class="bar"><span style="width:' + (r.score * ease(sa) * 100).toFixed(1) + '%;background:' + reachColor(r.score) + '"></span></span>' +
        (sa > 0 ? (r.score * ease(sa)).toFixed(2) : '—') + '</td>' +
        '<td>' + (p >= 0.6 ? r.pMix.toFixed(2) : '—') + '</td>' +
        '<td>' + (p >= 0.7 ? shown : '—') + '</td></tr>';
    }).join('');
    setReadout('<table class="rtable"><thead><tr><th>region</th><th>score <i>s</i><sub>i</sub></th><th><i>p̃</i><sub>i</sub></th><th>goals</th></tr></thead><tbody>' + rows + '</tbody></table>');
  }
  function sideStep2(p, tStep) {
    setLines(p < 0.12 ? [1] : p < 0.92 ? [2, 3, 4, 5, 6] : [7]);
    var nPaths = scene.goals.length * PATHS_PER_GOAL;
    setReadout('<div class="stats">' +
      stat('DDIM step', tStep + ' / ' + DDIM_STEPS) +
      stat('goal configs <i>q</i><sub>g</sub>', scene.goals.length) +
      stat('path samples', nPaths + ' <small>(' + PATHS_PER_GOAL + ' per goal)</small>') +
      '</div>');
  }
  function sideStep3(info) {
    var cur = info.cur, lines, live;
    if (info.done) { lines = [28]; live = ''; }
    else if (!cur) { lines = [1, 2, 3]; live = ''; }
    else if (cur.biased) {
      lines = [4, 5, 6, 7, 8, 13, 14, 15].concat(cur.xOk ? [16] : [], [18, 19, 20], cur.yOk ? [21, 22] : [23, 24]);
      live = 'goal';
    } else {
      lines = [4, 5, 9, 10, 11, 13, 14, 15].concat(cur.xOk ? [16] : []);
      live = 'uniform';
    }
    setLines(lines);
    setLive(live);
    var g = scene.goals[info.topIdx];
    var status = info.done ? '<span class="ok">solved ✓</span>'
      : !cur ? 'ready'
      : cur.biased ? (cur.yOk ? '<span class="ok">reward</span>' : '<span class="bad">penalize</span>')
      : 'uniform';
    setReadout('<div class="stats">' +
      stat('iteration', info.n + ' <small>/ ' + scene.tree.iters.length + '</small>') +
      stat('tree nodes', cur ? cur.nodeCount : 1) +
      stat('top goal · weight', swatch(scene.regions[g.region].color) + 'WGR<sub>' + (g.region + 1) + '</sub> ' + info.weights[g.index].toFixed(2)) +
      stat('last step', status) +
      '</div>');
  }

  function draw() {
    if (!scene || !size) return;
    drawEnv();
    if (state.step === 1) { drawStep1(state.t); sideStep1(state.t); }
    else if (state.step === 2) { var ts = drawStep2(state.t); sideStep2(state.t, ts); }
    else { sideStep3(drawStep3(state.t)); }
    if (state.step !== 3) setLive('');
    if (document.activeElement !== scrub) scrub.value = Math.round(progress() * 1000);
  }

  function stepLength() { return state.step === 3 ? scene.tree.iters.length : 1; }

  // Positions the ◀ / ▶ buttons stop at. Step 1: one pseudo-code stage (one per region while scoring);
  // step 2: one DDIM step; step 3: one planner iteration (handled directly in stepBy).
  var stops = { 1: null, 2: null };
  function buildStops() {
    var n = scene.regions.length, s1 = [0, 0.2];
    for (var i = 1; i <= n; i++) s1.push(0.2 + 0.3 * i / n);
    stops[1] = s1.concat([0.6, 0.7, 1]);
    var s2 = [0, 0.12];                                   // x_T, then one stop per DDIM step
    for (var t = DDIM_STEPS - 1; t >= 0; t--) {
      var target = 1 - t / DDIM_STEPS, lo = 0, hi = 1;    // invert the easing: e(u) = target
      for (var it = 0; it < 30; it++) { var mid = (lo + hi) / 2; if (ease(mid) < target) lo = mid; else hi = mid; }
      s2.push(0.12 + 0.8 * hi);
    }
    stops[2] = s2.concat([1]);
  }
  function stepBy(dir) {
    if (!scene) return;
    var t = state.t, eps = 1e-6;
    if (state.step === 3) {
      t = dir > 0 ? Math.floor(t + eps) + 1 : Math.ceil(t - eps) - 1;
      t = Math.max(0, Math.min(stepLength(), t));
    } else {
      var list = stops[state.step], next = dir > 0 ? 1 : 0;
      if (dir > 0) { for (var i = 0; i < list.length; i++) if (list[i] > t + eps) { next = list[i]; break; } }
      else { for (var j = list.length - 1; j >= 0; j--) if (list[j] < t - eps) { next = list[j]; break; } }
      t = next;
    }
    state.t = t; state.hold = 0; state.playing = false; state.tour = false;
    syncPlayUI(); draw();
  }
  function progress() { return scene ? state.t / stepLength() : 0; }

  /* ================= control ================= */

  function syncStepUI() {
    tabs.forEach(function (b) {
      var on = +b.getAttribute('data-step') === state.step;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on);
    });
    clickables.forEach(function (g) {
      g.classList.toggle('is-active', +g.getAttribute('data-step') === state.step);
    });
    heads.forEach(function (h) { h.hidden = +h.getAttribute('data-step') !== state.step; });
    algos.forEach(function (a) { a.hidden = +a.getAttribute('data-step') !== state.step; });
    last.lines = null;
    syncPlayUI();
  }
  function syncPlayUI() {
    root.classList.toggle('is-playing', state.playing);
    playBtn.setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
  }
  function goStep(step, byUser) {
    state.step = step;
    state.t = 0;
    state.hold = 0;
    if (byUser) { state.tour = false; state.playing = true; }
    syncStepUI();
    draw();
  }

  function advance(dt) {
    var len = stepLength();
    if (state.t < len) {
      var rate = state.step === 3 ? Math.min(30, Math.max(8, len / 18)) : 1 / DURATION[state.step];
      state.t = Math.min(len, state.t + dt * rate);
      return;
    }
    state.hold += dt;
    if (state.tour) {
      if (state.hold > HOLD[state.step]) goStep(state.step % 3 + 1, false);
    } else {
      state.playing = false;
      syncPlayUI();
    }
  }

  var raf = 0, lastTs = null;
  function frame(ts) {
    raf = 0;
    var dt = lastTs == null ? 0 : Math.min(0.05, (ts - lastTs) / 1000);
    lastTs = ts;
    if (state.playing) advance(dt);
    draw();
    schedule();
  }
  function schedule() {
    if (!raf && state.visible && state.playing) raf = requestAnimationFrame(frame);
    if (!state.playing) lastTs = null;
  }

  tabs.forEach(function (b) {
    b.addEventListener('click', function () { goStep(+b.getAttribute('data-step'), true); schedule(); });
  });
  clickables.forEach(function (g) {
    function go() { goStep(+g.getAttribute('data-step'), true); schedule(); }
    g.addEventListener('click', go);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
    });
  });
  playBtn.addEventListener('click', function () {
    if (!scene) return;
    if (!state.playing && state.t >= stepLength()) { state.t = 0; state.hold = 0; }
    state.playing = !state.playing;
    syncPlayUI();
    schedule();
  });
  backBtn.addEventListener('click', function () { stepBy(-1); });
  fwdBtn.addEventListener('click', function () { stepBy(1); });
  root.addEventListener('keydown', function (e) {        // ← / → step while focus is on the demo controls
    if (e.target === scrub || !e.target.closest('.demo-controls')) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); stepBy(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stepBy(1); }
  });
  replayBtn.addEventListener('click', function () {
    state.t = 0; state.hold = 0; state.playing = true;
    syncPlayUI(); draw(); schedule();
  });
  scrub.addEventListener('input', function () {
    if (!scene) return;
    state.t = (scrub.value / 1000) * stepLength();
    state.hold = 0; state.playing = false; state.tour = false;
    syncPlayUI(); draw();
  });

  /* ================= boot ================= */

  function start() {
    if (!scene) {
      scene = buildScene();
      buildStops();
      readPalette();
      syncStepUI();
      resize();
    }
    schedule();
  }

  new MutationObserver(function () { readPalette(); draw(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener('resize', resize);

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      state.visible = entries[0].isIntersecting;
      if (state.visible) start();
    }, { rootMargin: '200px' }).observe(root);
  } else {
    state.visible = true;
    start();
  }
})();
