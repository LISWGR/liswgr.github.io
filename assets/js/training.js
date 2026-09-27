// Training section: animated "frozen z_env" arrows from the encoder figure to the two learned models,
// plus click-to-enlarge for the figures.
(function () {
  'use strict';

  var diagram = document.getElementById('train-diagram');
  if (!diagram) return;

  var NS = 'http://www.w3.org/2000/svg';
  var svg = diagram.querySelector('.train-links');
  var src = diagram.querySelector('.link-src');
  var dsts = [].slice.call(diagram.querySelectorAll('.link-dst'));
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var DUR = 2600;               // ms for a pulse to travel one arrow
  var paths = [], dots = [], label = null, visible = false, raf = 0;

  function el(name, attrs) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function layout() {
    var box = diagram.getBoundingClientRect();
    svg.innerHTML = '';
    paths = []; dots = [];
    if (getComputedStyle(svg).display === 'none' || !box.width) return;

    svg.setAttribute('viewBox', '0 0 ' + box.width + ' ' + box.height);
    var defs = el('defs', {});
    var marker = el('marker', { id: 'tl-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' });
    marker.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', class: 'tl-head' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    var s = src.getBoundingClientRect();
    var x0 = s.right - box.left, y0 = s.top + s.height / 2 - box.top, xEnd = x0;
    dsts.forEach(function (d) {
      var frame = d.closest('.train-img').getBoundingClientRect(), r = d.getBoundingClientRect();
      var x1 = frame.left - box.left - 2, y1 = r.top + r.height / 2 - box.top;
      var dx = Math.max(30, (x1 - x0) * 0.55);
      xEnd = x1;
      var p = el('path', {
        class: 'tl-path',
        d: 'M' + x0 + ' ' + y0 + ' C' + (x0 + dx) + ' ' + y0 + ' ' + (x1 - dx) + ' ' + y1 + ' ' + x1 + ' ' + y1,
        'marker-end': 'url(#tl-arrow)'
      });
      svg.appendChild(p);
      paths.push(p);
    });
    if (!reduceMotion) {
      paths.forEach(function (p) {
        for (var i = 0; i < 2; i++) {
          var c = el('circle', { r: 4, class: 'tl-dot' });
          svg.appendChild(c);
          dots.push({ c: c, p: p, len: p.getTotalLength(), offset: i / 2 });
        }
      });
    }
    // label in the empty wedge between the two branches
    label = el('text', { x: (x0 + xEnd) / 2 + 6, y: y0 + 4, class: 'tl-label', 'text-anchor': 'middle' });
    label.textContent = '❄ frozen z';
    var sub = el('tspan', { dy: 4, 'font-size': '10' });
    sub.textContent = 'env';
    label.appendChild(sub);
    svg.appendChild(label);
    tick(performance.now());
  }

  function tick(now) {
    raf = 0;
    dots.forEach(function (d) {
      var t = ((now / DUR) + d.offset) % 1;
      var pt = d.p.getPointAtLength(t * d.len);
      d.c.setAttribute('cx', pt.x);
      d.c.setAttribute('cy', pt.y);
      d.c.setAttribute('opacity', Math.min(1, t * 6, (1 - t) * 6).toFixed(2));
    });
    if (visible && dots.length) raf = requestAnimationFrame(tick);
  }

  function schedule() { if (!raf && visible && dots.length) raf = requestAnimationFrame(tick); }

  if (window.ResizeObserver) new ResizeObserver(layout).observe(diagram);
  else window.addEventListener('resize', layout);
  [].forEach.call(diagram.querySelectorAll('img'), function (img) { img.addEventListener('load', layout); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      schedule();
    }).observe(diagram);
  } else { visible = true; }
  layout();
  schedule();

  // click to enlarge
  var dialog = document.getElementById('train-zoom');
  if (dialog && dialog.showModal) {
    var big = dialog.querySelector('img');
    [].forEach.call(diagram.querySelectorAll('[data-zoom]'), function (btn) {
      btn.addEventListener('click', function () {
        var img = btn.querySelector('img');
        big.src = img.src;
        big.alt = img.alt;
        dialog.showModal();
      });
    });
    dialog.addEventListener('click', function (e) { if (e.target === dialog) dialog.close(); });
  }
})();
