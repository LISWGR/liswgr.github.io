// Light/dark toggle. The initial theme is set inline in <head> (see _layouts/default.html).
(function(){
  var root = document.documentElement;
  var btn = document.getElementById('theme-toggle');
  function sync(){
    var dark = root.getAttribute('data-theme') === 'dark';
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  btn.addEventListener('click', function(){
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch(e){}
    sync();
  });
  sync();
})();
