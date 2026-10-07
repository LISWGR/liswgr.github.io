// Chapter buttons under the project video: click to seek, highlight the current chapter.
(function(){
  var video = document.getElementById('project-video');
  var list = document.getElementById('video-chapters');
  if (!video || !list) return;

  function secs(s){
    return s.split(':').reduce(function(acc, p){ return acc * 60 + parseFloat(p); }, 0);
  }

  var buttons = Array.prototype.slice.call(list.querySelectorAll('button'));
  var times = buttons.map(function(b){ return secs(b.getAttribute('data-t')); });

  buttons.forEach(function(b, i){
    b.addEventListener('click', function(){
      video.currentTime = times[i];
      video.play();
    });
  });

  function sync(){
    var t = video.currentTime, cur = -1;
    for (var i = 0; i < times.length; i++) if (t >= times[i] - 0.25) cur = i;
    buttons.forEach(function(b, i){ b.classList.toggle('is-active', i === cur); });
  }
  video.addEventListener('timeupdate', sync);
  video.addEventListener('seeked', sync);
  sync();
})();
