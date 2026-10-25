(function () {
  var src = document.currentScript && document.currentScript.getAttribute('data-src')
  var frame = document.createElement('iframe')
  frame.src = src || '/embed.html'
  frame.title = 'Atelier Try-On'
  frame.style.cssText = 'width:100%;min-height:720px;border:0;border-radius:16px;background:#0c0d11'
  var host = document.getElementById('atelier-tryon')
  if (host) host.appendChild(frame)
})()
