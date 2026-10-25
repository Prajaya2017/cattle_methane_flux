// Hovering a plot title shows the full variable name(s), e.g. "FCH4 — Methane (CH4) flux".
// The text comes from figure.layout.meta.tip (set in app.py); subplot titles use meta.ann_tips[i].
(function () {
  function tipEl() { return document.getElementById("title-tooltip"); }

  function tipFor(target) {
    var gd = target.closest && target.closest(".js-plotly-plot");
    var meta = gd && gd.layout && gd.layout.meta;
    if (!meta) return "";
    if (target.closest(".gtitle")) return meta.tip || "";
    var ann = target.closest(".annotation");
    if (ann && meta.ann_tips) {
      var i = parseInt(ann.getAttribute("data-index"), 10);
      return meta.ann_tips[i] || "";
    }
    return "";
  }

  function place(el, e) {
    var pad = 14, w = el.offsetWidth, h = el.offsetHeight;
    var x = Math.min(e.clientX + pad, window.innerWidth - w - 8);
    var y = e.clientY + pad + h > window.innerHeight ? e.clientY - h - pad : e.clientY + pad;
    el.style.left = Math.max(8, x) + "px";
    el.style.top = y + "px";
  }

  document.addEventListener("mouseover", function (e) {
    var t = e.target;
    if (!t.closest || !(t.closest(".gtitle") || t.closest(".annotation"))) return;
    var text = tipFor(t), el = tipEl();
    if (!text || !el) return;
    el.textContent = text;
    el.classList.add("show");
    place(el, e);
  });
  document.addEventListener("mousemove", function (e) {
    var el = tipEl();
    if (el && el.classList.contains("show")) place(el, e);
  });
  document.addEventListener("mouseout", function (e) {
    var t = e.target;
    if (t.closest && (t.closest(".gtitle") || t.closest(".annotation"))) {
      var el = tipEl(); if (el) el.classList.remove("show");
    }
  });
})();

// Maximized plot: Esc or a click on the dark background closes it (same as the Close button)
(function () {
  function closeModal() {
    var o = document.getElementById("modal-overlay");
    var b = document.getElementById("close-modal-btn");
    if (o && b && o.style.display !== "none") b.click();
  }
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeModal(); });
  document.addEventListener("click", function (e) { if (e.target && e.target.id === "modal-overlay") closeModal(); });
})();
