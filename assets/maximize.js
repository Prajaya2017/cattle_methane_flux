// Maximize / restore button for each plot (.plot-box built by plot_box() in app.py).
// Dash loads every .js file in assets/ automatically.
(function () {
  function graphOf(box) {
    return box.querySelector(".js-plotly-plot");
  }

  function fit(box) {
    var gd = graphOf(box);
    if (gd && window.Plotly) {
      Plotly.relayout(gd, { width: box.clientWidth - 10, height: box.clientHeight - 10 });
    }
  }

  function setMaximized(box, on) {
    var gd = graphOf(box);
    var btn = box.querySelector(".max-btn");
    if (on) {
      box._savedHeight = gd && gd.layout ? gd.layout.height : null;
      box.classList.add("maximized");
      btn.textContent = "✕";
      btn.title = "Restore (Esc)";
      fit(box);
    } else {
      box.classList.remove("maximized");
      btn.textContent = "⛶";
      btn.title = "Maximize";
      if (gd && window.Plotly) {
        Plotly.relayout(gd, { width: null, height: box._savedHeight || null });
      }
    }
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest(".max-btn");
    if (!btn) return;
    var box = btn.closest(".plot-box");
    if (box) setMaximized(box, !box.classList.contains("maximized"));
  });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    document.querySelectorAll(".plot-box.maximized").forEach(function (box) {
      setMaximized(box, false);
    });
  });

  window.addEventListener("resize", function () {
    document.querySelectorAll(".plot-box.maximized").forEach(fit);
  });
})();
