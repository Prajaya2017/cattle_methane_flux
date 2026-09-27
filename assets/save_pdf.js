// "Save plots (PDF)": renders every plot on the current tab to a multi-page PDF in the browser.
// Uses Plotly.toImage (bundled with Dash) and jsPDF (assets/jspdf.umd.min.js). No server work needed.
(function () {
  async function savePDF(btn) {
    const { jsPDF } = window.jspdf || {};
    if (!jsPDF || !window.Plotly) { alert("PDF library not loaded yet - please try again."); return; }
    const plots = Array.from(document.querySelectorAll("#tab-content .js-plotly-plot"))
      .filter(gd => gd.offsetWidth > 0 && gd.offsetHeight > 0);
    if (!plots.length) { alert("No plots on this tab to save."); return; }

    const label = btn.innerHTML; btn.disabled = true;
    try {
      const tab = (document.querySelector("#tabs .tab--selected") || {}).innerText || "Dashboard";
      const title = (document.querySelector("h1") || {}).innerText || "Dashboard";
      const dates = ["dp-range", "dp-range-end-date"].map(id => (document.getElementById(id) || {}).value)
        .filter(v => v && /\d{4}-\d{2}-\d{2}/.test(v));
      const note = Array.from(document.querySelectorAll("#tab-content div"))
        .map(d => d.innerText).find(t => /records/.test(t) && t.length < 400) || "";

      const pdf = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
      const W = pdf.internal.pageSize.getWidth(), H = pdf.internal.pageSize.getHeight();
      const M = 10, GAP = 5;

      // header on page 1
      pdf.setFont("helvetica", "bold"); pdf.setFontSize(15); pdf.text(title.replace("\u2084", "4"), M, M + 5);
      pdf.setFont("helvetica", "normal"); pdf.setFontSize(10);
      pdf.text(`${tab}  |  ${dates.join(" to ")}  |  saved ${new Date().toLocaleString()}`, M, M + 11);
      if (note) { pdf.setFontSize(8); pdf.text(pdf.splitTextToSize(note, W - 2 * M), M, M + 16); }
      let y = M + (note ? 24 : 18), col = 0, rowH = 0, rowCols = 0;

      // small cards: 4 per row, larger analysis cards: 3 per row, wide plots: full width
      const colsFor = gd => {
        const aspect = gd.offsetHeight / gd.offsetWidth;
        if (aspect < 0.55) return 1;
        return gd.offsetWidth < 420 ? 4 : 3;
      };
      const newRow = () => { if (col > 0) { y += rowH + GAP; } col = 0; rowH = 0; };

      for (let i = 0; i < plots.length; i++) {
        btn.innerHTML = `Saving ${i + 1}/${plots.length}...`;
        const gd = plots[i];
        const cols = colsFor(gd);
        if (cols !== rowCols) { newRow(); rowCols = cols; }      // a different card size starts a new row
        const w = (W - 2 * M - GAP * (cols - 1)) / cols;
        const h = w * gd.offsetHeight / gd.offsetWidth;
        if (col === 0 && y + h > H - M) { pdf.addPage(); y = M; }
        const img = await Plotly.toImage(gd, { format: "jpeg", scale: 1.6,
                                               width: gd.offsetWidth, height: gd.offsetHeight });
        pdf.addImage(img, "JPEG", M + col * (w + GAP), y, w, h, undefined, "FAST");
        rowH = Math.max(rowH, h);
        col += 1;
        if (col >= cols) { y += rowH + GAP; col = 0; rowH = 0; }
      }
      const n = pdf.getNumberOfPages();
      for (let p = 1; p <= n; p++) { pdf.setPage(p); pdf.setFontSize(8); pdf.text(`${p} / ${n}`, W - M, H - 4, { align: "right" }); }
      const safe = s => s.replace(/[^A-Za-z0-9_-]+/g, "_");
      pdf.save(`TGA310_${safe(tab)}_${safe(dates.join("_to_") || "all")}.pdf`);
    } catch (e) {
      console.error(e); alert("Could not create the PDF: " + e);
    } finally { btn.innerHTML = label; btn.disabled = false; }
  }
  document.addEventListener("click", e => {
    const b = e.target.closest && e.target.closest("#save-pdf-btn");
    if (b && !b.disabled) { savePDF(b); }
  });
})();
