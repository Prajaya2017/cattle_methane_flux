// Close the wind-direction checkbox pop-up when clicking outside it, pressing Esc, or clicking "Done".
// (Dash loads every .js file in the assets/ folder automatically.)
(function () {
  function closeWD() {
    var d = document.getElementById("wd-details");
    if (d && d.open) { d.open = false; }
  }
  document.addEventListener("click", function (e) {
    var d = document.getElementById("wd-details");
    if (!d || !d.open) { return; }
    if (e.target.closest && e.target.closest("#wd-done")) { closeWD(); return; }
    if (!d.contains(e.target)) { closeWD(); }
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { closeWD(); }
  });
})();
