"use strict";

/**
 * Alpine.js component backing the Mass Upload page.
 * Registered as a global so it works regardless of alpine:init ordering.
 *
 * @param {{ max?: number, exts?: string[] }} config
 */
function uploadPanel(config) {
  var max = (config && config.max) || 500;
  var exts = (config && config.exts) || [];
  return {
    tab: "files",
    target: "user",
    dragging: false,
    files: [],
    max: max,
    exts: exts,

    get countLabel() {
      var ok = this.files.filter(function (f) { return f.ok; }).length;
      return this.files.length + " file(s) selected, " + ok + " supported (limit " + this.max + ")";
    },

    get overLimitLabel() {
      return "Too many files: the limit is " + this.max;
    },

    isSupported(name) {
      var lower = String(name || "").toLowerCase();
      return this.exts.some(function (ext) { return lower.endsWith(ext); });
    },

    onPick(event) {
      this.setFiles(event.target.files);
    },

    onDrop(event) {
      this.dragging = false;
      var input = document.getElementById("upload-file-input");
      var dropped = event.dataTransfer ? event.dataTransfer.files : null;
      if (input && dropped) {
        // Mirror the dropped FileList onto the real input so the form submits it.
        try {
          input.files = dropped;
        } catch (_err) {
          /* older browsers: form submit falls back to manual selection */
        }
      }
      this.setFiles(dropped);
    },

    setFiles(fileList) {
      var self = this;
      this.files = Array.from(fileList || []).map(function (f) {
        return { name: f.name, ok: self.isSupported(f.name) };
      });
    },
  };
}

window.uploadPanel = uploadPanel;

/**
 * Polling gate for the jobs panel: HTMX evaluates this before each periodic
 * refetch, so polling only happens while a job is queued/running (the jobs
 * partial renders the marker for those) and stops once everything is terminal.
 *
 * @returns {boolean}
 */
window.uploadJobsActive = function () {
  return document.querySelector("#jobs-panel [data-jobs-active]") !== null;
};
