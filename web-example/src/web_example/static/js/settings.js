"use strict";

/**
 * Settings-page helpers.
 *
 * 1. Live requirement hints for the embedder form. The remote (OpenAI-
 *    compatible /embeddings) backend cannot infer its vector dimension from
 *    the first response — the store is sized up-front — so model, endpoint,
 *    and dimension must be filled in before saving. Missing requirements are
 *    summarised in a single warning directly above the save button (the
 *    server validates this too).
 *
 * 2. Scroll stability: after an HTMX save the swapped section is re-rendered,
 *    which can yank the viewport to the page bottom. The scroll position is
 *    captured before each settings request and restored after the swap
 *    settles, so saving never moves the page.
 *
 * Event delegation + re-init on htmx:settle keeps both the global and the
 * per-user forms (and HTMX-partial re-renders) covered.
 */
(function () {
  var REMOTE = "remote";

  function fieldsetOf(element) {
    return element ? element.closest("fieldset") : null;
  }

  function valueOf(fieldset, selector) {
    var input = fieldset.querySelector(selector);
    return input ? String(input.value || "").trim() : "";
  }

  function missingRequirements(fieldset) {
    var missing = [];
    if (!valueOf(fieldset, ".js-embedder-model")) missing.push("a model id");
    if (!valueOf(fieldset, ".js-embedder-base-url")) missing.push("an endpoint URL");
    if (!valueOf(fieldset, ".js-embedder-dim")) missing.push("an explicit dimension");
    return missing;
  }

  function refresh(fieldset) {
    if (!fieldset) return;
    var form = fieldset.closest("form");
    if (!form) return;
    var status = form.querySelector(".js-remote-requirements");
    if (!status) return;
    var isRemote = valueOf(fieldset, ".js-embedder-backend") === REMOTE;
    var missing = isRemote ? missingRequirements(fieldset) : [];
    if (missing.length > 0) {
      status.textContent =
        "Remote embedding requires " + missing.join(", ") +
        " (the vector store is sized before the first response).";
      status.hidden = false;
    } else {
      status.hidden = true;
    }
  }

  function refreshAll(root) {
    var scope = root || document;
    var fieldsets = scope.querySelectorAll("fieldset");
    for (var i = 0; i < fieldsets.length; i++) {
      if (fieldsets[i].querySelector(".js-embedder-backend")) {
        refresh(fieldsets[i]);
      }
    }
  }

  var scrollBeforeRequest = null;

  function inSettingsScope(element) {
    return element && element.closest && element.closest(".settings-scope");
  }

  document.addEventListener("change", function (event) {
    var fieldset = fieldsetOf(event.target);
    if (fieldset && fieldset.querySelector(".js-embedder-backend")) refresh(fieldset);
  });
  document.addEventListener("input", function (event) {
    var fieldset = fieldsetOf(event.target);
    if (fieldset && fieldset.querySelector(".js-embedder-backend")) refresh(fieldset);
  });
  document.addEventListener("htmx:beforeRequest", function (event) {
    if (inSettingsScope(event.target)) {
      scrollBeforeRequest = window.scrollY;
    }
  });
  document.addEventListener("htmx:afterSettle", function (event) {
    if (inSettingsScope(event.target) && scrollBeforeRequest !== null) {
      window.scrollTo(0, scrollBeforeRequest);
      scrollBeforeRequest = null;
    }
    refreshAll(event.target);
  });
  document.addEventListener("DOMContentLoaded", function () {
    refreshAll(document);
  });
})();
