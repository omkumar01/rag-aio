/* Chat page behaviour: fetch + SSE streaming into the thread, sidebar refresh. */
(function () {
  "use strict";

  var composer = document.getElementById("composer");
  var thread = document.getElementById("thread");
  var messageInput = document.getElementById("message-input");
  var sendBtn = document.getElementById("send-btn");
  var errorsBox = document.getElementById("composer-errors");
  var sidebar = document.getElementById("sidebar");
  var newChatBtn = document.getElementById("new-chat-btn");

  if (!composer || !thread || !messageInput || !sendBtn) return;

  var currentConversationId = "";

  function showError(message) {
    if (!errorsBox) return;
    errorsBox.textContent = message;
    errorsBox.hidden = false;
  }

  function clearErrors() {
    if (!errorsBox) return;
    errorsBox.textContent = "";
    errorsBox.hidden = true;
  }

  function appendMessage(role, text) {
    var row = document.createElement("div");
    row.className = "chat-message " + role;
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    var body = document.createElement("div");
    body.className = "bubble-text";
    body.textContent = text;
    bubble.appendChild(body);
    row.appendChild(bubble);
    thread.appendChild(row);
    thread.scrollTop = thread.scrollHeight;
    return body;
  }

  function refreshSidebar() {
    if (window.htmx && sidebar) window.htmx.trigger(document.body, "conv-refresh");
  }

  function hasFiles(formData) {
    var entries = formData.getAll("doc_files").concat(formData.getAll("image_files"));
    return entries.some(function (item) {
      return item && typeof item === "object" && item.name;
    });
  }

  function resetThread() {
    currentConversationId = "";
    thread.innerHTML =
      '<div class="thread-empty"><h2>Ask your documents</h2>' +
      "<p>Attach files or just ask a question to get started.</p></div>";
  }

  composer.addEventListener("submit", function (event) {
    event.preventDefault();
    clearErrors();

    var question = messageInput.value.trim();
    var formData = new FormData(composer);
    if (!question && !hasFiles(formData)) return;

    appendMessage("user", question);
    messageInput.value = "";
    if (question) formData.set("message", question);
    if (currentConversationId) formData.set("conversation_id", currentConversationId);

    sendBtn.disabled = true;
    var assistantBody = appendMessage("assistant", "");
    assistantBody.classList.add("streaming");
    var sawError = false;

    fetch("/chat/ask", { method: "POST", body: formData })
      .then(function (response) {
        if (!response.ok) {
          return response
            .json()
            .catch(function () { return {}; })
            .then(function (data) {
              throw new Error((data && data.error) || "Request failed (" + response.status + ")");
            });
        }
        return consumeStream(response.body, assistantBody, function (payload) {
          if (payload.type === "meta" && payload.conversation_id) {
            currentConversationId = payload.conversation_id;
          } else if (payload.type === "error") {
            sawError = true;
            showError(payload.message || "Something went wrong.");
          }
        });
      })
      .catch(function (err) {
        sawError = true;
        showError(err && err.message ? err.message : String(err));
      })
      .then(function () {
        assistantBody.classList.remove("streaming");
        if (sawError && !assistantBody.textContent) {
          var row = assistantBody.closest(".chat-message");
          if (row) row.remove();
        }
        sendBtn.disabled = false;
        refreshSidebar();
      });
  });

  function consumeStream(body, assistantBody, onEvent) {
    var reader = body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";

    function pump() {
      return reader.read().then(function (result) {
        if (result.done) return null;
        buffer += decoder.decode(result.value, { stream: true });
        var parts = buffer.split("\n\n");
        buffer = parts.pop();
        parts.forEach(function (block) {
          var dataLine = block.split("\n").find(function (line) {
            return line.indexOf("data: ") === 0;
          });
          if (!dataLine) return;
          var payload;
          try {
            payload = JSON.parse(dataLine.slice(6));
          } catch (err) {
            return;
          }
          if (payload.type === "delta" && payload.text) {
            assistantBody.textContent += payload.text;
            thread.scrollTop = thread.scrollHeight;
          }
          onEvent(payload);
        });
        return pump();
      });
    }

    return pump();
  }

  // Track the conversation loaded into the thread via HTMX sidebar clicks.
  document.body.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail && event.detail.target && event.detail.target.id === "thread") {
      var inner = thread.querySelector("[data-conversation-id]");
      currentConversationId = inner ? inner.getAttribute("data-conversation-id") : "";
      thread.scrollTop = thread.scrollHeight;
    }
  });

  if (newChatBtn) newChatBtn.addEventListener("click", resetThread);
})();
