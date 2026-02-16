const state = {
  chats: [],
  activeChatId: null,
  activeChat: null,
  streaming: false,
};

const nodes = {
  chatList: document.getElementById("chat-list"),
  chatTitle: document.getElementById("chat-title"),
  messages: document.getElementById("messages"),
  modelStatus: document.getElementById("model-status"),
  newChatBtn: document.getElementById("new-chat-btn"),
  loadModelBtn: document.getElementById("load-model-btn"),
  form: document.getElementById("composer-form"),
  input: document.getElementById("composer-input"),
  sendBtn: document.getElementById("send-btn"),
  maxTokens: document.getElementById("max-tokens"),
  streamToggle: document.getElementById("stream-toggle"),
  toast: document.getElementById("toast"),
};

async function api(path, options = {}) {
  const request = {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  };
  const response = await fetch(path, request);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.error) message = payload.error;
    } catch (_) {
      // Keep fallback message.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function showToast(message) {
  nodes.toast.textContent = message;
  nodes.toast.classList.remove("hidden");
  setTimeout(() => nodes.toast.classList.add("hidden"), 2800);
}

function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  const withBlocks = escaped.replace(
    /```([\s\S]*?)```/g,
    (_, code) => `<pre><code>${code}</code></pre>`
  );
  const withInline = withBlocks.replace(/`([^`]+)`/g, "<code>$1</code>");
  return withInline.replace(/\n/g, "<br>");
}

function formatTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function setBusy(busy) {
  state.streaming = busy;
  nodes.sendBtn.disabled = busy;
  nodes.input.disabled = busy;
  nodes.newChatBtn.disabled = busy;
}

function renderSidebar() {
  nodes.chatList.innerHTML = "";
  if (!state.chats.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No chats yet.";
    nodes.chatList.append(empty);
    return;
  }

  for (const chat of state.chats) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `chat-item ${chat.id === state.activeChatId ? "active" : ""}`;
    item.innerHTML = `
      <div class="chat-item-title">${chat.title}</div>
      <div class="chat-item-meta">${chat.message_count} messages</div>
      <div class="chat-item-meta">${formatTime(chat.updated_at)}</div>
    `;
    item.addEventListener("click", () => openChat(chat.id));

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn btn-danger";
    removeBtn.textContent = "Delete";
    removeBtn.style.marginTop = "8px";
    removeBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      try {
        await api(`/api/chats/${chat.id}`, { method: "DELETE" });
        await reloadChats();
      } catch (error) {
        showToast(error.message);
      }
    });

    item.appendChild(removeBtn);
    nodes.chatList.append(item);
  }
}

function renderMessages() {
  if (!state.activeChat) {
    nodes.chatTitle.textContent = "New Chat";
    nodes.messages.innerHTML = `<div class="empty">Start a new conversation.</div>`;
    return;
  }

  nodes.chatTitle.textContent = state.activeChat.title || "New Chat";
  nodes.messages.innerHTML = "";

  if (!state.activeChat.messages.length) {
    nodes.messages.innerHTML = `<div class="empty">Send your first message.</div>`;
    return;
  }

  for (const message of state.activeChat.messages) {
    const role = message.role === "user" ? "user" : "assistant";
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;
    wrapper.innerHTML = `
      <div class="message-role">${message.role}</div>
      <div class="message-content">${renderMarkdown(message.content || "")}</div>
    `;
    nodes.messages.append(wrapper);
  }

  if (state.streaming) {
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.textContent = "Assistant is typing...";
    nodes.messages.append(typing);
  }

  nodes.messages.scrollTop = nodes.messages.scrollHeight;
}

function renderStatus(status) {
  if (status.loaded) {
    nodes.modelStatus.classList.add("loaded");
    nodes.modelStatus.textContent = `Model: ${status.model_id || "loaded"} (${status.arch || "n/a"})`;
    return;
  }
  nodes.modelStatus.classList.remove("loaded");
  nodes.modelStatus.textContent = "Model: not loaded";
}

async function refreshStatus() {
  const status = await api("/api/status", { method: "GET" });
  renderStatus(status);
}

async function reloadChats() {
  const payload = await api("/api/chats", { method: "GET" });
  state.chats = payload.chats || [];

  if (!state.chats.length) {
    state.activeChatId = null;
    state.activeChat = null;
  } else if (!state.activeChatId || !state.chats.some((v) => v.id === state.activeChatId)) {
    state.activeChatId = state.chats[0].id;
    state.activeChat = await api(`/api/chats/${state.activeChatId}`, { method: "GET" });
  }

  renderSidebar();
  renderMessages();
}

async function openChat(chatId) {
  state.activeChatId = chatId;
  state.activeChat = await api(`/api/chats/${chatId}`, { method: "GET" });
  renderSidebar();
  renderMessages();
}

async function createChat() {
  const chat = await api("/api/chats", {
    method: "POST",
    body: JSON.stringify({ title: null }),
  });
  await reloadChats();
  await openChat(chat.id);
}

function parseSsePayloads(buffer) {
  const events = [];
  const normalized = buffer.replace(/\r/g, "");
  const chunks = normalized.split("\n\n");
  const remainder = chunks.pop() || "";

  for (const chunk of chunks) {
    const lines = chunk
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());
    if (!lines.length) continue;
    try {
      events.push(JSON.parse(lines.join("\n")));
    } catch (_) {
      // Ignore malformed event.
    }
  }
  return { events, remainder };
}

async function streamChat(chatId, content, maxNewTokens) {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      max_new_tokens: maxNewTokens,
    }),
  });
  if (!response.ok || !response.body) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.error) message = payload.error;
    } catch (_) {
      // Keep fallback.
    }
    throw new Error(message);
  }

  const placeholder = {
    role: "assistant",
    content: "",
    created_at: new Date().toISOString(),
  };
  state.activeChat.messages.push(placeholder);
  renderMessages();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const { events, remainder } = parseSsePayloads(buffer);
    buffer = remainder;

    for (const event of events) {
      if (event.type === "token") {
        placeholder.content += event.content || "";
        renderMessages();
      } else if (event.type === "done") {
        placeholder.content = event.content || placeholder.content;
        renderMessages();
      } else if (event.type === "error") {
        throw new Error(event.content || "Streaming error");
      }
    }
  }
}

async function sendMessage(content) {
  if (!state.activeChatId || !state.activeChat) {
    await createChat();
  }

  const maxNewTokens = Math.max(
    1,
    Math.min(256, Number.parseInt(nodes.maxTokens.value, 10) || 128)
  );

  state.activeChat.messages.push({
    role: "user",
    content,
    created_at: new Date().toISOString(),
  });
  renderMessages();
  setBusy(true);

  try {
    if (nodes.streamToggle.checked) {
      await streamChat(state.activeChatId, content, maxNewTokens);
      state.activeChat = await api(`/api/chats/${state.activeChatId}`, { method: "GET" });
    } else {
      state.activeChat = await api(`/api/chats/${state.activeChatId}/messages`, {
        method: "POST",
        body: JSON.stringify({
          content,
          max_new_tokens: maxNewTokens,
        }),
      });
    }

    await refreshStatus();
    await reloadChats();
    await openChat(state.activeChatId);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadModel() {
  setBusy(true);
  try {
    const payload = await api("/api/model/load", {
      method: "POST",
      body: JSON.stringify({ model_dir: "models" }),
    });
    showToast(`Loaded ${payload.model_id}`);
    await refreshStatus();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function wireEvents() {
  nodes.newChatBtn.addEventListener("click", async () => {
    try {
      await createChat();
    } catch (error) {
      showToast(error.message);
    }
  });

  nodes.loadModelBtn.addEventListener("click", loadModel);

  nodes.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.streaming) return;

    const content = nodes.input.value.trim();
    if (!content) return;
    nodes.input.value = "";
    await sendMessage(content);
  });

  nodes.input.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      nodes.form.requestSubmit();
    }
  });
}

async function init() {
  wireEvents();
  await refreshStatus();
  await reloadChats();
  if (!state.activeChatId) {
    await createChat();
  } else {
    renderSidebar();
    renderMessages();
  }
}

init().catch((error) => {
  showToast(error.message);
});
