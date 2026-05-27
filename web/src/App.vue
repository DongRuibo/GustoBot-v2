<template>
  <main class="app-shell">
    <aside class="sidebar">
      <div class="brand-row">
        <div>
          <h1>GustoBot-v2</h1>
          <p>Recipe QA Workspace</p>
        </div>
        <button class="icon-button" type="button" title="刷新会话" @click="loadSessions">
          <RefreshCw :size="18" />
        </button>
      </div>

      <button class="new-session" type="button" @click="createSession">
        <Plus :size="18" />
        <span>新建会话</span>
      </button>

      <div class="session-list">
        <button
          v-for="session in sessions"
          :key="session.session_id"
          :class="['session-item', { active: session.session_id === activeSessionId }]"
          type="button"
          @click="selectSession(session.session_id)"
        >
          <MessageSquare :size="17" />
          <span>{{ session.title }}</span>
          <small>{{ session.message_count }}</small>
        </button>
      </div>
    </aside>

    <section class="chat-surface">
      <header class="chat-header">
        <div>
          <h2>{{ activeTitle }}</h2>
          <p>{{ activeSessionId || "未创建会话" }}</p>
        </div>
        <button
          class="icon-button danger"
          type="button"
          title="删除当前会话"
          :disabled="!activeSessionId"
          @click="deleteActiveSession"
        >
          <Trash2 :size="18" />
        </button>
      </header>

      <div ref="messagesEl" class="message-list">
        <div v-if="messages.length === 0" class="empty-state">
          <Database :size="24" />
          <span>开始新的菜谱问答</span>
        </div>
        <button
          v-for="message in messages"
          :key="message.id"
          :class="['message-row', message.role]"
          type="button"
          @click="selectedMessage = message"
        >
          <span class="message-role">{{ message.role === "user" ? "你" : "GustoBot" }}</span>
          <span class="message-content">{{ message.content }}</span>
          <span v-if="message.routeType" class="route-pill">{{ message.routeType }}</span>
        </button>
      </div>

      <footer class="composer">
        <div class="attachment-row">
          <label class="tool-button">
            <Upload :size="17" />
            <span>文件</span>
            <input type="file" class="hidden-input" @change="onUpload($event, 'file')" />
          </label>
          <label class="tool-button">
            <Image :size="17" />
            <span>图片</span>
            <input accept="image/*" type="file" class="hidden-input" @change="onUpload($event, 'image')" />
          </label>
          <span v-if="pendingAttachment" class="attachment-name">{{ pendingAttachment.filename }}</span>
          <span v-if="statusText" class="status-text">{{ statusText }}</span>
        </div>

        <form class="input-row" @submit.prevent="sendMessage">
          <textarea
            v-model="draft"
            rows="3"
            placeholder="输入问题"
            :disabled="isSending"
            @keydown.enter.exact.prevent="sendMessage"
          />
          <button class="send-button" type="submit" :disabled="isSending || !draft.trim()">
            <Send :size="18" />
            <span>发送</span>
          </button>
        </form>
      </footer>
    </section>

    <aside class="evidence-panel">
      <header>
        <h2>Evidence</h2>
        <button class="icon-button" type="button" title="选择最新回答" @click="selectLatestAssistant">
          <ChevronRight :size="18" />
        </button>
      </header>
      <div v-if="selectedMessage?.routeType" class="meta-line">
        <span>Route</span>
        <strong>{{ selectedMessage.routeType }}</strong>
      </div>
      <div v-if="selectedMessage?.traceId" class="meta-line">
        <span>Trace</span>
        <code>{{ selectedMessage.traceId }}</code>
      </div>
      <div class="evidence-list">
        <article v-for="(evidence, index) in selectedEvidence" :key="index" class="evidence-item">
          <div class="evidence-head">
            <strong>{{ evidence.source_type }}</strong>
            <span>{{ evidence.source_id }}</span>
          </div>
          <p>{{ evidence.content }}</p>
        </article>
      </div>
    </aside>
  </main>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import {
  ChevronRight,
  Database,
  Image,
  MessageSquare,
  Plus,
  RefreshCw,
  Send,
  Trash2,
  Upload
} from "lucide-vue-next";

interface Attachment {
  type: string;
  filename?: string;
  content_type?: string;
  uri?: string;
}

interface RouteDecision {
  route_type: string;
  confidence: number;
  reason: string;
}

interface Evidence {
  source_type: string;
  content: string;
  score: number;
  source_id: string;
  metadata: Record<string, unknown>;
  trace_id: string;
}

interface ChatResponse {
  trace_id: string;
  answer: string;
  route_decision: RouteDecision;
  evidences: Evidence[];
  need_clarification: boolean;
  session_id?: string;
  message_id?: string;
}

interface ChatStreamEvent {
  event: "assistant_start" | "answer_delta" | "done" | "error";
  session_id?: string;
  message_id?: string;
  trace_id?: string;
  route_decision?: RouteDecision;
  evidences?: Evidence[];
  need_clarification?: boolean;
  delta?: string;
  response?: ChatResponse;
  message?: string;
}

interface SessionSummary {
  session_id: string;
  user_id?: string;
  title: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  message_count: number;
}

interface MessageItem {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  route_type?: string;
  trace_id?: string;
  evidences: Evidence[];
  metadata: Record<string, unknown>;
  created_at: string;
  order_index: number;
}

interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  routeType?: string;
  traceId?: string;
  evidences: Evidence[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const USER_KEY = "gustobot-v2-user-id";
const storedUserId = localStorage.getItem(USER_KEY) || crypto.randomUUID();
localStorage.setItem(USER_KEY, storedUserId);

const sessions = ref<SessionSummary[]>([]);
const messages = ref<UiMessage[]>([]);
const selectedMessage = ref<UiMessage | null>(null);
const activeSessionId = ref("");
const draft = ref("");
const pendingAttachment = ref<Attachment | null>(null);
const statusText = ref("");
const isSending = ref(false);
const messagesEl = ref<HTMLElement | null>(null);

const activeTitle = computed(() => {
  const session = sessions.value.find((item) => item.session_id === activeSessionId.value);
  return session?.title || "新会话";
});

const selectedEvidence = computed(() => selectedMessage.value?.evidences || []);

onMounted(async () => {
  await loadSessions();
});

async function loadSessions() {
  sessions.value = await api<SessionSummary[]>(`/api/v1/sessions?user_id=${encodeURIComponent(storedUserId)}`);
}

async function createSession() {
  const session = await api<SessionSummary>("/api/v1/sessions", {
    method: "POST",
    body: JSON.stringify({ user_id: storedUserId, title: "新会话" })
  });
  sessions.value = [session, ...sessions.value];
  activeSessionId.value = session.session_id;
  messages.value = [];
  selectedMessage.value = null;
}

async function selectSession(sessionId: string) {
  activeSessionId.value = sessionId;
  const items = await api<MessageItem[]>(`/api/v1/sessions/${sessionId}/messages`);
  messages.value = items.map((item) => ({
    id: item.message_id,
    role: item.role,
    content: item.content,
    routeType: item.route_type,
    traceId: item.trace_id,
    evidences: item.evidences || []
  }));
  selectLatestAssistant();
  await scrollToBottom();
}

async function deleteActiveSession() {
  if (!activeSessionId.value) return;
  await api<void>(`/api/v1/sessions/${activeSessionId.value}`, { method: "DELETE" });
  sessions.value = sessions.value.filter((item) => item.session_id !== activeSessionId.value);
  activeSessionId.value = "";
  messages.value = [];
  selectedMessage.value = null;
}

async function sendMessage() {
  const text = draft.value.trim();
  if (!text || isSending.value) return;

  const userMessage: UiMessage = {
    id: crypto.randomUUID(),
    role: "user",
    content: text,
    evidences: []
  };
  messages.value.push(userMessage);
  draft.value = "";
  isSending.value = true;
  statusText.value = "请求中";
  const assistantMessage: UiMessage = {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "正在生成...",
    evidences: []
  };
  messages.value.push(assistantMessage);
  selectedMessage.value = assistantMessage;
  await scrollToBottom();

  try {
    const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        user_id: storedUserId,
        session_id: activeSessionId.value || undefined,
        attachments: pendingAttachment.value ? [pendingAttachment.value] : []
      })
    });
    if (!response.ok) throw new Error(await response.text());
    await readChatStream(response, assistantMessage);
    pendingAttachment.value = null;
    await loadSessions();
    await scrollToBottom();
  } catch (error) {
    assistantMessage.content = error instanceof Error ? error.message : "请求失败，请稍后再试。";
    selectedMessage.value = assistantMessage;
  } finally {
    isSending.value = false;
    statusText.value = "";
  }
}

async function onUpload(event: Event, kind: "file" | "image") {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  statusText.value = "上传中";
  try {
    const formData = new FormData();
    formData.append(kind === "image" ? "image" : "file", file);
    const response = await fetch(`${API_BASE}/api/v1/upload/${kind}`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = (await response.json()) as { attachment: Attachment };
    pendingAttachment.value = payload.attachment;
    statusText.value = "已附加";
  } finally {
    input.value = "";
  }
}

function selectLatestAssistant() {
  selectedMessage.value = [...messages.value].reverse().find((item) => item.role === "assistant") || null;
}

async function scrollToBottom() {
  await nextTick();
  if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
}

async function readChatStream(response: Response, assistantMessage: UiMessage) {
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = await consumeStreamBuffer(buffer, assistantMessage);
  }
  buffer += decoder.decode();
  await consumeStreamBuffer(`${buffer}\n`, assistantMessage);
}

async function consumeStreamBuffer(buffer: string, assistantMessage: UiMessage): Promise<string> {
  const lines = buffer.split("\n");
  const rest = lines.pop() || "";
  for (const line of lines) {
    if (!line.trim()) continue;
    await applyChatStreamEvent(JSON.parse(line) as ChatStreamEvent, assistantMessage);
  }
  return rest;
}

async function applyChatStreamEvent(event: ChatStreamEvent, assistantMessage: UiMessage) {
  if (event.event === "assistant_start") {
    if (event.message_id) assistantMessage.id = event.message_id;
    activeSessionId.value = event.session_id || activeSessionId.value;
    assistantMessage.traceId = event.trace_id;
    assistantMessage.routeType = event.route_decision?.route_type;
    assistantMessage.evidences = event.evidences || [];
    assistantMessage.content = "";
    statusText.value = "生成中";
  } else if (event.event === "answer_delta") {
    assistantMessage.content += event.delta || "";
    await scrollToBottom();
  } else if (event.event === "done" && event.response) {
    activeSessionId.value = event.response.session_id || activeSessionId.value;
    assistantMessage.id = event.response.message_id || assistantMessage.id;
    assistantMessage.content = event.response.answer;
    assistantMessage.routeType = event.response.route_decision.route_type;
    assistantMessage.traceId = event.response.trace_id;
    assistantMessage.evidences = event.response.evidences || [];
  } else if (event.event === "error") {
    throw new Error(event.message || "流式响应失败。");
  }
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {})
    }
  });
  if (!response.ok) throw new Error(await response.text());
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
</script>
