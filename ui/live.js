if (!window.UI_CONFIG.mock) {
  const byId = (id) => document.getElementById(id);
  const stages = {queued: "접수", processing: "이벤트 선별 · 임베딩 중", ready: "검색 가능", failed: "처리 실패"};

  async function refresh() {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      // The standalone D API supports search and playback only.
      if (response.status === 404) return false;
      if (!response.ok) throw new Error();
      const status = await response.json();
      byId("ingest").hidden = false;
      byId("modeBadge").textContent = (status.llm_ready ? "Qwen3.5-4B 연결됨" : "Qwen 연결 끊김") + " · 벡터 " + status.vectors + "개";
      const rows = status.jobs.map((job) => {
        const row = document.createElement("li");
        const name = document.createElement("strong");
        name.textContent = job.name;
        const state = document.createElement("span");
        state.textContent = stages[job.state] + (job.state === "ready" ? ` · 이벤트 ${job.events} · 벡터 ${job.vectors}` : "");
        row.append(name, state);
        if (job.error) row.append(document.createTextNode(job.error));
        return row;
      });
      byId("jobs").replaceChildren(...rows);
    } catch {
      byId("modeBadge").textContent = "서비스 연결 대기 중";
    }
  }

  async function sendEvent(path, body) {
    const message = byId("uploadStatus");
    message.textContent = "전송 중…";
    try {
      const response = await fetch(path, {method: "POST", body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "전송에 실패했습니다.");
      message.textContent = "접수했습니다. 아래에서 처리 상태를 확인하세요.";
      await refresh();
    } catch (err) {
      message.textContent = err.message;
    }
  }
  byId("demo").onclick = () => {
    byId("representation").value = "object";
    if (!byId("text").value.trim()) byId("text").value = byId("text").placeholder;
    return sendEvent("/api/events/demo");
  };
  byId("clip").onchange = async (event) => {
    const file = event.target.files[0];
    if (file) await sendEvent("/api/events/clip?name=" + encodeURIComponent(file.name), file);
    event.target.value = "";
  };
  async function poll() {
    if (await refresh() !== false) setTimeout(poll, 2000);
  }
  poll();
}
