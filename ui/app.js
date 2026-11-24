(() => {
  "use strict";
  const C = window.UI_CONFIG;
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("form"), gallery: $("gallery"), videoIds: $("videoIds"),
    versionGroup: $("searchVersions"), compare: $("compare"), comparison: $("comparison"),
    tabText: $("tabText"), tabImage: $("tabImage"),
    textBox: $("textBox"), text: $("text"),
    imageBox: $("imageBox"), image: $("image"), imagePreview: $("imagePreview"),
    representation: $("representation"), topK: $("topK"), submit: $("submit"),
    status: $("status"), meta: $("meta"), results: $("results"),
    stage: $("stage"), video: $("video"), overlay: $("overlay"), stageEmpty: $("stageEmpty"),
    playSegment: $("playSegment"), mockFile: $("mockFile"), mockVideo: $("mockVideo"),
    info: $("info"), modeBadge: $("modeBadge"),
  };

  let mode = "text";          // "text" | "image"
  let reqCounter = 0;         // 가장 최근 요청만 화면에 반영
  let current = null;         // 선택된 결과
  let stopAt = null;          // 구간 재생 종료 시각(초)
  let mockVideoUrl = C.mock_video_url;
  let lastSeekErrorMs = null;
  let searchVersion = "original";
  let priorRepresentation = "combined";
  const turns = [];
  const versionButtons = [...el.versionGroup.querySelectorAll("button[data-version]")];

  function addTurn(role, content) {
    const bubble = document.createElement("p");
    const who = document.createElement("strong");
    who.textContent = role === "user" ? "나" : "Qwen3.5";
    bubble.className = role;
    bubble.append(who, document.createTextNode(content));
    $("chat").append(bubble);
    $("chat").scrollTop = $("chat").scrollHeight;
    turns.push({role, content: content.slice(0, 1200)});
  }

  el.modeBadge.textContent = C.mock ? "mock (예시 응답)" : "API " + (C.apiBase || "(같은 주소)");
  el.modeBadge.classList.toggle("mock", !!C.mock);
  el.mockFile.hidden = !C.mock;

  // ---------- 상태 표시 ----------
  function setStatus(kind, msg) {
    el.status.className = kind;
    el.status.textContent = msg;
  }
  const ERR_KO = {
    INVALID_INPUT: "입력값이 올바르지 않습니다.",
    TEXT_TOO_LONG: "검색어가 500자를 넘었습니다.",
    IMAGE_TOO_LARGE: "이미지가 너무 큽니다(5MB 이하).",
    INVALID_IMAGE: "이미지를 읽을 수 없습니다. JPEG 또는 PNG를 사용하세요.",
    UNKNOWN_VIDEO: "존재하지 않는 영상 ID가 있습니다.",
    UNKNOWN_GALLERY: "존재하지 않는 데이터셋입니다.",
    UNKNOWN_MEDIA: "요청한 이미지·영상 파일을 찾을 수 없습니다.",
    INDEX_NOT_READY: "검색 서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요.",
    INTERNAL_ERROR: "서버 내부 오류가 발생했습니다.",
  };

  // ---------- 탭 ----------
  function setMode(m) {
    mode = m;
    const t = m === "text";
    el.tabText.setAttribute("aria-selected", String(t));
    el.tabImage.setAttribute("aria-selected", String(!t));
    el.textBox.hidden = !t;
    el.imageBox.hidden = t;
  }
  el.tabText.onclick = () => setMode("text");
  el.tabImage.onclick = () => setMode("image");

  // ---------- 이미지 검사 ----------
  function readImageSize(file) {
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = () => { res({ w: im.naturalWidth, h: im.naturalHeight, url }); };
      im.onerror = () => { URL.revokeObjectURL(url); rej(new Error("decode")); };
      im.src = url;
    });
  }
  async function checkImage(file) {
    if (!file) return "이미지를 선택하세요.";
    if (!/^image\/(jpeg|png)$/.test(file.type)) return "JPEG 또는 PNG 이미지만 사용할 수 있습니다.";
    if (file.size > C.image_max_bytes) return "이미지가 5MB를 넘습니다.";
    let d;
    try { d = await readImageSize(file); } catch { return "이미지를 읽을 수 없습니다(손상된 파일)."; }
    if (d.w < C.image_min_side_px || d.h < C.image_min_side_px)
      return `이미지는 가로·세로 ${C.image_min_side_px}px 이상이어야 합니다. (현재 ${d.w}×${d.h})`;
    if (d.w * d.h > C.image_max_pixels) return "이미지가 1,600만 화소를 넘습니다.";
    return { url: d.url };
  }
  el.image.onchange = async () => {
    el.imagePreview.hidden = true;
    const f = el.image.files[0];
    if (!f) return;
    const r = await checkImage(f);
    if (typeof r === "string") { setStatus("error", r); return; }
    el.imagePreview.src = r.url; el.imagePreview.hidden = false;
    setStatus("", "");
  };

  // ---------- 요청 ----------
  function parseVideoIds() {
    const s = el.videoIds.value.trim();
    if (!s) return undefined;                      // 생략 = 전체 gallery
    return s.split(/[\s,]+/).filter(Boolean);
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function mockFetch(text) {
    const t = (text || "").toLowerCase();
    await sleep(t.includes("slow") ? 3000 : C.mock_delay_ms);
    const errs = { "404": [404, "UNKNOWN_VIDEO"], "422": [422, "INVALID_INPUT"], "503": [503, "INDEX_NOT_READY"] };
    for (const k in errs) if (t.includes(k))
      return { status: errs[k][0], body: { request_id: "reqE", error: { code: errs[k][1], message: "mock error" } } };
    const f = t.includes("empty") ? "mock/search_empty.json" : "mock/search_ok.json";
    const r = await fetch(f);
    return { status: 200, body: await r.json() };
  }

  async function realFetch(payload, file, retrievalOnly = false) {
    let r;
    if (payload.modality === "text") {
      r = await fetch(C.apiBase + "/search" + (retrievalOnly ? "?retrieval_only=true" : ""), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } else {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("gallery_id", payload.gallery_id);
      fd.append("top_k", String(payload.top_k));
      fd.append("representation", payload.representation);
      if (payload.search_version) fd.append("search_version", payload.search_version);
      if (payload.video_ids) fd.append("video_ids", JSON.stringify(payload.video_ids));
      r = await fetch(C.apiBase + "/search/image", { method: "POST", body: fd });
    }
    let body = null;
    try { body = await r.json(); } catch { /* 본문이 JSON이 아님 */ }
    return { status: r.status, body };
  }

  function clearPlayback() {
    current = null;
    stopAt = null;
    measuring = false;
    el.video.pause();
    el.video.onloadedmetadata = el.video.onerror = null;
    el.video.removeAttribute("src");
    el.video.load();
    el.playSegment.disabled = true;
    el.info.replaceChildren();
    drawBox();
    showStageMsg("선택된 검색 결과가 없습니다.");
  }

  function setBusy(busy) {
    el.submit.disabled = el.compare.disabled = busy;
  }

  function chooseVersion(version) {
    const pixel = version === "pixel";
    if (pixel && !el.representation.disabled) {
      priorRepresentation = el.representation.value;
      el.representation.value = "scene";
    } else if (!pixel && el.representation.disabled) {
      el.representation.value = priorRepresentation;
    }
    searchVersion = version;
    el.representation.disabled = pixel;
    for (const button of versionButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.version === version));
    }
    turns.length = 0;
    $("chat").replaceChildren();
    if (current) clearPlayback();
  }

  if (C.search_versions) {
    el.versionGroup.hidden = el.compare.hidden = false;
    for (const button of versionButtons) button.onclick = () => {
      if (searchVersion === button.dataset.version) return;
      reqCounter += 1;
      setBusy(false);
      chooseVersion(button.dataset.version);
      el.comparison.hidden = true;
      el.results.replaceChildren();
      el.meta.textContent = "";
      setStatus("", "");
    };
  }

  function describeIndex(stats) {
    if (stats.windows !== undefined) return `4초 구간 ${stats.windows} · 임베딩 ${stats.embeddings}`;
    const tracks = stats.track_events === undefined ? "" : `트랙 이벤트 ${stats.track_events} · `;
    const objects = stats.object_embeddings ? ` · 객체 ${stats.object_embeddings}` : "";
    return `${tracks}임베딩 ${stats.embeddings} (장면 ${stats.scene_embeddings}${objects})`;
  }

  async function compareVersions(payload, file, request) {
    const rows = [];
    for (const button of versionButtons) {
      const version = button.dataset.version;
      const label = button.textContent;
      setStatus("loading", `${label} 검색 중…`);
      const result = await realFetch({...payload, search_version: version, representation: "combined"}, file, true);
      if (request !== reqCounter) return;
      rows.push({version, label, ...result});
    }
    const body = el.comparison.tBodies[0];
    body.replaceChildren();
    for (const {version, label, status, body: result} of rows) {
      const row = body.insertRow();
      const versionCell = row.insertCell();
      versionCell.textContent = label;
      if (status !== 200 || !result) {
        const message = row.insertCell();
        message.colSpan = 3;
        message.textContent = typeof result?.detail === "string" ? result.detail : `검색 실패 (HTTP ${status})`;
        continue;
      }
      if (result.index_stats) {
        const summary = document.createElement("small");
        summary.textContent = describeIndex(result.index_stats);
        versionCell.append(summary);
      }
      row.insertCell().textContent = result.total_candidates;
      row.insertCell().textContent = `${result.retrieval_ms} ms`;
      const view = document.createElement("button");
      view.type = "button";
      view.textContent = `${result.results.length}건 보기`;
      view.disabled = result.results.length === 0;
      view.onclick = () => {
        chooseVersion(version);
        render(result);
      };
      row.insertCell().append(view);
    }
    el.comparison.hidden = false;
    setStatus("ok", "버전 비교 완료");
  }

  el.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const my = ++reqCounter;
    const comparing = C.search_versions && e.submitter === el.compare;
    // 입력 검사
    const topK = Number(el.topK.value);
    if (!Number.isInteger(topK) || topK < 1 || topK > C.top_k_max)
      return setStatus("error", `top_k는 1~${C.top_k_max} 사이의 정수여야 합니다.`);
    const payload = {
      modality: mode, gallery_id: el.gallery.value, top_k: topK,
      representation: el.representation.value,
    };
    if (C.search_versions) payload.search_version = searchVersion;
    const vids = parseVideoIds();
    if (vids) payload.video_ids = vids;
    let file = null, text = "";
    if (mode === "text") {
      text = el.text.value.trim();
      if (text.length < 1) return setStatus("error", "검색어를 입력하세요.");
      if (text.length > C.text_max_len) return setStatus("error", `검색어는 ${C.text_max_len}자 이하여야 합니다. (현재 ${text.length}자)`);
      payload.text = text;
      payload.history = turns.slice(-4);
    } else {
      file = el.image.files[0];
      const r = await checkImage(file);
      if (typeof r === "string") return setStatus("error", r);
    }

    el.comparison.hidden = true;
    if (comparing || (C.default_gallery === "gallery_all" && current)) {
      if (current) clearPlayback();
      el.results.replaceChildren();
      el.meta.textContent = "";
    }
    setBusy(true);
    setStatus("loading", payload.search_version === "pixel" ? "픽셀 구간 준비 · 검색 중…" : "검색 중…");
    let res;
    try {
      if (comparing) {
        await compareVersions(payload, file, my);
      } else {
        res = C.mock ? await mockFetch(text) : await realFetch(payload, file);
      }
    } catch (err) {
      if (my !== reqCounter) return;
      setBusy(false);
      return setStatus("error", "서버에 연결할 수 없습니다. 검색 서버가 켜져 있는지 확인하세요.");
    }
    if (my !== reqCounter) return;               // 오래된 응답 버림
    setBusy(false);
    if (comparing) return;

    if (res.status !== 200) {
      const code = res.body && res.body.error && res.body.error.code;
      const rid = res.body && res.body.request_id;
      el.results.innerHTML = ""; el.meta.textContent = "";
      const detail = res.body && res.body.detail;
      return setStatus("error", `${typeof detail === "string" ? detail : ERR_KO[code] || "오류가 발생했습니다."} (HTTP ${res.status}${code ? ", " + code : ""}${rid ? ", " + rid : ""})`);
    }
    if (res.body.answer) {
      addTurn("user", text);
      if (res.body.llm_status === "generated") addTurn("assistant", res.body.answer);
      if (!C.search_versions) el.text.value = "";
    }
    render(res.body);
  });

  // ---------- 결과 ----------
  const fmt = (ms) => (ms / 1000).toFixed(2) + "s";
  function render(b) {
    el.results.innerHTML = "";
    const label = C.search_versions ? `${versionButtons.find(button => button.dataset.version === searchVersion).textContent} · ` : "";
    const elapsed = b.retrieval_ms === undefined ? `${b.latency_ms}ms` : `검색 ${b.retrieval_ms}ms`;
    const stats = b.index_stats ? ` · ${describeIndex(b.index_stats)}` : "";
    el.meta.textContent = `${label}request ${b.request_id} · index ${b.index_version} · 후보 ${b.total_candidates} · ${elapsed}${stats}`;
    if (!b.results.length) { setStatus("empty", "검색 결과가 없습니다."); return; }
    setStatus("ok", `${b.results.length}건`);
    b.results.forEach((r) => {
      const li = document.createElement("li");
      li.tabIndex = 0;
      li.setAttribute("aria-selected", "false");
      const th = document.createElement("div");
      th.className = "thumb"; th.textContent = "썸네일";
      if (!C.mock && r.thumbnail_url) {
        const im = document.createElement("img");
        im.className = "thumb"; im.alt = "";
        im.src = C.apiBase + r.thumbnail_url;
        im.onerror = () => im.replaceWith(th);
        li.appendChild(im);
      } else li.appendChild(th);
      const d = document.createElement("div");
      d.innerHTML = "";
      const a = document.createElement("div"); a.className = "t1";
      a.textContent = `#${r.rank}  ${r.display_name || r.video_id}  ${fmt(r.representative_ms - r.source_offset_ms)}`;
      const s = document.createElement("div"); s.className = "t2";
      const rankScore = Number.isFinite(r.ranking_score) && r.ranking_score !== r.score
        ? `순위 점수 ${r.ranking_score.toFixed(3)} · ` : "";
      s.textContent = `${rankScore}유사도 ${Number(r.score).toFixed(3)} · ${r.track_uid || "전체 장면"}`;
      d.append(a, s); li.appendChild(d);
      const pick = () => { [...el.results.children].forEach((x) => x.setAttribute("aria-selected", "false")); li.setAttribute("aria-selected", "true"); select(r, b); };
      li.onclick = pick;
      li.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } };
      el.results.appendChild(li);
      if (r.rank === 1 && C.default_gallery === "gallery_all") pick();
    });
  }

  // ---------- 원본 재생 ----------
  function videoSrc(r) { return C.mock ? mockVideoUrl : C.apiBase + r.video_url; }

  function select(r, b) {
    current = r; stopAt = null; lastSeekErrorMs = null;
    el.stageEmpty.hidden = true;
    el.playSegment.disabled = false;
    const src = videoSrc(r);
    if (C.mock && !src) { showStageMsg("로컬 영상 선택 버튼으로 mp4 파일을 먼저 고르세요."); showInfo(r, b); return; }
    el.video.onloadedmetadata = () => seekTo(r);
    el.video.onerror = () => {
      const e = el.video.error, code = e ? ` (오류 코드 ${e.code}${e.code === 4 ? ": 형식·코덱 미지원" : ""})` : "";
      showStageMsg((C.mock
        ? "이 영상은 브라우저에서 재생할 수 없습니다. H.264 mp4로 변환해서 다시 고르세요."
        : "원본 영상을 불러오지 못했습니다.") + code);
    };
    if (el.video.getAttribute("src") === src && el.video.readyState >= 1) seekTo(r);
    else el.video.src = src;
    showInfo(r, b);
  }

  let measuring = false;      // 대표 시각으로 seek한 경우에만 오차를 잰다
  function showStageMsg(m) { el.stageEmpty.textContent = m; el.stageEmpty.hidden = false; }
  function seekTo(r) {
    measuring = true;
    const t = Math.max(0, (r.representative_ms - r.source_offset_ms) / 1000);
    el.video.currentTime = t;
  }
  el.video.addEventListener("seeked", () => {
    if (!current || !measuring) return;
    measuring = false;
    const want = current.representative_ms - current.source_offset_ms;
    lastSeekErrorMs = Math.round(Math.abs(el.video.currentTime * 1000 - want));
    drawBox(); showInfo(current);
  });

  el.playSegment.onclick = () => {
    if (!current) return;
    measuring = false;
    el.video.currentTime = Math.max(0, (current.start_ms - current.source_offset_ms) / 1000);
    stopAt = (current.end_ms - current.source_offset_ms) / 1000;
    el.video.play();
  };
  // timeupdate는 약 250ms 간격이라 end_ms를 넘겨서 멈춘다. 프레임마다 확인해 오차를 줄인다.
  function watchStop() {
    if (stopAt === null) return;
    if (el.video.currentTime >= stopAt) { el.video.pause(); el.video.currentTime = stopAt; stopAt = null; return; }
    requestAnimationFrame(watchStop);
  }
  el.video.addEventListener("play", () => { if (stopAt !== null) requestAnimationFrame(watchStop); });
  el.video.addEventListener("ended", () => { stopAt = null; });

  el.mockVideo.onchange = () => {
    const f = el.mockVideo.files[0];
    if (!f) return;
    mockVideoUrl = URL.createObjectURL(f);
    if (current) select(current);
  };

  // ---------- 박스 그리기 ----------
  // object-fit: contain 이므로 scale = min(viewW/origW, viewH/origH), 남는 여백만큼 offset
  function boxTransform(r) {
    const vw = el.stage.clientWidth, vh = el.stage.clientHeight;
    const scale = Math.min(vw / r.width, vh / r.height);
    const ox = (vw - r.width * scale) / 2, oy = (vh - r.height * scale) / 2;
    const [x1, y1, x2, y2] = r.bbox;
    return { vw, vh, scale, ox, oy, x: ox + x1 * scale, y: oy + y1 * scale, w: (x2 - x1) * scale, h: (y2 - y1) * scale };
  }
  function drawBox() {
    const cv = el.overlay, dpr = window.devicePixelRatio || 1;
    const vw = el.stage.clientWidth, vh = el.stage.clientHeight;
    cv.width = Math.round(vw * dpr); cv.height = Math.round(vh * dpr);
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, vw, vh);
    if (!current || !current.bbox) return;
    const t = boxTransform(current);
    g.lineWidth = 3; g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--box").trim() || "#ffcc00";
    g.strokeRect(t.x, t.y, t.w, t.h);
    window.__lastBox = { x: t.x, y: t.y, w: t.w, h: t.h, scale: t.scale, ox: t.ox, oy: t.oy };
  }
  new ResizeObserver(drawBox).observe(el.stage);

  function showInfo(r, b) {
    const rows = [
      ["video_id", r.video_id], ["track_uid", r.track_uid],
      ["구간", `${r.start_ms}–${r.end_ms} ms`],
      ["대표 시각", `${r.representative_ms} ms (source_offset ${r.source_offset_ms} ms)`],
      ["원본 영역", `${r.bbox ? r.bbox.join(", ") : "전체 장면"} / ${r.width}×${r.height}`],
    ];
    if (r.bbox && window.__lastBox) {
      const t = window.__lastBox;
      rows.push(["화면 박스", `x ${t.x.toFixed(1)}, y ${t.y.toFixed(1)}, ${t.w.toFixed(1)}×${t.h.toFixed(1)} (scale ${t.scale.toFixed(4)})`]);
    }
    el.info.innerHTML = "";
    rows.forEach(([k, v]) => {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      el.info.append(dt, dd);
    });
    if (lastSeekErrorMs !== null) {
      const ok = lastSeekErrorMs <= C.seek_tolerance_ms;
      const dt = document.createElement("dt"); dt.textContent = "seek 오차";
      const dd = document.createElement("dd");
      dd.className = ok ? "pass" : "fail";
      dd.textContent = `${lastSeekErrorMs} ms ${ok ? "PASS" : "FAIL"} (허용 ${C.seek_tolerance_ms} ms)`;
      el.info.append(dt, dd);
    }
  }

  // 점검용 훅 (Playwright 등에서 사용)
  window.__ui = { boxTransform, get seekErrorMs() { return lastSeekErrorMs; } };
})();
