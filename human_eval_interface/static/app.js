(() => {
  const state = {
    evaluator: null,
    scope: null,
    current: null,        // {image_id, url}
    shownAt: 0,
    busy: false,
  };

  const $ = (id) => document.getElementById(id);

  const screens = {
    login: $("login-screen"),
    eval: $("eval-screen"),
    done: $("done-screen"),
  };

  function show(name) {
    for (const k of Object.keys(screens)) {
      screens[k].hidden = (k !== name);
    }
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || `HTTP ${res.status}`);
      err.payload = data;
      throw err;
    }
    return data;
  }

  function updateProgress(p) {
    if (!p) return;
    $("hdr-done").textContent = p.evaluated;
    $("hdr-total").textContent = p.total;
  }

  async function startSession() {
    const evaluator = $("evaluator").value.trim();
    const scope = $("scope").value;
    $("login-error").hidden = true;
    if (!evaluator) {
      $("login-error").textContent = "Please enter an evaluator name.";
      $("login-error").hidden = false;
      return;
    }
    try {
      const data = await api("/api/session", { evaluator, scope });
      state.evaluator = data.evaluator;
      state.scope = data.scope;
      $("hdr-evaluator").textContent = state.evaluator;
      $("hdr-scope").textContent = state.scope;
      updateProgress(data.progress);
      show("eval");
      await loadNext();
    } catch (e) {
      $("login-error").textContent = e.message;
      $("login-error").hidden = false;
    }
  }

  async function loadNext() {
    state.busy = true;
    $("eval-error").hidden = true;
    try {
      const data = await api("/api/next", {
        evaluator: state.evaluator,
        scope: state.scope,
      });
      updateProgress(data.progress);
      if (data.done) {
        renderSummary(data.summary);
        show("done");
        return;
      }
      state.current = { image_id: data.image_id, url: data.url };
      const img = $("current-img");
      img.onload = () => {
        state.shownAt = performance.now();
        state.busy = false;
      };
      img.onerror = () => {
        $("eval-error").textContent = "Failed to load image: " + data.url;
        $("eval-error").hidden = false;
        state.busy = false;
      };
      img.src = data.url;
    } catch (e) {
      $("eval-error").textContent = e.message;
      $("eval-error").hidden = false;
      state.busy = false;
    }
  }

  async function submit(choice) {
    if (state.busy || !state.current) return;
    state.busy = true;
    const elapsed_ms = Math.max(0, Math.round(performance.now() - state.shownAt));
    try {
      const data = await api("/api/submit", {
        evaluator: state.evaluator,
        scope: state.scope,
        image_id: state.current.image_id,
        choice,
        elapsed_ms,
      });
      updateProgress(data.progress);
      await loadNext();
    } catch (e) {
      $("eval-error").textContent = e.message;
      $("eval-error").hidden = false;
      state.busy = false;
    }
  }

  async function undo() {
    if (state.busy) return;
    state.busy = true;
    try {
      const data = await api("/api/undo", {
        evaluator: state.evaluator,
        scope: state.scope,
      });
      updateProgress(data.progress);
      await loadNext();
    } catch (e) {
      $("eval-error").textContent = e.message;
      $("eval-error").hidden = false;
      state.busy = false;
    }
  }

  function renderSummary(s) {
    if (!s) return;
    $("summary-path").textContent =
      `human_eval_interface/Results/${state.evaluator}/${state.scope}/summary.json`;

    const rows = [];
    rows.push(`<div class="row head"><span>Bucket / Model</span><span>N · Accuracy</span></div>`);
    for (const b of (s.per_bucket || [])) {
      const tag = b.gen_model ? `DeepFake/${b.gen_model}` : b.bucket;
      const acc = (b.accuracy == null) ? "—" : (100 * b.accuracy).toFixed(1) + "%";
      rows.push(`<div class="row"><span>${tag}</span><span>${b.n} · ${acc}</span></div>`);
    }
    const total = s.evaluated || 0;
    const accAll = (s.accuracy == null) ? "—" : (100 * s.accuracy).toFixed(1) + "%";
    rows.push(`<div class="row head"><span>Total</span><span>${total} · ${accAll}</span></div>`);
    $("summary-body").innerHTML = rows.join("");
  }

  // Event wiring
  $("start-btn").addEventListener("click", startSession);
  $("evaluator").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startSession();
  });
  $("btn-real").addEventListener("click", () => submit("real"));
  $("btn-fake").addEventListener("click", () => submit("fake"));
  $("btn-undo").addEventListener("click", undo);
  $("restart-btn").addEventListener("click", () => location.reload());

  document.addEventListener("keydown", (e) => {
    if (screens.eval.hidden) return;
    if (e.repeat) return;
    const k = e.key.toLowerCase();
    if (k === "f" || e.key === "ArrowLeft") submit("real");
    else if (k === "j" || e.key === "ArrowRight") submit("fake");
    else if (k === "u") undo();
  });
})();
