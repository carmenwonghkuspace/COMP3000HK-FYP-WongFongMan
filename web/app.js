/**
 * Spam scoring in the browser — TF-IDF (train-time settings) + logistic weights.
<<<<<<< HEAD
 * Model weights load from a fixed path next to this page (see MODEL_JSON_PATH).
 */

/** Fixed path: same directory as index.html (deploy web/model.json alongside this file). */
const MODEL_JSON_PATH = "model.json";

let MODEL = null;

function setAnalyzeEnabled(enabled) {
  const btn = document.getElementById("analyze");
  if (btn) btn.disabled = !enabled;
}

function validateModel(obj) {
  if (!obj || typeof obj !== "object") return "Invalid JSON (expected an object).";
  const need = ["coef", "feature_names", "idf", "intercept", "stop_words"];
  for (const k of need) {
    if (!(k in obj)) return `Missing required field: ${k}`;
  }
  const n = obj.feature_names.length;
  if (obj.coef.length !== n || obj.idf.length !== n) {
    return "Length mismatch: coef, idf, and feature_names must align.";
  }
  return null;
}

function applyModel(obj) {
  const st = document.getElementById("status");
  const err = validateModel(obj);
  if (err) {
    st.textContent = err;
    MODEL = null;
    setAnalyzeEnabled(false);
    return false;
  }
  MODEL = obj;
  st.textContent = "Model ready.";
  setAnalyzeEnabled(true);
  return true;
}

=======
 */

let MODEL = null;

>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8
function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadModel() {
  const st = document.getElementById("status");
<<<<<<< HEAD
  setAnalyzeEnabled(false);
  st.textContent = "Loading model…";
  try {
    const r = await fetch(MODEL_JSON_PATH, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const obj = await r.json();
    if (applyModel(obj)) return true;
  } catch (e) {
    console.error(e);
    st.textContent =
      `Could not load ${MODEL_JSON_PATH}. Serve the web folder over HTTP (e.g. python -m http.server) so the file can be fetched.`;
    MODEL = null;
  }
  return false;
=======
  try {
    const r = await fetch("model.json", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    MODEL = await r.json();
    st.textContent = "Model loaded.";
    return true;
  } catch (e) {
    st.textContent =
      "Could not load model.json — use a local HTTP server from the web/ folder.";
    console.error(e);
    return false;
  }
>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8
}

function cleanText(raw, stopWords) {
  const sw = stopWords instanceof Set ? stopWords : new Set(stopWords);
  let t = raw.toLowerCase();
  t = t.replace(/http\S+|www\.\S+/gi, " url ");
  t = t.replace(/\b\d+\b/g, " num ");
  t = t.replace(/[^a-z\s]/g, " ");
  t = t.replace(/\s+/g, " ").trim();
  return t
    .split(/\s+/g)
    .filter((w) => w.length > 1 && !sw.has(w));
}

/** Build uni / bigram counts from cleaned token sequence */
function gramCounts(tokens) {
  const uni = new Map();
  const bi = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const w = tokens[i];
    uni.set(w, (uni.get(w) || 0) + 1);
    if (i + 1 < tokens.length) {
      const bg = `${w} ${tokens[i + 1]}`;
      bi.set(bg, (bi.get(bg) || 0) + 1);
    }
  }
  return { uni, bi };
}

function countGram(fname, uni, bi, ngramRange) {
  const [ngMin, ngMax] = ngramRange;
  if (fname.includes(" ")) {
    if (ngMax < 2) return 0;
    return bi.get(fname) || 0;
  }
  if (ngMin > 1) return 0;
  return uni.get(fname) || 0;
}

/**
 * TF-IDF row matching sklearn defaults used in training (no max_df trim at predict).
 */
function tfidfDense(tokens, model) {
  const { uni, bi } = gramCounts(tokens);
  const dim = model.feature_names.length;
  const vec = new Float64Array(dim);
  let ngRange = model.ngram_range || [1, 2];
  const sublinear = !!model.sublinear_tf;

  for (let k = 0; k < dim; k++) {
    const fname = model.feature_names[k];
    const c = countGram(fname, uni, bi, ngRange);
    if (c === 0) continue;
    let tf = sublinear ? 1 + Math.log(c) : c;
    vec[k] = tf * model.idf[k];
  }

  let sq = 0;
  for (let k = 0; k < dim; k++) sq += vec[k] * vec[k];
  const inv = sq > 0 ? 1 / Math.sqrt(sq) : 1;
  const norm = new Float64Array(dim);
  for (let k = 0; k < dim; k++) norm[k] = vec[k] * inv;
  return { raw: vec, norm };
}

function sigmoid(z) {
  if (z > 35) return 1;
  if (z < -35) return 0;
  return 1 / (1 + Math.exp(-z));
}

/** Allocate signed logistic contribution onto surface tokens */
function tokenContributions(normVec, model) {
  const dim = model.feature_names.length;
  const acc = new Map();
  const coef = model.coef;

  for (let k = 0; k < dim; k++) {
    const v = normVec[k];
    if (v === 0) continue;
    const w = coef[k] * v;
    if (w === 0) continue;
    const fn = model.feature_names[k];
    const parts = fn.split(" ");
    if (parts.length === 1) {
      bump(acc, parts[0], w);
    } else {
      const half = w / parts.length;
      for (const p of parts) bump(acc, p, half);
    }
  }
  return acc;
}

function bump(acc, token, delta) {
  acc.set(token, (acc.get(token) || 0) + delta);
}

function intensityMap(contribMap) {
  let m = 0;
  contribMap.forEach((v) => {
    const a = Math.abs(v);
    if (a > m) m = a;
  });
  const scale = m > 0 ? 1 / m : 1;
  const out = new Map();
  contribMap.forEach((v, k) => out.set(k, Math.abs(v) * scale));
  return out;
}

function renderTokens(tokens, contribMap, intenMap) {
  const el = document.getElementById("token-view");
  if (!tokens.length) {
<<<<<<< HEAD
    el.innerHTML = `<em class="token-empty">No tokens left after preprocessing (URL/number placeholders and stop-word removal).</em>`;
=======
    el.innerHTML = `<em>(No tokens left after preprocessing — URL/NUM placeholders and stop-word removal)</em>`;
>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8
    return;
  }
  const parts = tokens.map((tok) => {
    const signed = contribMap.get(tok) || 0;
    const intenRaw = intenMap.get(tok) ?? 0;
<<<<<<< HEAD
    const amp = Math.min(0.55, 0.1 + intenRaw * 0.5);
    let bg;
    let border;
    if (signed >= 0) {
      bg = `rgba(220, 38, 38, ${amp.toFixed(3)})`;
      border = `rgba(248, 113, 113, 0.45)`;
    } else {
      bg = `rgba(37, 99, 235, ${amp.toFixed(3)})`;
      border = `rgba(147, 197, 253, 0.5)`;
=======
    const amp = Math.min(0.5, 0.12 + intenRaw * 0.45);
    let bg;
    let border;
    if (signed >= 0) {
      bg = `rgba(239, 68, 68, ${amp.toFixed(3)})`;
      border = `rgba(248, 113, 113, 0.5)`;
    } else {
      bg = `rgba(59, 130, 246, ${amp.toFixed(3)})`;
      border = `rgba(147, 197, 253, 0.45)`;
>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8
    }
    const title = `${tok}: contribution (logit) ${signed >= 0 ? "+" : ""}${signed.toFixed(4)}`;
    const titEsc = escHtml(title).replace(/"/g, "&quot;");
    const sp = `<span class="tok" title="${titEsc}" style="background:${bg};border-color:${border}">${escHtml(tok)}</span>`;
    return sp;
  });
  el.innerHTML = parts.join("");
}

function runAnalyze() {
  if (!MODEL) return;
  const raw = document.getElementById("email").value;
  const tokens = cleanText(raw, new Set(MODEL.stop_words));

  const { norm } = tfidfDense(tokens, MODEL);
  let logit = MODEL.intercept;
  for (let k = 0; k < MODEL.coef.length; k++) logit += MODEL.coef[k] * norm[k];
  const p = sigmoid(logit);
  const contrib = tokenContributions(norm, MODEL);
  const inten = intensityMap(contrib);

  document.getElementById("prob").textContent = p.toFixed(4);
  document.getElementById("meter").style.width = `${(p * 100).toFixed(1)}%`;
  document.getElementById("inf-mode").textContent = MODEL.inference_mode;
  document.getElementById("best-cv").textContent = MODEL.best_model_cv;
  document.getElementById("infer-note").textContent = MODEL.note || "";
<<<<<<< HEAD

  const ph = document.getElementById("output-placeholder");
  const rb = document.getElementById("result-block");
  if (ph) ph.hidden = true;
  if (rb) rb.hidden = false;

=======
  document.getElementById("result-block").hidden = false;
>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8
  renderTokens(tokens, contrib, inten);
}

document.getElementById("analyze").addEventListener("click", runAnalyze);

loadModel();
