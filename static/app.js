const fmt = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value;
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
};

const formatCompact = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("id-ID", {
    notation: "compact",
    compactDisplay: "short",
    maximumFractionDigits: 2,
  }).format(value);
};

const statusBadge = (status) => {
  const label = status === "signal" ? "SIGNAL" : "WATCH";
  const cls = status === "signal" ? "badge badge-signal" : "badge badge-watch";
  return `<span class="${cls}">${label}</span>`;
};

const aiBadge = (ai) => {
  if (!ai || !ai.bias) return "-";
  const conf = ai.confidence ? ` ${ai.confidence}/10` : "";
  return `<span class="ai-pill">${ai.bias}${conf}</span>`;
};

const aiReport = (ai) => {
  if (!ai) return "-";
  const summary = ai.summary || ai.setup || ai.notes || "";
  const action = ai.action ? ` | ${ai.action}` : "";
  const text = `${summary}${action}`.trim();
  return text
    ? `<div class="ai-report" data-full="${text}">
         <span class="ai-report-text">${text}</span>
         <button class="ai-detail-btn" type="button">Detail</button>
       </div>`
    : "-";
};

const buildTable = (items, kind) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada sinyal sesuai rule. Coba beberapa saat lagi.</div>';
  }
  const header = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Score</th>
            <th>Ticker</th>
            <th>Close</th>
            <th>Change%</th>
            <th>RSI</th>
            <th>Vol</th>
            <th>Entry</th>
            <th>Harga Now</th>
            <th>P/L%</th>
            <th>Entry Time</th>
            <th>Tx Value</th>
            <th>SL</th>
            <th>TP1</th>
            <th>TP2</th>
            <th>TP3</th>
            <th>AI</th>
            <th>AI Report</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr data-ticker="${fmt(item.ticker)}" data-kind="${kind}">
        <td>${statusBadge(item.status)}</td>
        <td>${fmt(item.score)}</td>
        <td>
          <a class="tv-link" href="https://www.tradingview.com/symbols/IDX-${fmt(item.ticker)}/" target="_blank" rel="noreferrer">
            ${fmt(item.ticker)}
          </a>
        </td>
        <td>${formatNumber(item.close, 2)}</td>
        <td>${formatNumber(item.change_pct, 2)}</td>
        <td>${fmt(item.rsi)}</td>
        <td>${formatNumber(item.vol_spike, 2)}</td>
        <td>${formatNumber(item.entry_plan, 2)}</td>
        <td>${formatNumber(item.entry_now, 2)}</td>
        <td>${formatNumber(item.pnl_pct, 2)}</td>
        <td>${fmt(item.entry_plan_at)}</td>
        <td>${formatCompact(item.tx_value)}</td>
        <td>${formatNumber(item.sl, 2)}</td>
        <td>${formatNumber(item.tp1, 2)}</td>
        <td>${formatNumber(item.tp2, 2)}</td>
        <td>${formatNumber(item.tp3, 2)}</td>
        <td>${aiBadge(item.ai)}</td>
        <td>${aiReport(item.ai)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

const buildCorporateTable = (items) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada berita corporate action.</div>';
  }
  const cards = items.slice(0, 10).map((item) => {
    const image = item.image
      ? `<div class="ca-image" style="background-image: url('${item.image}')"></div>`
      : `<div class="ca-image ca-placeholder"></div>`;
    const title = item.url
      ? `<a href="${item.url}" target="_blank" rel="noreferrer">${fmt(item.title)}</a>`
      : fmt(item.title);
    return `
      <article class="ca-card">
        ${image}
        <div class="ca-body">
          <div class="ca-tag">${fmt(item.tag)}</div>
          <h3>${title}</h3>
          <div class="ca-meta">${fmt(item.date)} • ${fmt(item.source)}</div>
          <div class="ca-ai">
            ${aiBadge(item.ai)}
            ${aiReport(item.ai)}
          </div>
        </div>
      </article>
    `;
  }).join("");
  return `<div class="ca-grid">${cards}</div>`;
};

const renderFundamentals = (data) => {
  const ratios = data.ratios || {};
  const inputs = data.inputs || {};
  document.getElementById("fundamental-selected").textContent = data.ticker || "-";
  document.getElementById("fundamental-asof").textContent = data.as_of || "-";

  const rows = [
    ["Market Cap", formatCompact(ratios.market_cap)],
    ["PBV", formatNumber(ratios.pbv, 2)],
    ["BVPS", formatNumber(ratios.bvps, 2)],
    ["PER", formatNumber(ratios.per, 2)],
    ["EPS", formatNumber(ratios.eps, 2)],
    ["Net Profit Margin (%)", formatNumber(ratios.net_profit_margin, 2)],
    ["DER", formatNumber(ratios.der, 2)],
    ["ROE (%)", formatNumber(ratios.roe, 2)],
    ["Nilai Wajar (Market Cap)", formatCompact(ratios.fair_value_market_cap)],
    ["Harga Wajar (Target MC)", formatNumber(ratios.fair_price_target_mc, 2)],
    ["PBV Band", ratios.pbv_band],
  ]
    .map(([label, value]) => `
      <div class="fund-row">
        <span>${label}</span>
        <span>${fmt(value)}</span>
      </div>
    `)
    .join("");

  const inputsHtml = `
    <div class="fund-subtitle">Input Data</div>
    <div class="fund-row"><span>Price</span><span>${formatNumber(inputs.price, 2)}</span></div>
    <div class="fund-row"><span>Shares Outstanding</span><span>${formatCompact(inputs.shares_outstanding)}</span></div>
    <div class="fund-row"><span>Net Income</span><span>${formatCompact(inputs.net_income)}</span></div>
    <div class="fund-row"><span>Revenue</span><span>${formatCompact(inputs.revenue)}</span></div>
    <div class="fund-row"><span>Equity</span><span>${formatCompact(inputs.equity)}</span></div>
    <div class="fund-row"><span>Liabilities</span><span>${formatCompact(inputs.liabilities)}</span></div>
    <div class="fund-row"><span>PE Wajar</span><span>${formatNumber(inputs.pe_wajar, 2)}</span></div>
    <div class="fund-row"><span>Target Market Cap</span><span>${formatCompact(inputs.target_market_cap)}</span></div>
  `;

  document.getElementById("fundamental-body").innerHTML = rows + inputsHtml;
  const ai = data.ai;
  const aiText = ai
    ? `AI: ${ai.bias || "-"} (${ai.confidence || "-"}/10) ${ai.summary || ""} ${ai.action || ""} ${ai.risk || ""}`
    : "";
  document.getElementById("fundamental-note").textContent =
    [data.note || "", aiText].filter(Boolean).join(" ");
};

const buildFundamentalUrl = (ticker) => {
  const pe = document.getElementById("pe-wajar").value;
  const target = document.getElementById("target-market-cap").value;
  const params = new URLSearchParams();
  if (pe) params.set("pe_wajar", pe);
  if (target) params.set("target_market_cap", target);
  const qs = params.toString();
  return `/api/fundamentals/${ticker}${qs ? `?${qs}` : ""}`;
};

const loadFundamentals = async (ticker) => {
  if (!ticker) return;
  document.getElementById("fundamental-selected").textContent = ticker;
  document.getElementById("fundamental-body").innerHTML = '<div class="empty">Loading...</div>';
  try {
    const url = buildFundamentalUrl(ticker);
    const data = await fetch(url).then((r) => r.json());
    renderFundamentals(data);
  } catch (err) {
    document.getElementById("fundamental-body").innerHTML =
      '<div class="empty">Gagal mengambil data fundamental.</div>';
  }
};

const bindTableClicks = () => {
  ["scalping-table", "swing-table"].forEach((id) => {
    const container = document.getElementById(id);
    if (!container) return;
    container.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      const row = event.target.closest("tr[data-ticker]");
      if (!row) return;
      const ticker = row.getAttribute("data-ticker");
      loadFundamentals(ticker);
    });
  });
};

const bindFundControls = () => {
  const btn = document.getElementById("fund-refresh");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const ticker = document.getElementById("fundamental-selected").textContent;
    if (ticker && ticker !== "-") {
      loadFundamentals(ticker);
    }
  });
};

const initTheme = () => {
  const saved = localStorage.getItem("theme");
  if (saved) {
    document.body.setAttribute("data-theme", saved);
  }
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme") || "light";
    const next = current === "dark" ? "light" : "dark";
    document.body.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  });
};

const buildDailyTable = (items) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada hasil screener.</div>';
  }
  const header = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Close</th>
            <th>Change%</th>
            <th>Volume</th>
            <th>Value</th>
            <th>AI</th>
            <th>AI Report</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr>
        <td>
          <a class="tv-link" href="https://www.tradingview.com/symbols/IDX-${fmt(item.ticker)}/" target="_blank" rel="noreferrer">
            ${fmt(item.ticker)}
          </a>
        </td>
        <td>${formatNumber(item.close, 2)}</td>
        <td>${formatNumber(item.change_pct, 2)}</td>
        <td>${formatCompact(item.volume)}</td>
        <td>${formatCompact(item.tx_value)}</td>
        <td>${aiBadge(item.ai)}</td>
        <td>${aiReport(item.ai)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

const buildReview = (review) => {
  if (!review) return '<div class="empty">Belum ada review sesi.</div>';
  const summary = review.summary || review.setup || review.notes || "";
  const action = review.action || "";
  const risk = review.risk || "";
  const bias = review.bias || "";
  const market = review.market_behavior || "";
  const worked = review.what_worked || "";
  const failed = review.what_failed || "";
  const conf = review.confidence ? `${review.confidence}/10` : "";
  return `
    <div class="review-card">
      <div class="review-title">AI Review ${bias} ${conf}</div>
      <div class="review-text">${summary}</div>
      <div class="review-text">${market}</div>
      <div class="review-text">${worked}</div>
      <div class="review-text">${failed}</div>
      <div class="review-text">${action}</div>
      <div class="review-text">${risk}</div>
    </div>
  `;
};

async function refresh() {
  const [scalping, swing, corporate, bsjp, bpjs] = await Promise.all([
    fetch("/api/scalping").then((r) => r.json()),
    fetch("/api/swing").then((r) => r.json()),
    fetch("/api/corporate-actions").then((r) => r.json()),
    fetch("/api/bsjp").then((r) => r.json()),
    fetch("/api/bpjs").then((r) => r.json()),
  ]);

  fetch("/api/health")
    .then((r) => r.json())
    .then((health) => {
      const status = health.market_open ? "Open" : "Closed";
      const session = health.market_status || "-";
      const now = health.now || "-";
      document.querySelectorAll(".market-status").forEach((el) => {
        el.textContent = `Market: ${status} (${session}) - ${now}`;
      });
    })
    .catch(() => {});

  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  const setHtml = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = value;
  };

  setText("scalping-updated", scalping.updated_at || "-");
  if (scalping.stats) {
    const lossRate = scalping.stats.loss_rate ?? null;
    const winRate = lossRate !== null ? Math.max(0, 1 - lossRate) : null;
    setText("scalping-lossrate", lossRate === null ? "-" : `${formatNumber(lossRate * 100, 1)}%`);
    setText("scalping-winrate", winRate === null ? "-" : `${formatNumber(winRate * 100, 1)}%`);
    setText("scalping-mode", scalping.stats.tighten ? "Ketat" : "Normal");
  }
  setText("swing-updated", swing.updated_at || "-");
  setText("scalping-error", scalping.error || "");
  setText("swing-error", swing.error || "");
  setHtml("scalping-table", buildTable(scalping.items, "scalping"));
  if (scalping.review) {
    let reviewHtml = buildReview(scalping.review.text);
    if (scalping.adjust) {
      const adj = scalping.adjust;
      const adjLine = `Auto-Adjust: score ${adj.score}, rsi ${adj.rsi_min}/${adj.rsi_max}, vol ${adj.vol_spike}, tx ${formatCompact(adj.tx_value)}`;
      reviewHtml += `<div class="review-text">${adjLine}</div>`;
    }
    setHtml("scalping-review", reviewHtml);
    setText("scalping-review-at", scalping.review.at || "-");
  }
  setHtml("swing-table", buildTable(swing.items, "swing"));

  setText("ca-updated", corporate.updated_at || "-");
  setText("ca-error", corporate.error || "");
  setHtml("ca-table", buildCorporateTable(corporate.items));

  setText("bsjp-updated", bsjp.updated_at || "-");
  setText("bsjp-error", bsjp.error || "");
  setHtml("bsjp-table", buildDailyTable(bsjp.items));

  setText("bpjs-updated", bpjs.updated_at || "-");
  setText("bpjs-error", bpjs.error || "");
  setHtml("bpjs-table", buildDailyTable(bpjs.items));
}

refresh();
bindTableClicks();
bindFundControls();
initTheme();
setInterval(refresh, window.UI_POLL_SECONDS ? window.UI_POLL_SECONDS * 1000 : 1000);

const openModal = (title, content) => {
  const modal = document.getElementById("ai-modal");
  if (!modal) return;
  modal.querySelector(".ai-modal-title").textContent = title || "AI Report";
  modal.querySelector(".ai-modal-body").textContent = content || "-";
  modal.classList.add("open");
};

const closeModal = () => {
  const modal = document.getElementById("ai-modal");
  if (!modal) return;
  modal.classList.remove("open");
};

document.addEventListener("click", (event) => {
  if (event.target.matches(".ai-detail-btn")) {
    const wrap = event.target.closest(".ai-report");
    const text = wrap?.getAttribute("data-full") || "-";
    openModal("AI Report", text);
    return;
  }
  if (event.target.matches(".ai-modal-overlay") || event.target.matches(".ai-modal-close")) {
    closeModal();
  }
});
