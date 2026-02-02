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
            <th>Entry0</th>
            <th>Chg Entry%</th>
            <th>Tx Value</th>
            <th>Entry</th>
            <th>SL</th>
            <th>TP1</th>
            <th>TP2</th>
            <th>TP3</th>
            <th>AI</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr data-ticker="${fmt(item.ticker)}" data-kind="${kind}">
        <td>${statusBadge(item.status)}</td>
        <td>${fmt(item.score)}</td>
        <td>${fmt(item.ticker)}</td>
        <td>${formatNumber(item.close, 2)}</td>
        <td>${formatNumber(item.change_pct, 2)}</td>
        <td>${fmt(item.rsi)}</td>
        <td>${formatNumber(item.vol_spike, 2)}</td>
        <td>${formatNumber(item.entry_base, 2)}</td>
        <td>${formatNumber(item.change_from_entry_pct, 2)}</td>
        <td>${formatCompact(item.tx_value)}</td>
        <td>${formatNumber(item.entry, 2)}</td>
        <td>${formatNumber(item.sl, 2)}</td>
        <td>${formatNumber(item.tp1, 2)}</td>
        <td>${formatNumber(item.tp2, 2)}</td>
        <td>${formatNumber(item.tp3, 2)}</td>
        <td>${aiBadge(item.ai)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

const buildCorporateTable = (items) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada berita corporate action.</div>';
  }
  const header = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Tag</th>
            <th>Perihal</th>
            <th>Tanggal</th>
            <th>Sumber</th>
            <th>AI</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr>
        <td><span class="badge badge-watch">${fmt(item.tag)}</span></td>
        <td>${item.url ? `<a href="${item.url}" target="_blank" rel="noreferrer">${fmt(item.title)}</a>` : fmt(item.title)}</td>
        <td>${fmt(item.date)}</td>
        <td>${fmt(item.source)}</td>
        <td>${aiBadge(item.ai)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
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
    ? `AI: ${ai.bias || "-"} (${ai.confidence || "-"}/10) - ${ai.notes || ""}`
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

async function refresh() {
  const [scalping, swing, corporate] = await Promise.all([
    fetch("/api/scalping").then((r) => r.json()),
    fetch("/api/swing").then((r) => r.json()),
    fetch("/api/corporate-actions").then((r) => r.json()),
  ]);

  document.getElementById("scalping-updated").textContent = scalping.updated_at || "-";
  document.getElementById("swing-updated").textContent = swing.updated_at || "-";
  document.getElementById("scalping-error").textContent = scalping.error || "";
  document.getElementById("swing-error").textContent = swing.error || "";
  document.getElementById("scalping-table").innerHTML = buildTable(scalping.items, "scalping");
  document.getElementById("swing-table").innerHTML = buildTable(swing.items, "swing");

  document.getElementById("ca-updated").textContent = corporate.updated_at || "-";
  document.getElementById("ca-error").textContent = corporate.error || "";
  document.getElementById("ca-table").innerHTML = buildCorporateTable(corporate.items);
}

refresh();
bindTableClicks();
bindFundControls();
setInterval(refresh, window.UI_POLL_SECONDS ? window.UI_POLL_SECONDS * 1000 : 1000);
